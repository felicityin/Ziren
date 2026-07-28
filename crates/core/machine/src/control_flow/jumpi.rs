use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, JumpEvent},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{
        clk_low_expr, eval_cpu_state, eval_j_type_reader, eval_state_chain, CpuState,
        InstructionCols, JTypeReader,
    },
    air::ZKMCoreAirBuilder,
    operations::KoalaBearWordRangeChecker,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `JumpiChip`.
pub const NUM_JUMPI_COLS: usize = size_of::<JumpiCols<u8>>();

/// A chip that implements the MIPS immediate-target jump instructions J/JAL (`Opcode::Jumpi`).
/// J's `op_a` is hardcoded to `$zero` (its return address is discarded); JAL's is always the real
/// link register 31 -- both routed through the same masked adapter, `JTypeReader` (see its doc
/// comment). `op_b` is always the instruction's own encoded jump target, never a register.
///
/// Every row here is a real, retired instruction: J/JAL's target is `op_b`'s own immediate value
/// used directly (`next_next_pc == op_b`, no arithmetic), so unlike `JumpDirectChip` this chip
/// sends no synthetic dependency rows.
#[derive(Default)]
pub struct JumpiChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct JumpiCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current program counter.
    pub pc: T,

    /// Register/immediate operand access for `a`/`b`: `op_a` (written, possibly `$zero`) is the
    /// link register, `op_b` is the instruction's own encoded jump target (never a register).
    pub reader: JTypeReader<T>,

    /// The next program counter (`pc + 4`). There is no separate `next_next_pc` column: the jump
    /// target is `op_b`'s own immediate value, used directly wherever `next_next_pc` is needed.
    pub next_pc: Word<T>,
    pub next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// The link value (`next_pc + 4`) written to `op_a`. Its own witness, separate from
    /// `reader.op_a_access.value`: the addition must be checked at the reduced-scalar level
    /// (`next_pc.reduce() + 4`), not limb-by-limb (a byte-wise `next_pc.0[0] + 4` would silently
    /// drop the carry whenever that byte is >= 252) -- see `eval`.
    pub op_a_value: Word<T>,
    pub op_a_range_checker: KoalaBearWordRangeChecker<T>,

    /// Whether this is a real, retired J/JAL instruction.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for JumpiChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Jumpi".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.jumpi_events.len(),
            None,
            <JumpiChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.jumpi_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <JumpiChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_JUMPI_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_JUMPI_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_JUMPI_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut JumpiCols<F> = row.borrow_mut();

                    if idx < input.jumpi_events.len() {
                        let event = &input.jumpi_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_JUMPI_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.jumpi_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl JumpiChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &JumpEvent,
        cols: &mut JumpiCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);
        cols.is_real = F::ONE;

        let instruction = program.fetch(event.pc);
        cols.pc = F::from_canonical_u32(event.pc);

        cols.reader.populate(blu, instruction.op_a, event.a_record, instruction.op_b);

        cols.next_pc = Word::from(event.next_pc);
        cols.next_pc_range_checker.populate(event.next_pc);
        cols.op_a_value = Word::from(event.a);
        cols.op_a_range_checker.populate(event.a);
    }
}

impl<F> BaseAir<F> for JumpiChip {
    fn width(&self) -> usize {
        NUM_JUMPI_COLS
    }
}

impl<AB> Air<AB> for JumpiChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &JumpiCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The candidate link value: always `next_pc + 4`, masked to zero by `eval_j_type_reader`
        // when `op_a_0` (J, or -- impossible in practice, since JAL's op_a is hardcoded to the
        // real register 31 by the decoder, never a variable field -- a masked JAL). Checked at
        // the reduced-scalar level, not limb-by-limb: a byte-wise `next_pc.0[0] + 4` would
        // silently drop the carry whenever that byte is >= 252.
        builder.when(local.is_real).assert_eq(
            local.op_a_value.reduce::<AB>(),
            local.next_pc.reduce::<AB>() + AB::Expr::from_canonical_u32(4),
        );
        let op_a_computed_value = local.op_a_value.map(Into::into);

        // The instruction word is reconstructed here rather than stored: `opcode` is hardcoded
        // (this chip only ever sees `Opcode::Jumpi`), `imm_b`/`imm_c` are compile-time constants
        // (`op_b` is always the encoded jump target, an immediate; `op_c` is always unused,
        // hardcoded zero), and `op_a`/`op_a_0`/`op_b` come from the adapter.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: AB::Expr::from_canonical_u32(Opcode::Jumpi as u32),
            op_a: local.reader.op_a.into(),
            op_b: local.reader.op_b.map(Into::into),
            op_c: Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            op_a_0: local.reader.op_a_0.into(),
            imm_b: AB::Expr::one(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, local.is_real.into());

        eval_j_type_reader(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            op_a_computed_value,
            local.is_real.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

        // J/JAL's target is `op_b`'s own immediate value, used directly: no arithmetic, so no
        // dependency-chip send is needed here (unlike `JumpDirectChip`'s ADD lookup).
        let next_next_pc = local.reader.op_b.reduce::<AB>();
        eval_state_chain(
            builder,
            clk_high,
            clk_low,
            local.pc.into(),
            local.next_pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            local.is_real.into(),
        );

        // Range check `next_pc` and the value written to `op_a`.
        // SAFETY: `is_real` is already checked to be boolean. `op_b` needs no separate range
        // check here: it's a fixed instruction immediate from the (public, honest) Program table.
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_pc,
            local.next_pc_range_checker,
            local.is_real.into(),
        );
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.op_a_value,
            local.op_a_range_checker,
            local.is_real.into(),
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::JumpEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::JumpiChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::Jumpi,
                op_a: 31,
                op_b: 100,
                op_c: 0,
                imm_b: true,
                imm_c: true,
                raw: None,
            }],
            pc_start: 0,
            pc_base: 0,
            next_pc: 4,
            image: Default::default(),
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.jumpi_events = vec![JumpEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            next_next_pc: 100,
            opcode: Opcode::Jumpi,
            a: 8,
            b: 100,
            c: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = JumpiChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
