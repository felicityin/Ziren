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
        clk_low_expr, eval_cpu_state, eval_r_type_reader_masked, eval_state_chain, CpuState,
        InstructionCols, RTypeReaderMasked,
    },
    air::ZKMCoreAirBuilder,
    operations::KoalaBearWordRangeChecker,
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `JumpChip`.
pub const NUM_JUMP_COLS: usize = size_of::<JumpCols<u8>>();

/// A chip that implements the MIPS register-target jump instructions JR/JALR (`Opcode::Jump`).
/// JR's `op_a` is hardcoded to `$zero` (its return address is discarded); JALR's is a real,
/// possibly-zero link register -- both routed through the same masked adapter,
/// `RTypeReaderMasked` (see its doc comment), rather than splitting further into a dedicated
/// "JumpX0"-style chip.
///
/// Every row here is a real, retired instruction: JR/JALR's target is a register value used
/// directly (`next_next_pc == op_b_val()`, no arithmetic), so unlike `JumpDirectChip` this chip
/// sends no synthetic dependency rows.
#[derive(Default)]
pub struct JumpChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct JumpCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current program counter.
    pub pc: T,

    /// Register operand access for `a`/`b`: `op_a` (written, possibly `$zero`) is the link
    /// register, `op_b` (read) is the jump-target register.
    pub reader: RTypeReaderMasked<T>,

    /// The next program counter (`pc + 4`). There is no separate `next_next_pc` column: the jump
    /// target is `op_b`'s own (already range-checked) register value, used directly wherever
    /// `next_next_pc` is needed.
    pub next_pc: Word<T>,
    pub next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// The link value (`next_pc + 4`) written to `op_a`. Its own witness, separate from
    /// `reader.op_a_access.value`: the addition must be checked at the reduced-scalar level
    /// (`next_pc.reduce() + 4`), not limb-by-limb (a byte-wise `next_pc.0[0] + 4` would silently
    /// drop the carry whenever that byte is >= 252) -- see `eval`.
    pub op_a_value: Word<T>,
    pub op_a_range_checker: KoalaBearWordRangeChecker<T>,

    /// Whether this is a real, retired JR/JALR instruction.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for JumpChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Jump".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.jump_events.len(),
            None,
            <JumpChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.jump_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <JumpChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_JUMP_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_JUMP_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_JUMP_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut JumpCols<F> = row.borrow_mut();

                    if idx < input.jump_events.len() {
                        let event = &input.jump_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_JUMP_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.jump_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl JumpChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &JumpEvent,
        cols: &mut JumpCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);
        cols.is_real = F::ONE;

        let instruction = program.fetch(event.pc);
        cols.pc = F::from_canonical_u32(event.pc);

        cols.reader.populate(blu, instruction.op_a, event.a_record, instruction.op_b, event.b_record);

        cols.next_pc = Word::from(event.next_pc);
        cols.next_pc_range_checker.populate(blu, event.next_pc);
        cols.op_a_value = Word::from(event.a);
        cols.op_a_range_checker.populate(blu, event.a);
    }
}

impl<F> BaseAir<F> for JumpChip {
    fn width(&self) -> usize {
        NUM_JUMP_COLS
    }
}

impl<AB> Air<AB> for JumpChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &JumpCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The candidate link value: always `next_pc + 4`, masked to zero by
        // `eval_r_type_reader_masked` when `op_a_0` (JR, or a real `jalr $zero, ...`). Checked at
        // the reduced-scalar level, not limb-by-limb: a byte-wise `next_pc.0[0] + 4` would
        // silently drop the carry whenever that byte is >= 252.
        builder.when(local.is_real).assert_eq(
            local.op_a_value.reduce::<AB>(),
            local.next_pc.reduce::<AB>() + AB::Expr::from_canonical_u32(4),
        );
        let op_a_computed_value = local.op_a_value.map(Into::into);

        // The instruction word is reconstructed here rather than stored: `opcode` is hardcoded
        // (this chip only ever sees `Opcode::Jump`), `imm_b`/`imm_c` are compile-time constants
        // (`op_b` is always a register, `op_c` is always unused, hardcoded zero), and
        // `op_a`/`op_a_0`/`op_b` come from the adapter.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: AB::Expr::from_canonical_u32(Opcode::Jump as u32),
            op_a: local.reader.op_a.into(),
            op_b: Word::extend_var::<AB>(local.reader.op_b),
            op_c: Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            op_a_0: local.reader.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, local.is_real.into());

        eval_r_type_reader_masked(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            op_a_computed_value,
            local.is_real.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

        // JR/JALR's target is `op_b`'s own register value, used directly: no arithmetic, so no
        // dependency-chip send is needed here (unlike `JumpDirectChip`'s ADD lookup).
        let next_next_pc = local.reader.op_b_val().reduce::<AB>();
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
        // SAFETY: `is_real` is already checked to be boolean. `op_b_val()` needs no separate
        // range check here: it's an existing register value, already range-checked by whichever
        // chip originally wrote it.
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

    use super::JumpChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::Jump,
                op_a: 31,
                op_b: 5,
                op_c: 0,
                imm_b: false,
                imm_c: true,
                raw: None,
            }],
            pc_start: 0,
            pc_base: 0,
            next_pc: 4,
            image: Default::default(),
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.jump_events = vec![JumpEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            next_next_pc: 100,
            opcode: Opcode::Jump,
            a: 8,
            b: 100,
            c: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = JumpChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
