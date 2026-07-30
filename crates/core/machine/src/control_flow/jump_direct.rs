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
        clk_low_expr, eval_cpu_state, eval_j_type_reader_non_zero, eval_state_chain, CpuState,
        InstructionCols, JTypeReaderNonZero,
    },
    air::ZKMCoreAirBuilder,
    operations::{AddOperation, KoalaBearWordRangeChecker},
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `JumpDirectChip`.
pub const NUM_JUMP_DIRECT_COLS: usize = size_of::<JumpDirectCols<u8>>();

/// A chip that implements the MIPS pc-relative jump-and-link instruction BAL (`Opcode::JumpDirect`).
/// Unlike JAL, BAL's link register isn't even encoded as a variable register field -- it's
/// unconditionally 31 -- so there is no "BAL to $zero" case to route or mask at all, hence the
/// guaranteed-nonzero `JTypeReaderNonZero` adapter (see its doc comment).
///
/// Unlike `JumpChip`/`JumpiChip`, BAL's target is pc-relative (`next_next_pc = next_pc + op_b`),
/// verified via a locally embedded `AddOperation` (no cross-chip lookup).
#[derive(Default)]
pub struct JumpDirectChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct JumpDirectCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current program counter.
    pub pc: T,

    /// Register/immediate operand access for `a`/`b`: `op_a` (written, always register 31) is the
    /// link register, `op_b` is the instruction's own encoded pc-relative offset (never a
    /// register).
    pub reader: JTypeReaderNonZero<T>,

    /// The next program counter (`pc + 4`) -- also the basis for the link value written to
    /// `op_a`, and the base of the pc-relative ADD lookup below.
    pub next_pc: Word<T>,
    pub next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// The resolved jump target (`next_pc + op_b`), computed locally by `add_operation`.
    pub add_operation: AddOperation<T>,
    pub next_next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// The value written to `op_a` (`next_pc + 4`). Unlike `JumpChip`/`JumpiChip`'s masked
    /// adapters, `JTypeReaderNonZero`'s `op_a_access` (`RegisterAccessCols`) has no separate
    /// `value` witness of its own -- the computed value is sent directly, unmasked -- so this
    /// chip needs its own witness to independently range-check the `next_pc + 4` addition.
    pub op_a_value: Word<T>,
    pub op_a_range_checker: KoalaBearWordRangeChecker<T>,

    /// Whether this is a real, retired BAL instruction.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for JumpDirectChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "JumpDirect".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.jumpdirect_events.len(),
            None,
            <JumpDirectChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.jumpdirect_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <JumpDirectChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_JUMP_DIRECT_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_JUMP_DIRECT_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_JUMP_DIRECT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut JumpDirectCols<F> = row.borrow_mut();

                    if idx < input.jumpdirect_events.len() {
                        let event = &input.jumpdirect_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_JUMP_DIRECT_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.jumpdirect_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl JumpDirectChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &JumpEvent,
        cols: &mut JumpDirectCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);
        cols.is_real = F::ONE;

        let instruction = program.fetch(event.pc);
        cols.pc = F::from_canonical_u32(event.pc);

        cols.reader.populate(blu, instruction.op_a, event.a_record, instruction.op_b);

        cols.next_pc = Word::from(event.next_pc);
        cols.next_pc_range_checker.populate(blu, event.next_pc);
        let target_pc = cols.add_operation.populate(blu, event.next_pc, event.b);
        cols.next_next_pc_range_checker.populate(blu, target_pc);
        cols.op_a_value = Word::from(event.a);
        cols.op_a_range_checker.populate(blu, event.a);
    }
}

impl<F> BaseAir<F> for JumpDirectChip {
    fn width(&self) -> usize {
        NUM_JUMP_DIRECT_COLS
    }
}

impl<AB> Air<AB> for JumpDirectChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &JumpDirectCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // BAL's link register is unconditionally 31 -- no masking, sent directly. `op_a_value` is
        // its own witness (see the column's doc comment): assert it equals `next_pc + 4` here,
        // rather than reconstructing the addition inline, so it can be independently range
        // checked below (a raw `next_pc + 4` field-arithmetic expression isn't otherwise known to
        // decompose into valid bytes).
        builder.when(local.is_real).assert_eq(
            local.op_a_value.reduce::<AB>(),
            local.next_pc.reduce::<AB>() + AB::Expr::from_canonical_u32(4),
        );
        let op_a_computed_value = local.op_a_value.map(Into::into);

        // The instruction word is reconstructed here rather than stored: `opcode` is hardcoded
        // (this chip only ever sees `Opcode::JumpDirect`), `op_a` is hardcoded to register 31
        // (BAL's link register isn't a variable field at all), `imm_b`/`imm_c` are compile-time
        // constants (`op_b` is always the encoded pc-relative offset, an immediate; `op_c` is
        // always unused, hardcoded zero).
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: AB::Expr::from_canonical_u32(Opcode::JumpDirect as u32),
            op_a: local.reader.op_a.into(),
            op_b: local.reader.op_b.map(Into::into),
            op_c: Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::one(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, local.is_real.into());

        eval_j_type_reader_non_zero(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            op_a_computed_value,
            local.is_real.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

        // BAL's target is pc-relative: `next_next_pc = next_pc + op_b`, computed locally by
        // `add_operation` (no cross-chip lookup).
        AddOperation::<AB::F>::eval(
            builder,
            local.next_pc,
            local.reader.op_b,
            local.add_operation,
            local.is_real.into(),
        );
        let next_next_pc = local.add_operation.value;

        eval_state_chain(
            builder,
            clk_high,
            clk_low,
            local.pc.into(),
            local.next_pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            next_next_pc.reduce::<AB>(),
            AB::Expr::from_canonical_u32(5),
            local.is_real.into(),
        );

        // Range check `next_pc`, `next_next_pc`, and the value written to `op_a`.
        // SAFETY: `is_real` is already checked to be boolean. `next_next_pc` (`add_operation`'s
        // own witnessed result) isn't already known to be a valid word the way an existing
        // register value or Program-table immediate would be -- it still needs its own check.
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_pc,
            local.next_pc_range_checker,
            local.is_real.into(),
        );
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            next_next_pc,
            local.next_next_pc_range_checker,
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

    use super::JumpDirectChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::JumpDirect,
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
        shard.jumpdirect_events = vec![JumpEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            next_next_pc: 104,
            opcode: Opcode::JumpDirect,
            a: 8,
            b: 100,
            c: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = JumpDirectChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
