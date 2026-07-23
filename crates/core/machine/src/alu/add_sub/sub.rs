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
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, UNUSED_PC,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::InstructionCols,
    adapter::{clk_low_expr, eval_cpu_state, eval_r_type_reader, eval_state_chain, CpuState, RTypeReader},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    operations::AddOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `SubChip`.
pub const NUM_SUB_COLS: usize = size_of::<SubCols<u8>>();

/// A chip that implements subtraction for the opcodes SUB and SUBU.
///
/// SUB is basically an ADD with a re-arrangement of the operands and result: `a = b - c` is
/// verified as `b = a + c`. MIPS has no immediate-form SUBI, so every SUB event (real or a
/// dependency row) lands here, and every real, retired SUB instruction is register-register --
/// this is what lets the chip use the narrow [`RTypeReader`] (see its doc comment) instead of the
/// generic `InstructionCols`+`RegisterReader` pair.
///
/// Not every row corresponds to a real retired instruction: some rows are internal dependency
/// checks emitted by other chips (currently, only `emit_memory_dependencies`'s LB/LH
/// sign-extension check) that reuse this chip's arithmetic circuit instead of duplicating it.
/// Those rows carry the sentinel `pc == UNUSED_PC` and hand off via the chip-to-chip
/// `send_alu`/`receive_instruction` pair on `LookupKind::Instruction`, bypassing program lookup
/// and register access entirely. `is_retired` distinguishes the two (kept as its own witnessed
/// column rather than folded into a product, since interaction values/multiplicities in this
/// lookup argument must stay affine in the trace columns). See `AddChip`'s doc comment for the
/// symmetric register-form ADD chip this was split from.
#[derive(Default)]
pub struct SubChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct SubCols<T: Copy> {
    /// The current shard and clk. Only meaningful when `is_retired == 1`.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`. Only meaningful when this row is a real
    /// instruction (`is_retired == 1`).
    pub adapter: RTypeReader<T>,

    /// Whether this row is a real, retired SUB instruction (as opposed to an internal
    /// dependency check from another chip, or padding).
    pub is_retired: T,

    /// Instance of `AddOperation` to handle the underlying addition: `value = operand_1 +
    /// operand_2`, i.e. `b = a + c`, which verifies `a = b - c`.
    pub add_operation: AddOperation<T>,

    /// The first input operand: `a` (the sub result).
    pub operand_1: Word<T>,

    /// The second input operand: `c`.
    pub operand_2: Word<T>,

    /// Flag indicating whether this row is a real SUB-shaped row (retired instruction or
    /// synthetic dependency check) as opposed to padding.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for SubChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Sub".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.sub_events.len(),
            None,
            <SubChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        SubCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the rows for the trace.
        let chunk_size = std::cmp::max(input.sub_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <SubChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SUB_COLS);

        values.chunks_mut(chunk_size * NUM_SUB_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SUB_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SubCols<F> = row.borrow_mut();

                    if idx < input.sub_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.sub_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events, &input.program);
                    }
                    // A padding row is left all-zero: `is_real`/`is_retired` default to 0, which
                    // gates every interaction below to zero multiplicity on its own -- unlike the
                    // generic `RegisterReader`, `RTypeReader` needs no separate "force immediate
                    // flags" workaround for this.
                });
            },
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_SUB_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.sub_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .sub_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_SUB_COLS];
                    let cols: &mut SubCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.sub_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl SubChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut SubCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        cols.is_real = F::ONE;

        let operand_1 = event.a;
        let operand_2 = event.c;

        cols.add_operation.populate(blu, operand_1, operand_2);
        cols.operand_1 = Word::from(operand_1);
        cols.operand_2 = Word::from(operand_2);

        let is_real_instruction = event.pc != UNUSED_PC;
        if is_real_instruction {
            cols.is_retired = F::ONE;

            cols.state.populate(blu, event.clk);

            let instruction = program.fetch(event.pc);
            cols.adapter.populate(
                blu,
                instruction.op_a,
                event.a_record,
                instruction.op_b,
                event.b_record,
                instruction.op_c,
                event.c_record,
            );
        }
    }
}

impl<F> BaseAir<F> for SubChip {
    fn width(&self) -> usize {
        NUM_SUB_COLS
    }
}

impl<AB> Air<AB> for SubChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SubCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_retired);
        // `is_retired` can only be set alongside `is_real` -- kept as its own witnessed column
        // (not the product `is_real * is_retired`) because interaction values/multiplicities in
        // this lookup argument must stay affine in the trace columns.
        builder.when_not(local.is_real).assert_zero(local.is_retired);

        // Evaluate the addition operation: `add_operation.value = operand_1 + operand_2`, i.e.
        // `b = a + c`.
        AddOperation::<AB::F>::eval(
            builder,
            local.operand_1,
            local.operand_2,
            local.add_operation,
            local.is_real.into(),
        );

        // Register `a` holds `operand_1`, register `b` is written `add_operation.value`, and
        // register `c` always holds `operand_2`.
        let op_b_role = local.add_operation.value;
        let op_c_role = local.operand_2;

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: opcode/`imm_b`/`imm_c`
        // are compile-time constants (SUB is always register-register), and `op_b`/`op_c` are
        // zero-extended from the adapter's register-index columns. `send_program`'s lookup
        // against `ProgramChip`'s preprocessed ROM is what makes `op_a_0`/`op_a`/`op_b`/`op_c`
        // trustworthy -- there's no separate opcode-binding check needed, since the opcode here
        // is never a variable a malicious prover could substitute.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SUB.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: Word::extend_var::<AB>(local.adapter.op_c),
            op_a_0: local.adapter.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::zero(),
        };
        builder.send_program(local.pc, instruction, local.is_retired.into());

        eval_r_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.operand_1.map(Into::into),
            local.is_retired.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_retired.into());

        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk_high,
            clk_low,
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            local.is_retired.into(),
        );

        builder
            .when(local.is_retired)
            .assert_word_eq(local.adapter.op_b_val(), op_b_role.map(Into::into));
        builder
            .when(local.is_retired)
            .assert_word_eq(local.adapter.op_c_val(), op_c_role.map(Into::into));

        // ---- Synthetic dependency path: matches whichever chip generated this internal check via
        // `send_alu`/`send_alu_with_hi` (always at the `UNUSED_PC` sentinel, shard/clk zero).
        // `is_real - is_retired` is 1 exactly when this is a real SUB row that is *not* a real
        // instruction, i.e. a synthetic dependency row -- and stays affine (degree 1), unlike the
        // product `is_real * (1 - is_retired)`. `operand_1` is `a`, `add_operation.value` is `b`,
        // and `operand_2` is `c`. ----
        let zero_word =
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]);

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.next_pc,
            local.next_pc + AB::Expr::from_canonical_u32(4),
            AB::Expr::zero(),
            Opcode::SUB.as_field::<AB::F>(),
            local.operand_1,
            local.add_operation.value,
            local.operand_2,
            zero_word,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_real - local.is_retired,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode, UNUSED_PC};
    use zkm_hypercube::air::MachineAir;

    use super::SubChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        // `UNUSED_PC` keeps this a synthetic-dependency-style row, so trace generation doesn't
        // need a real `Program` to fetch an instruction from.
        shard.sub_events = vec![AluEvent::new(UNUSED_PC, Opcode::SUB, 14, 8, 6)];
        let chip = SubChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
