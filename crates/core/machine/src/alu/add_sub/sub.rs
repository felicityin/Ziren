use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
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
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `SubChip`.
pub const NUM_SUB_COLS: usize = size_of::<SubCols<u8>>();

/// A chip that implements subtraction for the opcodes SUB and SUBU.
///
/// SUB is basically an ADD with a re-arrangement of the operands and result: `a = b - c` is
/// verified as `b = a + c`. MIPS has no immediate-form SUBI, so every SUB event lands here --
/// except a real `sub $zero, ...`, which is routed to `AluX0Chip` instead (its result is
/// unobservable, and discarding it soundly requires a different, cheaper register-write scheme
/// than a real result does; see `RTypeReader`'s doc comment). Every SUB instruction reaching
/// *this* chip therefore has a genuine, non-zero destination register, which is what lets it use
/// the narrow [`RTypeReader`] (see its doc comment) instead of the generic
/// `InstructionCols`+`RegisterReader` pair.
///
/// Every row is a real, retired instruction: `LoadByteChip`/`LoadHalfChip`'s LB/LH sign-extension
/// check -- formerly a dependency row reusing this chip's arithmetic circuit -- is now a direct
/// byte assertion local to those chips instead (their sign-correction constant is compile-time
/// fixed, so no genuine addition dependency exists), so nothing sends synthetic rows here anymore.
/// See `AddChip`'s doc comment for the symmetric register-form ADD chip this was split from.
#[derive(Default)]
pub struct SubChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct SubCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: RTypeReader<T>,

    /// Instance of `AddOperation` to handle the underlying addition: `value = operand_1 +
    /// op_c_val()`, i.e. `b = a + c`, which verifies `a = b - c`.
    pub add_operation: AddOperation<T>,

    /// The first input operand: `a` (the sub result).
    pub operand_1: Word<T>,

    /// Flag indicating whether this row is a real, retired SUB instruction, as opposed to
    /// padding.
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
        let nb_rows = next_multiple_of_32(
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

        let operand_1 = event.a;
        let operand_2 = event.c;

        cols.add_operation.populate(blu, operand_1, operand_2);
        cols.operand_1 = Word::from(operand_1);
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

        // Evaluate the addition operation: `add_operation.value = operand_1 + op_c_val()`, i.e.
        // `b = a + c`. `op_c_val()` is used directly rather than a separate witnessed operand,
        // since (unlike the old dependency-row scheme) every row here reads a genuine register c.
        AddOperation::<AB::F>::eval(
            builder,
            local.operand_1,
            local.adapter.op_c_val(),
            local.add_operation,
            local.is_real.into(),
        );

        // Register `a` holds `operand_1`, register `b` is written `add_operation.value`.
        let op_b_role = local.add_operation.value;

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: opcode/`op_a_0`/`imm_b`/
        // `imm_c` are compile-time constants (SUB is always register-register, with `op_a != 0`
        // guaranteed -- any `op_a==0` row is routed to `AluX0Chip` instead, see `RTypeReader`'s
        // doc comment), and `op_b`/`op_c` are zero-extended from the adapter's register-index
        // columns. `send_program`'s lookup against `ProgramChip`'s preprocessed ROM is what makes
        // `op_a`/`op_b`/`op_c` trustworthy -- there's no separate opcode-binding check needed,
        // since the opcode here is never a variable a malicious prover could substitute.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SUB.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: Word::extend_var::<AB>(local.adapter.op_c),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::zero(),
        };
        builder.send_program(local.pc, instruction, local.is_real.into());

        eval_r_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.operand_1.map(Into::into),
            local.is_real.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

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
            local.is_real.into(),
        );

        builder
            .when(local.is_real)
            .assert_word_eq(local.adapter.op_b_val(), op_b_role.map(Into::into));
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::SubChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction::new(Opcode::SUB, 30, 29, 28, false, false)],
            pc_start: 0,
            pc_base: 0,
            ..Default::default()
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.sub_events = vec![AluEvent::new(0, Opcode::SUB, 2, 8, 6)];
        let chip = SubChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
