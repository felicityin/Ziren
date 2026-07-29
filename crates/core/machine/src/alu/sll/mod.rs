//! Verifies left shift.
//!
//! This module implements left shift (b << c) as a combination of bit and byte shifts, via the
//! shared, embeddable `ShiftLeftOperation` (see `operations/shift_left.rs`).

pub mod lui;
pub use lui::*;

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
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
    adapter::{
        clk_low_expr, eval_alu_type_reader, eval_cpu_state, eval_state_chain, AluTypeReader,
        CpuState, InstructionCols,
    },
    air::ZKMCoreAirBuilder,
    operations::ShiftLeftOperation,
    utils::{next_power_of_two, pad_rows_fixed},
    CoreChipError,
};

/// The number of main trace columns for `ShiftLeft`.
pub const NUM_SHIFT_LEFT_COLS: usize = size_of::<ShiftLeftCols<u8>>();

/// A chip that implements bitwise operations for the opcodes SLL and SLLV.
///
/// Every row is a real, retired instruction: `ExtChip`/`InsChip` (the only other chips with an
/// internal SLL dependency) each verify their own copy locally via an embedded
/// `ShiftLeftOperation` instead of a cross-chip lookup into this chip. A real SLL/SLLV whose
/// destination is register 0 is routed to `AluX0Chip` instead (see its doc comment), since its
/// result is unobservable and discarding it soundly requires a different (cheaper) register-write
/// scheme than a real result does -- see `AluTypeReader`'s doc comment. Every real row reaching
/// *this* chip therefore has a genuine, non-zero destination register, which is what lets it use
/// the narrow `AluTypeReader` (see its doc comment) instead of the generic
/// `InstructionCols`+`RegisterReader` pair.
///
/// LUI also decodes to `Opcode::SLL`, but with `imm_b=true` (`op_b` the instruction's own encoded
/// immediate, never a register) -- a distinct shape handled by `LuiChip` instead (see its doc
/// comment), so every row reaching *this* chip also has `imm_b=false`, i.e. a genuine register
/// `op_b`.
#[derive(Default)]
pub struct ShiftLeft;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct ShiftLeftCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: AluTypeReader<T>,

    /// `a = b << c`, computed locally.
    pub shift_left_operation: ShiftLeftOperation<T>,

    /// Whether this row is a real, retired SLL instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for ShiftLeft {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShiftLeft".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        ShiftLeftCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.shift_left_events.len(),
            None,
            <ShiftLeft as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows = input
            .shift_left_events
            .par_iter()
            .map(|event| {
                let mut row = [F::ZERO; NUM_SHIFT_LEFT_COLS];
                let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(event, cols, &mut blu, &input.program);
                row
            })
            .collect::<Vec<_>>();

        // Pad the trace to a power of two.
        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_SHIFT_LEFT_COLS],
            None,
            <ShiftLeft as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_SHIFT_LEFT_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.shift_left_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_left_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_SHIFT_LEFT_COLS];
                    let cols: &mut ShiftLeftCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.shift_left_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl ShiftLeft {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftLeftCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.is_real = F::ONE;

        debug_assert!(event.pc != UNUSED_PC, "every ShiftLeft row is now a real instruction");
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
            instruction.imm_c,
        );

        cols.shift_left_operation.populate(blu, event.b, event.c);
    }
}

impl<F> BaseAir<F> for ShiftLeft {
    fn width(&self) -> usize {
        NUM_SHIFT_LEFT_COLS
    }
}

impl<AB> Air<AB> for ShiftLeft
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShiftLeftCols<AB::Var> = (*local).borrow();

        let is_real = local.is_real;
        builder.assert_bool(is_real);

        let a_word = ShiftLeftOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val(),
            local.adapter.op_c_val(),
            local.shift_left_operation,
            is_real.into(),
        );

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode` is a
        // compile-time constant (never a variable a malicious prover could substitute -- the
        // program lookup against `ProgramChip`'s preprocessed ROM is what makes
        // `op_a`/`op_b`/`op_c` trustworthy), `op_a_0`/`imm_b` are compile-time constants (this
        // chip only ever sees a non-zero destination and a register `op_b` -- see
        // `AluTypeReader`'s doc comment), and `op_b` is zero-extended from the adapter's
        // register-index column.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SLL.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: local.adapter.imm_c.into(),
        };
        builder.send_program(local.pc, instruction, is_real.into());

        eval_alu_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            a_word.map(Into::into),
            is_real.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.into());

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
            is_real.into(),
        );
    }
}

#[cfg(test)]
mod tests {

    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::ShiftLeft;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SLL,
                op_a: 5,
                op_b: 8,
                op_c: 1,
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
        shard.shift_left_events = vec![AluEvent::new(0, Opcode::SLL, 16, 8, 1)];
        let chip = ShiftLeft::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
