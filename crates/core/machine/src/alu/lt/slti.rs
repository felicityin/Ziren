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
    adapter::{
        clk_low_expr, eval_cpu_state, eval_i_type_reader, eval_state_chain, CpuState,
        InstructionCols, ITypeReader,
    },
    air::ZKMCoreAirBuilder,
    operations::LtOperation,
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `SltiChip`.
pub const NUM_SLTI_COLS: usize = size_of::<SltiCols<u8>>();

/// A chip that implements the immediate-form opcodes SLTI and SLTIU: register `b` compared
/// against the instruction's own encoded immediate `c`.
///
/// See `LtChip`'s doc comment for the register-form SLT/SLTU chip this was split from -- MIPS's
/// SLT/SLTU secretly cover both register-register and register-immediate shapes under the same
/// opcode, exactly like `Opcode::ADD` did before its own `AddChip`/`AddiChip` split. Every row
/// here is a real, retired instruction: `LtChip`'s existing internal dependency checks (from
/// `DivRem`/`Branch`) are always register-form and stay on `LtChip`.
#[derive(Default)]
pub struct SltiChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct SltiCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReader<T>,

    /// The SLTI/SLTIU comparison circuit (shared with `LtChip`).
    pub lt_operation: LtOperation<T>,
}

impl<F: PrimeField32> MachineAir<F> for SltiChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Slti".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.slti_events.len(),
            None,
            <SltiChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        SltiCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.slti_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <SltiChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SLTI_COLS);

        values.chunks_mut(chunk_size * NUM_SLTI_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SLTI_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SltiCols<F> = row.borrow_mut();

                    if idx < input.slti_events.len() {
                        let event = &input.slti_events[idx];
                        self.event_to_row(event, cols, &mut Vec::new(), &input.program);
                    }
                });
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_SLTI_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.slti_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .slti_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_SLTI_COLS];
                    let cols: &mut SltiCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.slti_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl SltiChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut SltiCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);
        cols.adapter.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
        );

        let result = cols.lt_operation.populate(blu, event.opcode, event.b, event.c);
        assert_eq!(result, event.a);
    }
}

impl<F> BaseAir<F> for SltiChip {
    fn width(&self) -> usize {
        NUM_SLTI_COLS
    }
}

impl<AB> Air<AB> for SltiChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SltiCols<AB::Var> = (*local).borrow();

        // Every row here is real or padding -- there's no separate "is_retired"/synthetic split
        // (see `SltiChip`'s doc comment), so `is_real` is derived directly from `LtOperation`'s
        // own opcode selectors, exactly like the residual `LtChip` derives its own `is_real`.
        let is_real = local.lt_operation.is_slt + local.lt_operation.is_sltu;

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here: opcode is selected by `LtOperation`'s own
        // `is_slt`/`is_sltu`, `op_a`/`op_a_0`/`op_b` come from the adapter, `imm_b=0`/`imm_c=1`
        // are compile-time constants (SLTI/SLTIU always read a real `op_b` register and encode
        // `op_c` as an immediate).
        let opcode = local.lt_operation.is_slt * Opcode::SLT.as_field::<AB::F>()
            + local.lt_operation.is_sltu * Opcode::SLTU.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: local.adapter.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        LtOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val().map(Into::into),
            local.adapter.op_c.map(Into::into),
            local.lt_operation,
            is_real.clone(),
        );

        eval_i_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.lt_operation.a.map(Into::into),
            is_real.clone(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.clone());

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
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::SltiChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.program =
            Program::new(vec![Instruction::new(Opcode::SLT, 29, 30, 2, false, true)], 0, 0).into();
        shard.slti_events = vec![AluEvent::new(0, Opcode::SLT, 0, 3, 2)];
        let chip = SltiChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
