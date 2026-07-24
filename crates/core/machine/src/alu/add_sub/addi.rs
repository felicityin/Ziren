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
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{
        clk_low_expr, eval_cpu_state, eval_i_type_reader, eval_state_chain, CpuState,
        InstructionCols, ITypeReader,
    },
    air::ZKMCoreAirBuilder,
    operations::AddOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `AddiChip`.
pub const NUM_ADDI_COLS: usize = size_of::<AddiCols<u8>>();

/// A chip that implements addition for the immediate-form opcodes ADDI and ADDIU: register `b`
/// plus the instruction's own encoded immediate `c`.
///
/// Unlike `AddChip`/`SubChip`, `c` never comes from a register here -- it's read directly off the
/// adapter's own immediate `op_c` (already populated for the `send_program` lookup), so this chip
/// pays no register access for it, and every real row is `a = b + c` with no role mux. The
/// synthetic internal-dependency-check rows other chips emit via `send_alu`/`send_alu_with_hi` are
/// always register-shaped (see `AddChip`'s doc comment) and stay on `AddChip`/`SubChip`; every row
/// here is a real, retired instruction.
#[derive(Default)]
pub struct AddiChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct AddiCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReader<T>,

    /// Instance of `AddOperation` verifying `a = b + c`.
    pub add_operation: AddOperation<T>,

    /// Whether this row is a real, retired ADDI/ADDIU instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for AddiChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Addi".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.addi_events.len(),
            None,
            <AddiChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.addi_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <AddiChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADDI_COLS);

        values.chunks_mut(chunk_size * NUM_ADDI_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADDI_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddiCols<F> = row.borrow_mut();

                    if idx < input.addi_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.addi_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events, &input.program);
                    }
                });
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_ADDI_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.addi_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .addi_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_ADDI_COLS];
                    let cols: &mut AddiCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.addi_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddiChip {
    /// Create a row from a real, retired ADDI/ADDIU event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut AddiCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.is_real = F::ONE;

        cols.add_operation.populate(blu, event.b, event.c);

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
    }
}

impl<F> BaseAir<F> for AddiChip {
    fn width(&self) -> usize {
        NUM_ADDI_COLS
    }
}

impl<AB> Air<AB> for AddiChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddiCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // The instruction word is reconstructed here rather than stored: opcode is hardcoded to
        // ADD (MIPS has no immediate-form SUB), `op_a`/`op_a_0`/`op_b` come from the adapter, and
        // `imm_b=0`/`imm_c=1` are compile-time constants (every real row here is register `b` plus
        // an encoded immediate `c`).
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::ADD.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: local.adapter.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, local.is_real.into());

        AddOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val(),
            local.adapter.op_c,
            local.add_operation,
            local.is_real.into(),
        );

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        eval_i_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.add_operation.value.map(Into::into),
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
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::AddiChip;

    #[test]
    fn generate_trace() {
        // Unlike `AddChip`/`SubChip`, every real `AddiChip` row is a genuine retired instruction (no
        // synthetic-dependency-row shortcut), so `event_to_row` always fetches from `program` --
        // this needs a real single-instruction program to fetch, matching the event's pc.
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::ADD,
                op_a: 5,
                op_b: 8,
                op_c: 6,
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
        shard.addi_events = vec![AluEvent::new(0, Opcode::ADD, 14, 8, 6)];
        let chip = AddiChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
