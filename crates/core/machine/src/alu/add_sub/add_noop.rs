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
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState},
    adapter::InstructionCols,
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::RegisterWriteAccessCols,
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `AddNoopChip`.
pub const NUM_ADD_NOOP_COLS: usize = size_of::<AddNoopCols<u8>>();

/// A chip that implements MIPS's `SYNC`/`Pref` -- both decoded as `Opcode::ADD` with
/// `op_a=0, op_b=0, op_c=0, imm_b=true, imm_c=true` (see `Instruction::decode_from`): compile-time
/// constant operands, no register read for `b`/`c` at all, always writing (the hardwired-zero)
/// register 0. Split out of `AddChip` because this shape -- no real register access for `b`/`c`
/// -- can't be represented by `RTypeReader`, which assumes both are always real registers; see
/// `AddChip`'s doc comment for the split rationale. Unlike `AddChip`/`SubChip`, no other chip
/// ever sends a synthetic dependency row shaped like this (there's no meaningful "result" to
/// depend on), so every row here is a real, retired instruction -- no `is_retired` split needed.
#[derive(Default)]
pub struct AddNoopChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct AddNoopCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register 0's write access. Still tracked on the register-consistency bus (even though the
    /// value is always the constant zero) so that a later real read of register 0 has a
    /// timestamp to chain against.
    pub op_a_access: RegisterWriteAccessCols<T>,

    /// Whether this row is a real, retired instruction (as opposed to padding).
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for AddNoopChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "AddNoop".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.add_noop_events.len(),
            None,
            <AddNoopChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        AddNoopCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.add_noop_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <AddNoopChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADD_NOOP_COLS);

        values.chunks_mut(chunk_size * NUM_ADD_NOOP_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADD_NOOP_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddNoopCols<F> = row.borrow_mut();

                    if idx < input.add_noop_events.len() {
                        let event = &input.add_noop_events[idx];
                        self.event_to_row(event, cols, &mut Vec::new());
                    }
                });
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_ADD_NOOP_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.add_noop_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .add_noop_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_ADD_NOOP_COLS];
                    let cols: &mut AddNoopCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.add_noop_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddNoopChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut AddNoopCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.is_real = F::ONE;

        cols.state.populate(blu, event.clk);

        if let Some(record) = event.a_record {
            cols.op_a_access.populate(record, blu);
        }
    }
}

impl<F> BaseAir<F> for AddNoopChip {
    fn width(&self) -> usize {
        NUM_ADD_NOOP_COLS
    }
}

impl<AB> Air<AB> for AddNoopChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddNoopCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // The written value is always the constant zero -- SYNC/Pref always target register 0
        // (hardwired zero) and never compute any meaningful result.
        let zero_word =
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]);
        builder
            .when(local.is_real)
            .assert_word_eq(local.op_a_access.value.map(Into::into), zero_word);

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is fully reconstructed from constants: SYNC/Pref always decode to
        // `op_a=0, op_b=0, op_c=0, imm_b=true, imm_c=true` (see `Instruction::decode_from`), with
        // no real register access for `b`/`c` at all.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::ADD.as_field::<AB::F>().into(),
            op_a: AB::Expr::zero(),
            op_b: Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            op_c: Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            op_a_0: AB::Expr::one(),
            imm_b: AB::Expr::one(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, local.is_real.into());

        builder.eval_register_access_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(zkm_core_executor::events::MemoryAccessPosition::A as u32),
            AB::Expr::zero(),
            &local.op_a_access,
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
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode};
    use zkm_hypercube::air::MachineAir;

    use super::AddNoopChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.add_noop_events = vec![AluEvent::new(0, Opcode::ADD, 0, 0, 0)];
        let chip = AddNoopChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
