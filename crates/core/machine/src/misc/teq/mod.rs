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
    events::{ByteLookupEvent, ByteRecord, MiscEvent},
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
        clk_low_expr, eval_cpu_state, eval_i_type_immutable_reader, eval_state_chain, CpuState,
        ITypeImmutableReader, InstructionCols,
    },
    air::ZKMCoreAirBuilder,
    operations::IsEqualWordOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `TeqChip`.
pub const NUM_TEQ_COLS: usize = size_of::<TeqCols<u8>>();

/// A chip that implements the MIPS trap-on-equal instruction TEQ.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `teq_events`. TEQ never writes `op_a` (it's a pure comparison), so its register access is an
/// immutable read (`op_a` and `op_b` both read, never register-0-masked); `op_c` is always the
/// instruction's own encoded immediate (always exactly `0`, never read) -- the same shape
/// `ITypeImmutableReader` covers (already used by `BranchChip` and several load/store chips).
#[derive(Default)]
pub struct TeqChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct TeqCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeImmutableReader<T>,

    /// Whether `op_a == op_b` -- TEQ must trap only when they differ (equality is
    /// architecturally unreachable for a valid trap-free execution, mirrored by the `assert_zero`
    /// below).
    pub a_eq_b: IsEqualWordOperation<T>,

    /// Whether this row is a real, retired TEQ instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for TeqChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Teq".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        TeqCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.teq_events.len(),
            None,
            <TeqChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.teq_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <TeqChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_TEQ_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_TEQ_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_TEQ_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut TeqCols<F> = row.borrow_mut();

                    if idx < input.teq_events.len() {
                        let event = &input.teq_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_TEQ_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.teq_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl TeqChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MiscEvent,
        cols: &mut TeqCols<F>,
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
        );

        cols.a_eq_b.populate(event.a, event.b);
    }
}

impl<F> BaseAir<F> for TeqChip {
    fn width(&self) -> usize {
        NUM_TEQ_COLS
    }
}

impl<AB> Air<AB> for TeqChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &TeqCols<AB::Var> = (*local).borrow();

        let is_real = local.is_real;
        builder.assert_bool(is_real);

        let op_a_val = local.adapter.op_a_val();
        let op_b_val = local.adapter.op_b_val();

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode`/`imm_b` are
        // compile-time constants (this chip only ever sees register `op_a`/`op_b` -- see this
        // chip's doc comment), `imm_c` is always set (TEQ's `op_c` is always the immediate 0),
        // and `op_a`/`op_a_0`/`op_b`/`op_c` are the adapter's own columns.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::TEQ.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: local.adapter.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.into());

        eval_i_type_immutable_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
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

        // Check that a != b -- a real TEQ retirement never traps (the executor rejects a program
        // that would), so `op_a == op_b` is architecturally unreachable here.
        IsEqualWordOperation::<AB::F>::eval(
            builder,
            op_a_val.map(|x| x.into()),
            op_b_val.map(|x| x.into()),
            local.a_eq_b,
            is_real.into(),
        );
        builder.when(is_real).assert_zero(local.a_eq_b.is_diff_zero.result);
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::MiscEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::TeqChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::TEQ,
                op_a: 5,
                op_b: 8,
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
        shard.teq_events = vec![MiscEvent::new(
            0,
            0,
            4,
            Opcode::TEQ,
            5,
            9,
            0,
            0,
            Default::default(),
        )];
        let chip = TeqChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
