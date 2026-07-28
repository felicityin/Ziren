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
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent, MemoryAccessPosition, MemoryRecordEnum},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{
        clk_low_expr, eval_cpu_state, eval_i_type_immutable_reader, eval_state_chain, CpuState,
        InstructionCols, ITypeImmutableReader,
    },
    air::ZKMCoreAirBuilder,
    memory::MemoryReadCols,
    operations::WordAddressOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `LoadX0Chip`.
pub const NUM_LOAD_X0_COLS: usize = size_of::<LoadX0Cols<u8>>();

/// A chip that implements real, retired `lw $zero, offset($base)`/`ll $zero, offset($base)` --
/// i.e. `LoadWordChip`'s zero-destination case (see its doc comment). Structurally almost a clone
/// of `LoadWordChip`: same address computation/validation and the same real memory read, but the
/// loaded value is never asserted equal to anything (it's discarded, since `$zero` is hardwired to
/// 0), and `op_a` is pinned to register 0 by an explicit constraint rather than being a genuinely
/// variable register index -- letting `LoadWordChip` itself assume `op_a` is never register 0 and
/// use the unmasked `ITypeReaderNonZero`.
///
/// Reuses `ITypeImmutableReader` (built for `StoreWordChip`'s read-only `op_a`) rather than a
/// bespoke adapter: treating the discarded load as an immutable "read" of register 0 is exactly
/// the same shape as a store's real read, just with `op_a` fixed to 0 instead of variable.
#[derive(Default)]
pub struct LoadX0Chip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct LoadX0Cols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`. `op_a`/`op_a_0` are constrained to 0/1 (see
    /// `eval`), since this chip only ever handles `op_a == $zero`.
    pub adapter: ITypeImmutableReader<T>,

    /// The computed, validated memory address (`op_b + op_c`).
    pub word_address: WordAddressOperation<T>,

    /// The memory word being read. Range-checked and consistency-checked like any load, but
    /// never asserted equal to anything: the loaded value is discarded.
    pub memory_access: MemoryReadCols<T>,

    /// Whether this is a real, retired `lw $zero, ...` instruction.
    pub is_lw: T,
    /// Whether this is a real, retired `ll $zero, ...` instruction.
    pub is_ll: T,
}

impl<F: PrimeField32> MachineAir<F> for LoadX0Chip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "LoadX0".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.load_x0_events.len(),
            None,
            <LoadX0Chip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.load_x0_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <LoadX0Chip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_X0_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_X0_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_X0_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadX0Cols<F> = row.borrow_mut();

                    if idx < input.load_x0_events.len() {
                        let event = &input.load_x0_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_LOAD_X0_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.load_x0_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadX0Chip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadX0Cols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);
        cols.is_lw = F::from_bool(matches!(event.opcode, Opcode::LW));
        cols.is_ll = F::from_bool(matches!(event.opcode, Opcode::LL));

        let instruction = program.fetch(event.pc);
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        cols.adapter.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
        );

        cols.word_address.populate(blu, event.b, event.c);

        if let MemoryRecordEnum::Read(record) = event.mem_access {
            cols.memory_access.populate(record, blu);
        }
    }
}

impl<F> BaseAir<F> for LoadX0Chip {
    fn width(&self) -> usize {
        NUM_LOAD_X0_COLS
    }
}

impl<AB> Air<AB> for LoadX0Chip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadX0Cols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_lw);
        builder.assert_bool(local.is_ll);
        let is_real = local.is_lw + local.is_ll;
        builder.assert_bool(is_real.clone());

        // This chip is specifically for `op_a == $zero`: pin the adapter's witnessed `op_a`/
        // `op_a_0` so the register-access interaction below can only ever touch register 0 (a
        // freely-witnessed nonzero `op_a` here would let a real, non-zero-destination load sneak
        // its register write through this chip's discard-the-value path instead of
        // `LoadWordChip`'s real one).
        builder.when(is_real.clone()).assert_zero(local.adapter.op_a);
        builder.when(is_real.clone()).assert_one(local.adapter.op_a_0);

        let opcode = local.is_lw.into() * AB::Expr::from_canonical_u32(Opcode::LW as u32)
            + local.is_ll.into() * AB::Expr::from_canonical_u32(Opcode::LL as u32);
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

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        let addr_word = WordAddressOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val(),
            local.adapter.op_c,
            local.word_address,
            is_real.clone(),
        );

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            addr_word.reduce::<AB>(),
            &local.memory_access,
            is_real.clone(),
        );

        eval_i_type_immutable_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
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
    use zkm_core_executor::{
        events::{MemInstrEvent, MemoryReadRecord},
        ExecutionRecord, Instruction, Opcode, Program,
    };
    use zkm_hypercube::air::MachineAir;

    use super::LoadX0Chip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::LW,
                op_a: 0,
                op_b: 8,
                op_c: 4,
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
        shard.load_x0_events = vec![MemInstrEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            opcode: Opcode::LW,
            a: 0,
            b: 100,
            c: 4,
            mem_access: MemoryReadRecord::new(42, 5, 0).into(),
            prev_a_val: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = LoadX0Chip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
