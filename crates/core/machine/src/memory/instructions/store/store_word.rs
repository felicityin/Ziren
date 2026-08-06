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
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent, MemoryAccessPosition},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{
        clk_low_expr, eval_cpu_state, eval_i_type_immutable_reader, eval_state_chain, CpuState,
        InstructionCols, ITypeImmutableReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, MemoryReadWriteCols},
    operations::WordAddressOperation,
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `StoreWordChip`.
pub const NUM_STORE_WORD_COLS: usize = size_of::<StoreWordCols<u8>>();

/// A chip that implements the word-aligned store opcode SW.
///
/// See `LoadWordChip`'s doc comment for the general rationale (dropped sign-extension/
/// partial-merge/offset-mux columns, bare-immediate `c`). Symmetric to `LoadWordChip` except for
/// which side of the register/memory pair is mutable: SW reads `op_a` (immutable -- a store
/// never writes a register) and writes `memory_access`, so it uses `ITypeImmutableReader` (a
/// plain read for `op_a`, unlike `LoadWordChip`'s `ITypeReaderNonZero`) and keeps the
/// read-write `MemoryReadWriteCols` on `memory_access` instead.
///
/// Unlike `LoadWordChip`, a real `sw $zero, ...` needs no separate chip: `op_a` is only ever read
/// here, so reading `$zero` is already trivially safe with no masking.
#[derive(Default)]
pub struct StoreWordChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct StoreWordCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeImmutableReader<T>,

    /// The computed, validated memory address (`op_b + op_c`).
    pub word_address: WordAddressOperation<T>,

    /// The memory word being written (overwritten with `op_a`'s value in full -- no partial
    /// masking needed since SW is always a whole-word store).
    pub memory_access: MemoryReadWriteCols<T>,

    /// Whether this row is a real, retired SW instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for StoreWordChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "StoreWord".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.store_word_events.len(),
            None,
            <StoreWordChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.store_word_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <StoreWordChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_STORE_WORD_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_STORE_WORD_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_STORE_WORD_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut StoreWordCols<F> = row.borrow_mut();

                    if idx < input.store_word_events.len() {
                        let event = &input.store_word_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_STORE_WORD_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.store_word_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl StoreWordChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut StoreWordCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);
        cols.is_real = F::ONE;

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

        cols.memory_access.populate(event.mem_access, blu);
    }
}

impl<F> BaseAir<F> for StoreWordChip {
    fn width(&self) -> usize {
        NUM_STORE_WORD_COLS
    }
}

impl<AB> Air<AB> for StoreWordChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &StoreWordCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // The instruction word is reconstructed here rather than stored: `imm_b`/`imm_c` are
        // compile-time constants (this chip only ever sees real, retired SW), `opcode` is
        // hardcoded to SW, and `op_a`/`op_a_0`/`op_b`/`op_c` come from the adapter. Unlike
        // `LoadWordChip`, `op_a_0` is a real per-row value (not hardcoded to 0): `op_a` is only
        // ever read here, so a real `sw $zero, ...` is a valid, common row (zeroing memory), and
        // `send_program`'s lookup tuple must reflect the true decoded `op_a_0`.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SW.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: local.adapter.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, local.is_real.into());

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        let addr_word = WordAddressOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val(),
            local.adapter.op_c,
            local.word_address,
            local.is_real.into(),
        );

        // The stored word is always `op_a`'s value in full.
        builder
            .when(local.is_real)
            .assert_word_eq(local.memory_access.value().map(Into::into), local.adapter.op_a_val().map(Into::into));
        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            addr_word.reduce::<AB>(),
            &local.memory_access,
            local.is_real,
        );

        eval_i_type_immutable_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
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
    use zkm_core_executor::{
        events::{MemInstrEvent, MemoryWriteRecord},
        ExecutionRecord, Instruction, Opcode, Program,
    };
    use zkm_hypercube::air::MachineAir;

    use super::StoreWordChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SW,
                op_a: 5,
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
        shard.store_word_events = vec![MemInstrEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            opcode: Opcode::SW,
            a: 42,
            b: 100,
            c: 4,
            mem_access: MemoryWriteRecord::new(42, 5, 0, 0).into(),
            prev_a_val: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = StoreWordChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
