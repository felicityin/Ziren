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
        clk_low_expr, eval_cpu_state, eval_i_type_reader, eval_state_chain, CpuState,
        InstructionCols, ITypeReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, MemoryReadWriteCols},
    operations::WordAddressOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `StoreConditionalChip`.
pub const NUM_STORE_CONDITIONAL_COLS: usize = size_of::<StoreConditionalCols<u8>>();

/// A chip that implements the word-aligned atomic store-conditional opcode SC.
///
/// Always word-aligned (like `LoadWordChip`/`StoreWordChip`, hence `WordAddressOperation`), but
/// unlike `StoreWordChip`, `op_a` is a read-*and*-write: the register's *previous* value is what
/// gets stored to memory (this executor models SC as always succeeding), and the register is then
/// overwritten with the constant `1` (success). `ITypeReader`'s `RegisterWriteAccessCols` already
/// witnesses both the previous and new value of a write, so no bespoke adapter is needed --
/// `op_a_access.prev_value` is the stored word, and `op_a_access.value` is asserted to be
/// `Word([1, 0, 0, 0])` via `eval_i_type_reader`'s usual masked-write mechanism (a real `sc
/// $zero, ...` silently discards the write, per `ITypeReader`'s `op_a==0` masking).
#[derive(Default)]
pub struct StoreConditionalChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct StoreConditionalCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReader<T>,

    /// The computed, validated memory address (`op_b + op_c`).
    pub word_address: WordAddressOperation<T>,

    /// The memory word being written (overwritten in full with `op_a`'s previous value).
    pub memory_access: MemoryReadWriteCols<T>,

    /// Whether this row is a real, retired SC instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for StoreConditionalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "StoreConditional".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.store_conditional_events.len(),
            None,
            <StoreConditionalChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.store_conditional_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <StoreConditionalChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_STORE_CONDITIONAL_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_STORE_CONDITIONAL_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_STORE_CONDITIONAL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut StoreConditionalCols<F> = row.borrow_mut();

                    if idx < input.store_conditional_events.len() {
                        let event = &input.store_conditional_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_STORE_CONDITIONAL_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.store_conditional_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl StoreConditionalChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut StoreConditionalCols<F>,
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

impl<F> BaseAir<F> for StoreConditionalChip {
    fn width(&self) -> usize {
        NUM_STORE_CONDITIONAL_COLS
    }
}

impl<AB> Air<AB> for StoreConditionalChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &StoreConditionalCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        let is_real = local.is_real;

        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SC.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: local.adapter.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.into());

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        let addr_word = WordAddressOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val(),
            local.adapter.op_c,
            local.word_address,
            is_real.into(),
        );

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            addr_word.reduce::<AB>(),
            &local.memory_access,
            is_real,
        );

        // The word stored to memory is `op_a`'s previous value (the value being "swapped out").
        builder.when(is_real).assert_word_eq(
            local.memory_access.value().map(Into::into),
            local.adapter.op_a_access.prev_value.map(Into::into),
        );

        // `op_a` becomes 1 (success -- this executor always models SC as succeeding).
        let one_word: Word<AB::Expr> =
            Word([AB::Expr::one(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]);
        eval_i_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            one_word,
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
    use zkm_core_executor::{
        events::{MemInstrEvent, MemoryWriteRecord},
        ExecutionRecord, Instruction, Opcode, Program,
    };
    use zkm_hypercube::air::MachineAir;

    use super::StoreConditionalChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SC,
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
        shard.store_conditional_events = vec![MemInstrEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            opcode: Opcode::SC,
            a: 1,
            b: 100,
            c: 4,
            mem_access: MemoryWriteRecord::new(0x1234_5678, 5, 0xDEAD_BE42, 0).into(),
            prev_a_val: 0x1234_5678,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = StoreConditionalChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
