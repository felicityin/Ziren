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
    ByteOpcode, ExecutionRecord, Opcode, Program, NUM_REGISTERS,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::MachineAir;

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState, InstructionCols},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols},
    operations::{IsZeroOperation, KoalaBearWordRangeChecker},
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `StoreWordChip`.
pub const NUM_STORE_WORD_COLS: usize = size_of::<StoreWordCols<u8>>();

/// A chip that implements the word-aligned store opcode SW.
///
/// See `LoadWordChip`'s doc comment for the general rationale (dropped sign-extension/
/// partial-merge/offset-mux columns, bare-immediate `c`). Symmetric to `LoadWordChip` except for
/// which side of the register/memory pair is mutable: SW reads `op_a` (immutable -- a store
/// never writes a register) and writes `memory_access`, so those two columns' `MemoryReadCols`/
/// `MemoryReadWriteCols` types are swapped relative to `LoadWordChip`.
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

    /// The raw fetched instruction, used for the program lookup and register/immediate
    /// resolution.
    pub instruction: InstructionCols<T>,

    /// Register `a` read access (the word to store; immutable -- a store never writes it).
    pub op_a_access: MemoryReadCols<T>,
    /// Register `b` read access (the base address register).
    pub op_b_access: MemoryReadCols<T>,

    /// The computed, unaligned-checked memory address (`op_b + instruction.op_c`), verified via
    /// the ALU table.
    pub addr_word: zkm_hypercube::word::Word<T>,
    /// Gadget to verify that `addr_word` is within the Koala-Bear field.
    pub addr_word_range_checker: KoalaBearWordRangeChecker<T>,
    /// Used to check that the address is at least `NUM_REGISTERS`, i.e. doesn't alias the
    /// register file.
    pub most_sig_bytes_zero: IsZeroOperation<T>,

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
        let nb_rows = next_power_of_two(
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
        cols.instruction.populate(&instruction);

        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        if let Some(MemoryRecordEnum::Read(record)) = event.a_record {
            cols.op_a_access.populate(record, blu);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.b_record {
            cols.op_b_access.populate(record, blu);
        }

        let memory_addr = event.b.wrapping_add(event.c);
        cols.addr_word = memory_addr.into();
        cols.addr_word_range_checker.populate(memory_addr);
        assert!(memory_addr.is_multiple_of(4), "a real SW event must be word-aligned");

        cols.memory_access.populate(event.mem_access, blu);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: 0,
            a2: 0,
            b: memory_addr.to_le_bytes()[0],
            c: 0b11,
        });

        let addr_bytes = memory_addr.to_le_bytes();
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: addr_bytes[1],
            c: addr_bytes[2],
        });

        cols.most_sig_bytes_zero
            .populate_from_field_element(cols.addr_word[1] + cols.addr_word[2] + cols.addr_word[3]);
        if cols.most_sig_bytes_zero.result == F::ONE {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: NUM_REGISTERS as u8 - 1,
                c: cols.addr_word[0].as_canonical_u32() as u8,
            });
        }
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

        // Structural assumptions this chip's narrower layout relies on: `c` is always the
        // encoded immediate, the opcode is always SW, and every row is a real retired
        // instruction (no synthetic-dependency-row shortcut, unlike AddChip/SubChip).
        builder.when(local.is_real).assert_eq(local.instruction.imm_c, AB::Expr::one());
        builder
            .when(local.is_real)
            .assert_eq(local.instruction.opcode, Opcode::SW.as_field::<AB::F>());

        // Compute and verify the memory address via the ALU table.
        builder.send_alu(
            AB::Expr::from_canonical_u32(Opcode::ADD as u32),
            local.addr_word,
            *local.op_b_access.value(),
            local.instruction.op_c,
            local.is_real,
        );
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.addr_word,
            local.addr_word_range_checker,
            local.is_real.into(),
        );
        builder.slice_range_check_u8(&local.addr_word.0[1..3], local.is_real.into());

        // `addr_word >= NUM_REGISTERS`: if the most significant three bytes are zero, the least
        // significant byte alone must already clear `NUM_REGISTERS - 1`.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            AB::Expr::from_canonical_u8(NUM_REGISTERS as u8 - 1),
            local.addr_word[0],
            local.most_sig_bytes_zero.result,
        );
        builder.when(local.most_sig_bytes_zero.result).assert_one(local.is_real.into());
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.addr_word[1] + local.addr_word[2] + local.addr_word[3],
            local.most_sig_bytes_zero,
            local.is_real.into(),
        );

        // A real SW's address is always word-aligned: `addr_word[0] & 0b11 == 0`.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            AB::Expr::zero(),
            local.addr_word[0],
            AB::Expr::from_canonical_u8(0b11),
            local.is_real,
        );

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        builder.send_program(local.pc, local.instruction, local.is_real);

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::B as u32),
            local.instruction.op_b[0],
            &local.op_b_access,
            local.is_real,
        );

        // `op_a` is a plain, immutable register read -- no register-0 write-zeroing needed since
        // nothing is ever written back.
        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::A as u32),
            local.instruction.op_a,
            &local.op_a_access,
            local.is_real,
        );

        // The stored word is always `op_a`'s value in full.
        builder.when(local.is_real).assert_word_eq(
            local.memory_access.value().map(Into::into),
            local.op_a_access.value().map(Into::into),
        );
        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            local.addr_word.reduce::<AB>(),
            &local.memory_access,
            local.is_real,
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
