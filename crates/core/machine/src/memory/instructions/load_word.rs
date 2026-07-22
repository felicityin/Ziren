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

/// The number of main trace columns for `LoadWordChip`.
pub const NUM_LOAD_WORD_COLS: usize = size_of::<LoadWordCols<u8>>();

/// A chip that implements the word-aligned load opcode LW.
///
/// LW is always a real, retired instruction (memory instructions never produce synthetic
/// dependency rows). Unlike the general `MemoryInstructionsChip`, this chip drops every
/// column that only the byte/half/unaligned/atomic variants need -- sign extension
/// (`unsigned_mem_val`/`most_sig_bit`/`most_sig_byte`/`mem_value_is_neg`), partial-word merging
/// (`prev_a_val`, the LWL/LWR/SWL/SWR machinery), and the 4-way offset mux (`ls_bits_is_*`,
/// since a real LW always has `addr_word[0] & 0b11 == 0` -- no separate "aligned" address needs
/// deriving from an offset). `c` also never comes from a register for a memory instruction (it's
/// always the encoded immediate offset), so it's read directly off `InstructionCols::op_c`, same
/// as `AddiChip`.
#[derive(Default)]
pub struct LoadWordChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct LoadWordCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The raw fetched instruction, used for the program lookup and register/immediate
    /// resolution.
    pub instruction: InstructionCols<T>,

    /// Register `a` write access (the loaded word).
    pub op_a_access: MemoryReadWriteCols<T>,
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

    /// The memory word being read. A plain read (not read-write): a load never changes memory,
    /// so `value()`/`prev_value()` are structurally the same column here.
    pub memory_access: MemoryReadCols<T>,

    /// Whether this row is a real, retired LW instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for LoadWordChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "LoadWord".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.load_word_events.len(),
            None,
            <LoadWordChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.load_word_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <LoadWordChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_WORD_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_WORD_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_WORD_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadWordCols<F> = row.borrow_mut();

                    if idx < input.load_word_events.len() {
                        let event = &input.load_word_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_LOAD_WORD_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.load_word_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadWordChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadWordCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);
        cols.is_real = F::ONE;

        let instruction = program.fetch(event.pc);
        cols.instruction.populate(&instruction);

        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        if let Some(record) = event.a_record {
            cols.op_a_access.populate(record, blu);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.b_record {
            cols.op_b_access.populate(record, blu);
        }

        let memory_addr = event.b.wrapping_add(event.c);
        cols.addr_word = memory_addr.into();
        cols.addr_word_range_checker.populate(memory_addr);
        assert!(memory_addr.is_multiple_of(4), "a real LW event must be word-aligned");

        if let MemoryRecordEnum::Read(record) = event.mem_access {
            cols.memory_access.populate(record, blu);
        }

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

        // Matches `slice_range_check_u8(&op_a_access.access.value.0, is_real)` in `eval()`.
        let op_a_bytes = cols.op_a_access.access.value.0.map(|x| x.as_canonical_u32() as u8);
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: op_a_bytes[0],
            c: op_a_bytes[1],
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: op_a_bytes[2],
            c: op_a_bytes[3],
        });
    }
}

impl<F> BaseAir<F> for LoadWordChip {
    fn width(&self) -> usize {
        NUM_LOAD_WORD_COLS
    }
}

impl<AB> Air<AB> for LoadWordChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadWordCols<AB::Var> = (*local).borrow();


        builder.assert_bool(local.is_real);

        // Structural assumptions this chip's narrower layout relies on: `c` is always the
        // encoded immediate, the opcode is always LW, and every row is a real retired
        // instruction (no synthetic-dependency-row shortcut, unlike AddChip/SubChip).
        builder.when(local.is_real).assert_eq(local.instruction.imm_c, AB::Expr::one());
        builder
            .when(local.is_real)
            .assert_eq(local.instruction.opcode, Opcode::LW.as_field::<AB::F>());

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

        // A real LW's address is always word-aligned: `addr_word[0] & 0b11 == 0`.
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

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            local.addr_word.reduce::<AB>(),
            &local.memory_access,
            local.is_real,
        );

        // If we are writing to register 0, the new value must be zero; otherwise it must equal
        // the loaded word.
        builder.when(local.instruction.op_a_0).assert_word_zero(*local.op_a_access.value());
        builder.when_not(local.instruction.op_a_0).assert_word_eq(
            local.memory_access.value().map(Into::into),
            *local.op_a_access.value(),
        );

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::A as u32),
            local.instruction.op_a,
            &local.op_a_access,
            local.is_real,
        );
        builder.slice_range_check_u8(&local.op_a_access.access.value.0, local.is_real);

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
        events::{MemInstrEvent, MemoryReadRecord},
        ExecutionRecord, Instruction, Opcode, Program,
    };
    use zkm_hypercube::air::MachineAir;

    use super::LoadWordChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::LW,
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
        shard.load_word_events = vec![MemInstrEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            opcode: Opcode::LW,
            a: 42,
            b: 100,
            c: 4,
            mem_access: MemoryReadRecord::new(42, 5, 0).into(),
            prev_a_val: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = LoadWordChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
