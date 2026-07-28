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
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent, MemoryAccessPosition, MemoryRecordEnum},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{
        clk_low_expr, eval_cpu_state, eval_i_type_reader_non_zero, eval_state_chain, CpuState,
        InstructionCols, ITypeReaderNonZero,
    },
    air::ZKMCoreAirBuilder,
    memory::{MemoryCols, MemoryReadCols},
    operations::WordAddressOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `LoadWordChip`.
pub const NUM_LOAD_WORD_COLS: usize = size_of::<LoadWordCols<u8>>();

/// A chip that implements the word-aligned load opcodes LW and LL (the latter is MIPS's atomic
/// load-linked, structurally identical to a plain word load in this executor -- it writes the
/// full loaded word to `op_a` unmodified, with no separate atomicity bookkeeping to speak of).
///
/// LW/LL are always real, retired instructions (memory instructions never produce synthetic
/// dependency rows). Unlike `LoadByteChip`/`LoadHalfChip`/`LoadWordUnalignedChip`, this chip
/// drops every column that only the byte/half/unaligned/atomic-store variants need -- sign extension
/// (`unsigned_mem_val`/`most_sig_bit`/`most_sig_byte`/`mem_value_is_neg`), partial-word merging
/// (`prev_a_val`, the LWL/LWR/SWL/SWR machinery), and the 4-way offset mux (`ls_bits_is_*`,
/// since a real LW/LL always has `addr_word[0] & 0b11 == 0` -- no separate "aligned" address needs
/// deriving from an offset). `c` also never comes from a register for a memory instruction (it's
/// always the encoded immediate offset), so it's read directly off the adapter, same as `AddiChip`.
///
/// A real `lw $zero, ...`/`ll $zero, ...` is routed to `LoadX0Chip` instead (this chip's `op_a` is
/// guaranteed never register 0, letting it use the unmasked `ITypeReaderNonZero`).
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

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReaderNonZero<T>,

    /// The computed, validated memory address (`op_b + op_c`).
    pub word_address: WordAddressOperation<T>,

    /// The memory word being read. A plain read (not read-write): a load never changes memory,
    /// so `value()`/`prev_value()` are structurally the same column here.
    pub memory_access: MemoryReadCols<T>,

    /// Whether this is a real, retired LW instruction.
    pub is_lw: T,
    /// Whether this is a real, retired LL instruction.
    pub is_ll: T,
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

        builder.assert_bool(local.is_lw);
        builder.assert_bool(local.is_ll);
        let is_real = local.is_lw + local.is_ll;
        builder.assert_bool(is_real.clone());

        // The instruction word is reconstructed here rather than stored: `imm_b`/`imm_c`
        // are compile-time constants (this chip only ever sees real, retired LW/LL with a
        // non-zero destination -- any `op_a==0` row is routed to `LoadX0Chip` instead), and
        // `op_a`/`op_b`/`op_c` come from the adapter.
        let opcode = local.is_lw.into() * AB::Expr::from_canonical_u32(Opcode::LW as u32)
            + local.is_ll.into() * AB::Expr::from_canonical_u32(Opcode::LL as u32);
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        let addr_word = WordAddressOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val().map(Into::into),
            local.adapter.op_c.map(Into::into),
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

        // The loaded word is sent directly, unmasked, as `op_a`'s new value -- sound because
        // `ITypeReaderNonZero` guarantees `op_a` is never register 0 (see its doc comment).
        eval_i_type_reader_non_zero(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.memory_access.value().map(Into::into),
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
