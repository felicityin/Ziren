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
        clk_low_expr, eval_cpu_state, eval_i_type_reader, eval_state_chain, CpuState,
        InstructionCols, ITypeReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, MemoryReadCols},
    operations::UnalignedWordAddressOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `LoadWordUnalignedChip`.
pub const NUM_LOAD_WORD_UNALIGNED_COLS: usize = size_of::<LoadWordUnalignedCols<u8>>();

/// A chip that implements the unaligned partial-word load opcodes LWL/LWR ("load word
/// left"/"load word right").
///
/// Unlike `LoadByteChip`/`LoadHalfChip`, these opcodes merge the loaded bytes with the
/// register's *previous* value (`adapter.op_a_access.prev_value` -- the same value
/// `RegisterWriteAccessCols` already witnesses for every register write, so no separate
/// `prev_a_val` column is needed) rather than sign-extending: LWL/LWR always write a full,
/// unsigned word, so `mem_value_is_pos` is unconditionally true here and no SUB-based sign
/// computation is needed either.
#[derive(Default)]
pub struct LoadWordUnalignedChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct LoadWordUnalignedCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReader<T>,

    /// The computed, validated (possibly unaligned) memory address.
    pub word_address: UnalignedWordAddressOperation<T>,

    /// The memory word being read. A plain read (not read-write): a load never changes memory,
    /// so `value()`/`prev_value()` are structurally the same column here.
    pub memory_access: MemoryReadCols<T>,

    /// The word value loaded from memory, merged with `op_a`'s previous value.
    pub unsigned_mem_val: Word<T>,

    /// Whether this is a load word left instruction.
    pub is_lwl: T,
    /// Whether this is a load word right instruction.
    pub is_lwr: T,

    /// Whether this row is a real, retired LWL/LWR instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for LoadWordUnalignedChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "LoadWordUnaligned".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.load_word_unaligned_events.len(),
            None,
            <LoadWordUnalignedChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size =
            std::cmp::max(input.load_word_unaligned_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <LoadWordUnalignedChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_WORD_UNALIGNED_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_WORD_UNALIGNED_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_WORD_UNALIGNED_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadWordUnalignedCols<F> = row.borrow_mut();

                    if idx < input.load_word_unaligned_events.len() {
                        let event = &input.load_word_unaligned_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_LOAD_WORD_UNALIGNED_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.load_word_unaligned_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadWordUnalignedChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadWordUnalignedCols<F>,
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

        let memory_addr = cols.word_address.populate(blu, event.b, event.c);
        let addr_ls_two_bits = (memory_addr % 4) as u8;

        if let MemoryRecordEnum::Read(record) = event.mem_access {
            cols.memory_access.populate(record, blu);
        }

        let mem_value = event.mem_access.value();
        cols.is_lwl = F::from_bool(matches!(event.opcode, Opcode::LWL));
        cols.is_lwr = F::from_bool(matches!(event.opcode, Opcode::LWR));

        cols.unsigned_mem_val = match event.opcode {
            Opcode::LWL => {
                // let val = mem << (24 - (rs & 3) * 8);
                // let mask = 0xFFFFFFFF_u32 << (24 - (rs & 3) * 8);
                // (rt & (!mask)) | val
                let val = mem_value << (24 - addr_ls_two_bits * 8);
                let mask = 0xFFFF_FFFF_u32 << (24 - addr_ls_two_bits * 8);
                ((event.prev_a_val & (!mask)) | val).into()
            }
            Opcode::LWR => {
                // let val = mem >> ((rs & 3) * 8);
                // let mask = 0xFFFFFFFF_u32 >> ((rs & 3) * 8);
                // (rt & (!mask)) | val
                let val = mem_value >> (addr_ls_two_bits * 8);
                let mask = 0xFFFF_FFFF_u32 >> (addr_ls_two_bits * 8);
                ((event.prev_a_val & (!mask)) | val).into()
            }
            _ => unreachable!(),
        };
    }
}

impl<F> BaseAir<F> for LoadWordUnalignedChip {
    fn width(&self) -> usize {
        NUM_LOAD_WORD_UNALIGNED_COLS
    }
}

impl<AB> Air<AB> for LoadWordUnalignedChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadWordUnalignedCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_lwl);
        builder.assert_bool(local.is_lwr);
        let is_real = local.is_lwl + local.is_lwr;
        builder.assert_bool(is_real.clone());

        let opcode = local.is_lwl.into() * AB::Expr::from_canonical_u32(Opcode::LWL as u32)
            + local.is_lwr.into() * AB::Expr::from_canonical_u32(Opcode::LWR as u32);
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

        let addr_aligned = UnalignedWordAddressOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val().map(Into::into),
            local.adapter.op_c.map(Into::into),
            local.word_address,
            is_real.clone(),
        );

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            addr_aligned.into(),
            &local.memory_access,
            is_real.clone(),
        );

        let mem_val = *local.memory_access.value();
        let prev_a_val = local.adapter.op_a_access.prev_value;
        let offset_is_zero = AB::Expr::one()
            - local.word_address.ls_bits_is_one.into()
            - local.word_address.ls_bits_is_two.into()
            - local.word_address.ls_bits_is_three.into();

        // Compute the expected loaded value for a LWR instruction.
        let lwr_expected_load_value = Word([
            mem_val[0].into() * offset_is_zero.clone()
                + mem_val[1].into() * local.word_address.ls_bits_is_one
                + mem_val[2].into() * local.word_address.ls_bits_is_two
                + mem_val[3].into() * local.word_address.ls_bits_is_three,
            mem_val[1].into() * offset_is_zero.clone()
                + mem_val[2].into() * local.word_address.ls_bits_is_one
                + mem_val[3].into() * local.word_address.ls_bits_is_two
                + prev_a_val[1].into() * local.word_address.ls_bits_is_three,
            mem_val[2].into() * offset_is_zero.clone()
                + mem_val[3].into() * local.word_address.ls_bits_is_one
                + prev_a_val[2].into()
                    * (AB::Expr::one()
                        - local.word_address.ls_bits_is_one.into()
                        - offset_is_zero.clone()),
            mem_val[3].into() * offset_is_zero.clone()
                + prev_a_val[3].into() * (AB::Expr::one() - offset_is_zero.clone()),
        ]);
        builder
            .when(local.is_lwr)
            .assert_word_eq(local.unsigned_mem_val, lwr_expected_load_value);

        // Compute the expected loaded value for a LWL instruction.
        let lwl_expected_load_value = Word([
            mem_val[0].into() * local.word_address.ls_bits_is_three
                + prev_a_val[0].into() * (AB::Expr::one() - local.word_address.ls_bits_is_three),
            mem_val[1].into() * local.word_address.ls_bits_is_three
                + mem_val[0].into() * local.word_address.ls_bits_is_two
                + prev_a_val[1].into() * local.word_address.ls_bits_is_one
                + prev_a_val[1].into() * offset_is_zero.clone(),
            mem_val[2].into() * local.word_address.ls_bits_is_three
                + mem_val[1].into() * local.word_address.ls_bits_is_two
                + mem_val[0].into() * local.word_address.ls_bits_is_one
                + prev_a_val[2].into() * offset_is_zero.clone(),
            mem_val[3].into() * local.word_address.ls_bits_is_three
                + mem_val[2].into() * local.word_address.ls_bits_is_two
                + mem_val[1].into() * local.word_address.ls_bits_is_one
                + mem_val[0].into() * offset_is_zero,
        ]);
        builder
            .when(local.is_lwl)
            .assert_word_eq(local.unsigned_mem_val, lwl_expected_load_value);

        // LWL/LWR always write a full, unsigned word -- no sign extension needed.
        builder
            .when(is_real.clone())
            .assert_word_eq(local.unsigned_mem_val, local.adapter.op_a_access.value.map(Into::into));

        eval_i_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.adapter.op_a_access.value.map(Into::into),
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

    use super::LoadWordUnalignedChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::LWL,
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
        shard.load_word_unaligned_events = vec![MemInstrEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            opcode: Opcode::LWL,
            a: 42,
            b: 100,
            c: 4,
            mem_access: MemoryReadRecord::new(0xDEAD_BE42, 5, 0).into(),
            prev_a_val: 0x1234_5678,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = LoadWordUnalignedChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
