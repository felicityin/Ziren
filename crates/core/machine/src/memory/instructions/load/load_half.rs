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
    ByteOpcode, ExecutionRecord, Opcode, Program,
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

/// The number of main trace columns for `LoadHalfChip`.
pub const NUM_LOAD_HALF_COLS: usize = size_of::<LoadHalfCols<u8>>();

/// A chip that implements the halfword-load opcodes LH/LHU.
///
/// See `LoadByteChip`'s doc comment for the general rationale. Unlike `LoadByteChip`, only
/// offsets `0`/`2` are valid (the accessed halfword must be 2-byte aligned), so the offset flags
/// `ls_bits_is_one`/`ls_bits_is_three` are constrained to `0`.
#[derive(Default)]
pub struct LoadHalfChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct LoadHalfCols<T: Copy> {
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

    /// The zero-extended halfword value loaded from memory (before sign extension).
    pub unsigned_mem_val: Word<T>,

    /// The most significant bit of `unsigned_mem_val[1]`. Only meaningful for `LH`.
    pub most_sig_bit: T,

    /// The most significant byte of `unsigned_mem_val`, i.e. `unsigned_mem_val[1]`.
    pub most_sig_byte: T,

    /// Whether the loaded value is negative (`LH` only; always `0` for `LHU`).
    pub mem_value_is_neg: T,

    /// Whether this is a load half (signed) instruction.
    pub is_lh: T,
    /// Whether this is a load half unsigned instruction.
    pub is_lhu: T,

    /// Whether this row is a real, retired LH/LHU instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for LoadHalfChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "LoadHalf".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.load_half_events.len(),
            None,
            <LoadHalfChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.load_half_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <LoadHalfChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_HALF_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_HALF_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_HALF_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadHalfCols<F> = row.borrow_mut();

                    if idx < input.load_half_events.len() {
                        let event = &input.load_half_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_LOAD_HALF_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.load_half_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadHalfChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadHalfCols<F>,
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
        let value = match (addr_ls_two_bits >> 1) % 2 {
            0 => mem_value & 0x0000_FFFF,
            1 => (mem_value & 0xFFFF_0000) >> 16,
            _ => unreachable!(),
        };
        cols.unsigned_mem_val = value.into();

        cols.is_lh = F::from_bool(matches!(event.opcode, Opcode::LH));
        cols.is_lhu = F::from_bool(matches!(event.opcode, Opcode::LHU));

        if matches!(event.opcode, Opcode::LH) {
            let most_sig_mem_value_byte = cols.unsigned_mem_val.to_u32().to_le_bytes()[1];
            let most_sig_mem_value_bit = most_sig_mem_value_byte >> 7;
            if most_sig_mem_value_bit == 1 {
                cols.mem_value_is_neg = F::ONE;
            }
            cols.most_sig_byte = F::from_canonical_u8(most_sig_mem_value_byte);
            cols.most_sig_bit = F::from_canonical_u8(most_sig_mem_value_bit);

            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: most_sig_mem_value_bit as u16,
                a2: 0,
                b: most_sig_mem_value_byte,
                c: 0,
            });
        }
    }
}

impl<F> BaseAir<F> for LoadHalfChip {
    fn width(&self) -> usize {
        NUM_LOAD_HALF_COLS
    }
}

impl<AB> Air<AB> for LoadHalfChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadHalfCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_lh);
        builder.assert_bool(local.is_lhu);
        let is_real = local.is_lh + local.is_lhu;
        builder.assert_bool(is_real.clone());

        let opcode = local.is_lh.into() * AB::Expr::from_canonical_u32(Opcode::LH as u32)
            + local.is_lhu.into() * AB::Expr::from_canonical_u32(Opcode::LHU as u32);
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
            local.adapter.op_b_val(),
            local.adapter.op_c,
            local.word_address,
            is_real.clone(),
        );

        // A halfword access must be 2-byte aligned: offset 1 and 3 are invalid.
        builder
            .when(is_real.clone())
            .assert_zero(local.word_address.ls_bits_is_one + local.word_address.ls_bits_is_three);

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            addr_aligned.into(),
            &local.memory_access,
            is_real.clone(),
        );

        // Compute the halfword value selected by the offset mux.
        let mem_val = *local.memory_access.value();
        let offset_is_zero = AB::Expr::one()
            - local.word_address.ls_bits_is_one.into()
            - local.word_address.ls_bits_is_two.into()
            - local.word_address.ls_bits_is_three.into();
        let use_lower_half = offset_is_zero;
        let use_upper_half = local.word_address.ls_bits_is_two;
        let half_value = Word([
            use_lower_half.clone() * mem_val[0].into() + use_upper_half.into() * mem_val[2].into(),
            use_lower_half * mem_val[1].into() + use_upper_half.into() * mem_val[3].into(),
            AB::Expr::zero(),
            AB::Expr::zero(),
        ]);
        builder.when(is_real.clone()).assert_word_eq(half_value, local.unsigned_mem_val.map(Into::into));

        // Sign extension (LH only; LHU always treats the value as unsigned).
        builder.assert_eq(local.mem_value_is_neg, local.is_lh * local.most_sig_bit);
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.most_sig_bit,
            local.most_sig_byte,
            AB::Expr::zero(),
            local.is_lh,
        );
        builder.assert_eq(local.most_sig_byte, local.is_lh * local.unsigned_mem_val[1]);

        let signed_value =
            Word([AB::Expr::zero(), AB::Expr::zero(), local.is_lh.into(), AB::Expr::zero()]);
        builder.send_alu(
            Opcode::SUB.as_field::<AB::F>(),
            local.adapter.op_a_access.value.map(Into::into),
            local.unsigned_mem_val,
            signed_value,
            local.mem_value_is_neg.into(),
        );

        let mem_value_is_pos = (local.is_lh.into() - local.mem_value_is_neg.into()) + local.is_lhu.into();
        builder
            .when(mem_value_is_pos)
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

    use super::LoadHalfChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::LH,
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
        shard.load_half_events = vec![MemInstrEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            opcode: Opcode::LH,
            a: 42,
            b: 100,
            c: 4,
            mem_access: MemoryReadRecord::new(0xDEAD_BE42, 5, 0).into(),
            prev_a_val: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = LoadHalfChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
