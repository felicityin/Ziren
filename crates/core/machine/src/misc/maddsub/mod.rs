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
    events::{ByteLookupEvent, ByteRecord, MemoryAccessPosition, MemoryRecordEnum, MiscEvent},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    adapter::InstructionCols,
    adapter::{
        clk_low_expr, eval_cpu_state, eval_register_reader, eval_state_chain, CpuState,
        RegisterReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, MemoryReadWriteCols},
    operations::{AddDoubleOperation, MulOperation},
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `MaddsubChip`.
pub const NUM_MADDSUB_COLS: usize = size_of::<MaddsubCols<u8>>();

/// A chip that implements the MIPS multiply-accumulate instructions MADD/MADDU/MSUB/MSUBU.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `maddsub_events`. All four opcodes share one circuit via `is_add`/`is_sub`/`is_sign`
/// selectors (the same granularity `AddChip` uses for ADD/ADDU), so they stay one chip rather
/// than splitting further. This is a read-modify-write of `op_a`/`HI` (the accumulate result adds
/// onto the previous `{HI, op_a}` pair), so it needs `prev_a_value`.
#[derive(Default)]
pub struct MaddsubChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct MaddsubCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The raw fetched instruction.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`/`c`.
    pub reader: RegisterReader<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The value of the first operand.
    pub op_a_value: Word<T>,
    pub prev_a_value: Word<T>,
    /// The value of the second operand.
    pub op_b_value: Word<T>,
    /// The value of the third operand.
    pub op_c_value: Word<T>,

    /// The `b * c` product, computed locally (no cross-chip lookup into `MulChip`).
    pub mul_operation: MulOperation<T>,

    /// Add operations of low/high word.
    pub add_operation: AddDoubleOperation<T>,
    /// Add or Sub source value.
    pub src2_hi: Word<T>,
    pub src2_lo: Word<T>,

    /// Access to hi register.
    pub op_hi_access: MemoryReadWriteCols<T>,

    /// MADD/MADDU/MSUB/MSUBU instruction selectors.
    pub is_maddu: T,
    pub is_msubu: T,
    pub is_madd: T,
    pub is_msub: T,
}

impl<F: PrimeField32> MachineAir<F> for MaddsubChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Maddsub".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        MaddsubCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.maddsub_events.len(),
            None,
            <MaddsubChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.maddsub_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <MaddsubChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MADDSUB_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_MADDSUB_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_MADDSUB_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MaddsubCols<F> = row.borrow_mut();

                    if idx < input.maddsub_events.len() {
                        let event = &input.maddsub_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    } else {
                        // Padding row: force the register reader's b/c memory-access
                        // multiplicities to zero (see cpuchip-migration-register-reader-gotchas
                        // memory).
                        cols.instruction.imm_b = F::ONE;
                        cols.instruction.imm_c = F::ONE;
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_MADDSUB_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.maddsub_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl MaddsubChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MiscEvent,
        cols: &mut MaddsubCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);
        cols.instruction.populate(&instruction);

        *cols.reader.op_a_access.value_mut() = event.a.into();
        *cols.reader.op_b_access.value_mut() = event.b.into();
        *cols.reader.op_c_access.value_mut() = event.c.into();

        if let Some(record) = event.a_record {
            cols.reader.op_a_access.populate(record, blu);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.b_record {
            cols.reader.op_b_access.populate(record, blu);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.c_record {
            cols.reader.op_c_access.populate(record, blu);
        }
        cols.reader.populate_op_a_range_checks(blu);

        cols.op_a_value = event.a.into();
        cols.op_b_value = event.b.into();
        cols.op_c_value = event.c.into();
        cols.prev_a_value = event.prev_a.into();

        cols.is_maddu = F::from_bool(matches!(event.opcode, Opcode::MADDU));
        cols.is_msubu = F::from_bool(matches!(event.opcode, Opcode::MSUBU));
        cols.is_madd = F::from_bool(matches!(event.opcode, Opcode::MADD));
        cols.is_msub = F::from_bool(matches!(event.opcode, Opcode::MSUB));

        let is_sign = event.opcode == Opcode::MADD || event.opcode == Opcode::MSUB;
        let (mul_lo, mul_hi) = cols.mul_operation.populate(blu, event.b, event.c, is_sign);
        let multiply = ((mul_hi as u64) << 32) + (mul_lo as u64);

        let is_add = event.opcode == Opcode::MADDU || event.opcode == Opcode::MADD;
        let src2_lo = if is_add { event.prev_a } else { event.a };
        let src2_hi = if is_add { event.hi_record.prev_value } else { event.hi_record.value };
        let _ = cols
            .add_operation
            .populate(blu, multiply, ((src2_hi as u64) << 32) + (src2_lo as u64));
        cols.src2_lo = Word::from(src2_lo);
        cols.src2_hi = Word::from(src2_hi);

        // For maddu/msubu instructions, pass in a dummy byte lookup vector. This maddu/msubu
        // instruction chip also has an op_hi_access field that will be populated and that will
        // contribute to the byte lookup dependencies.
        cols.op_hi_access.populate(MemoryRecordEnum::Write(event.hi_record), blu);
    }
}

impl<F> BaseAir<F> for MaddsubChip {
    fn width(&self) -> usize {
        NUM_MADDSUB_COLS
    }
}

impl<AB> Air<AB> for MaddsubChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MaddsubCols<AB::Var> = (*local).borrow();

        let is_real = local.is_maddu + local.is_msubu + local.is_madd + local.is_msub;
        builder.assert_bool(local.is_maddu);
        builder.assert_bool(local.is_msubu);
        builder.assert_bool(local.is_madd);
        builder.assert_bool(local.is_msub);
        builder.assert_bool(is_real.clone());

        let is_sign = local.is_madd + local.is_msub;
        let is_add = local.is_maddu + local.is_madd;
        let is_sub = local.is_msubu + local.is_msub;

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        builder.send_program(local.pc, local.instruction, is_real.clone());

        // MADD-family is a read-modify-write of `op_a` (`prev_a_value` cross-checked against the
        // register's real previous value).
        eval_register_reader(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            &local.instruction,
            local.op_a_value.map(Into::into),
            local.prev_a_value.map(Into::into),
            is_real.clone(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.clone());

        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk_high.clone(),
            clk_low.clone(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            is_real.clone(),
        );

        builder
            .when(is_real.clone())
            .assert_word_eq(local.reader.op_b_val(), local.op_b_value.map(Into::into));
        builder
            .when(is_real.clone())
            .assert_word_eq(local.reader.op_c_val(), local.op_c_value.map(Into::into));

        // Bind each opcode flag to the row's actual fetched opcode, so a row can't claim the
        // wrong MADD-family variant while still passing the program lookup.
        builder
            .when(local.is_maddu)
            .assert_eq(local.instruction.opcode, Opcode::MADDU.as_field::<AB::F>());
        builder
            .when(local.is_msubu)
            .assert_eq(local.instruction.opcode, Opcode::MSUBU.as_field::<AB::F>());
        builder
            .when(local.is_madd)
            .assert_eq(local.instruction.opcode, Opcode::MADD.as_field::<AB::F>());
        builder
            .when(local.is_msub)
            .assert_eq(local.instruction.opcode, Opcode::MSUB.as_field::<AB::F>());

        // Compute b * c locally (no cross-chip lookup into `MulChip`).
        let (mul_lo, mul_hi) = MulOperation::<AB::F>::eval(
            builder,
            local.op_b_value,
            local.op_c_value,
            local.mul_operation,
            is_sign,
            is_real.clone(),
        );

        for i in 0..WORD_SIZE {
            builder.when(is_real.clone()).assert_eq(
                local.src2_hi[i],
                local.op_hi_access.prev_value[i] * is_add.clone()
                    + (*local.op_hi_access.value())[i] * is_sub.clone(),
            );
            builder.when(is_real.clone()).assert_eq(
                local.src2_lo[i],
                local.prev_a_value[i] * is_add.clone() + local.op_a_value[i] * is_sub.clone(),
            );
        }

        AddDoubleOperation::<AB::F>::eval(
            builder,
            mul_lo,
            mul_hi,
            local.src2_lo,
            local.src2_hi,
            local.add_operation,
            is_real.clone(),
        );

        builder.when(is_add.clone()).assert_word_eq(local.op_a_value, local.add_operation.value);
        builder
            .when(is_add)
            .assert_word_eq(*local.op_hi_access.value(), local.add_operation.value_hi);

        builder.when(is_sub.clone()).assert_word_eq(local.prev_a_value, local.add_operation.value);
        builder
            .when(is_sub)
            .assert_word_eq(local.op_hi_access.prev_value, local.add_operation.value_hi);

        builder.eval_memory_access(
            clk_high,
            clk_low + AB::F::from_canonical_u32(MemoryAccessPosition::HI as u32),
            AB::F::from_canonical_u32(33),
            &local.op_hi_access,
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{
        events::{MemoryWriteRecord, MiscEvent},
        ExecutionRecord, Instruction, Opcode, Program,
    };
    use zkm_hypercube::air::MachineAir;

    use super::MaddsubChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::MADDU,
                op_a: 5,
                op_b: 8,
                op_c: 9,
                imm_b: false,
                imm_c: false,
                raw: None,
            }],
            pc_start: 0,
            pc_base: 0,
            next_pc: 4,
            image: Default::default(),
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.maddsub_events = vec![MiscEvent::new(
            0,
            0,
            4,
            Opcode::MADDU,
            32,
            10,
            20,
            5,
            MemoryWriteRecord::new(0, 1, 0, 0),
        )];
        let chip = MaddsubChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
