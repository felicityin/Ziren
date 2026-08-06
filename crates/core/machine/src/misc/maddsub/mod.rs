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
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, MemoryReadWriteCols, RegisterAccessCols, RegisterWriteAccessCols},
    operations::{AddDoubleOperation, MulOperation},
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `MaddsubChip`.
pub const NUM_MADDSUB_COLS: usize = size_of::<MaddsubCols<u8>>();

/// A chip that implements the MIPS multiply-accumulate instructions MADD/MADDU/MSUB/MSUBU.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `maddsub_events`. All four opcodes share one circuit via `is_add`/`is_sub`/`is_sign`
/// selectors (the same granularity `AddChip` uses for ADD/ADDU), so they stay one chip rather
/// than splitting further.
///
/// MADD/MADDU/MSUB/MSUBU always decode with `op_a=32` (MIPS's HI/LO-style multiply-accumulate,
/// register 32 being `LO`) -- a compile-time constant, never a witnessed index, and never
/// register 0, so no `AluX0Chip` routing is needed. `op_b`/`op_c` are always registers (MIPS has
/// no MADDI), so they use the cheap [`RegisterAccessCols`] scheme. This is a read-modify-write of
/// `op_a`/`HI` (the accumulate result adds onto the previous `{HI, op_a}` pair): the written value
/// is a masked/muxed expression (depending on `is_add`/`is_sub`), degree 2, which can't be sent
/// directly as a lookup value the way an `RTypeReader`-style chip does (see
/// `RegisterWriteAccessCols`'s doc comment), so `op_a` uses that instead. `HI`'s own
/// read-modify-write stays on the general [`MemoryReadWriteCols`] scheme (unmigrated, matching
/// `MulChip`/`DivRemChip`'s own HI access).
#[derive(Default)]
pub struct MaddsubChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct MaddsubCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// `op_a`'s access (register 32, a compile-time constant -- see this chip's doc comment). A
    /// read-modify-write: its witnessed `value` is masked/muxed depending on `is_add`/`is_sub`,
    /// so it needs `RegisterWriteAccessCols` rather than a directly-fed lookup value.
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// The register index of `op_c` (read).
    pub op_c: T,
    pub op_c_access: RegisterAccessCols<T>,

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
        let nb_rows = next_multiple_of_32(
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

        // Every `maddsub_events` row is a real, retired instruction -- nothing ever produces a
        // synthetic dependency row here.
        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);

        if let Some(record) = event.a_record {
            cols.op_a_access.populate(record, blu);
        }

        cols.op_b = F::from_canonical_u32(instruction.op_b);
        if let Some(record) = event.b_record {
            cols.op_b_access.populate(record, blu);
        }

        cols.op_c = F::from_canonical_u32(instruction.op_c);
        if let Some(record) = event.c_record {
            cols.op_c_access.populate(record, blu);
        }

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

        let op_b_val = local.op_b_access.prev_value;
        let op_c_val = local.op_c_access.prev_value;
        let written_value: Word<AB::Expr> = local.op_a_access.value.map(Into::into);
        let prev_a_val: Word<AB::Expr> = local.op_a_access.prev_value.map(Into::into);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode` is a degree-1
        // linear combination of the one-hot selectors above, `op_a`/`op_a_0`/`imm_b`/`imm_c` are
        // compile-time constants (this chip only ever sees `op_a=32` and register-register
        // MADD/MADDU/MSUB/MSUBU -- see this chip's doc comment), and `op_b`/`op_c` are
        // zero-extended from their register-index columns.
        let opcode = local.is_maddu * Opcode::MADDU.as_field::<AB::F>()
            + local.is_msubu * Opcode::MSUBU.as_field::<AB::F>()
            + local.is_madd * Opcode::MADD.as_field::<AB::F>()
            + local.is_msub * Opcode::MSUB.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: AB::Expr::from_canonical_u32(32),
            op_b: Word::extend_var::<AB>(local.op_b),
            op_c: Word::extend_var::<AB>(local.op_c),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::zero(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        // Register positions must be read/written in the order C, B, A (see
        // `MemoryAccessPosition`'s doc comment); each gets its own `clk_low` offset, matching the
        // executor's own `rr_traced`/`rw_traced` timestamps for these accesses.
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
            local.op_c.into(),
            &local.op_c_access,
            is_real.clone(),
        );
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            local.op_b.into(),
            &local.op_b_access,
            is_real.clone(),
        );
        builder.eval_register_access_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            AB::Expr::from_canonical_u32(32),
            &local.op_a_access,
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

        // Compute b * c locally (no cross-chip lookup into `MulChip`).
        let (mul_lo, mul_hi) = MulOperation::<AB::F>::eval(
            builder,
            op_b_val,
            op_c_val,
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
                prev_a_val[i].clone() * is_add.clone() + written_value[i].clone() * is_sub.clone(),
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

        builder.when(is_add.clone()).assert_word_eq(written_value.clone(), local.add_operation.value);
        builder
            .when(is_add)
            .assert_word_eq(*local.op_hi_access.value(), local.add_operation.value_hi);

        builder.when(is_sub.clone()).assert_word_eq(prev_a_val, local.add_operation.value);
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
                op_a: 32,
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
