//! Implementation to check that b * c = product.
//!
//! See `operations::MulOperation` for the shared arithmetic (sign extension, carry-propagated
//! product). This chip covers real, retired MUL/MULT/MULTU instructions only -- `DivRemChip`'s
//! `c * quotient` overflow check and `MaddsubChip`'s MADD/MADDU/MSUB/MSUBU accumulate-multiply
//! each embed their own copy of `MulOperation` instead of depending on this chip.

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, CompAluEvent, MemoryAccessPosition, MemoryRecordEnum},
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
    operations::MulOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `MulChip`.
pub const NUM_MUL_COLS: usize = size_of::<MulCols<u8>>();

/// A chip that implements multiplication for the opcode MUL, MULT and MULTU.
#[derive(Default)]
pub struct MulChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct MulCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    #[cfg_attr(feature = "picus", picus(input))]
    pub pc: T,
    pub next_pc: T,

    /// The raw fetched instruction.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`/`c`.
    pub reader: RegisterReader<T>,

    /// The output operand.
    pub a: Word<T>,

    /// The `b * c` product (sign-aware): shared arithmetic with `DivRemChip`/`MaddsubChip`'s
    /// embedded copies, but computed locally here -- no cross-chip lookup.
    pub mul_operation: MulOperation<T>,

    /// Flag indicating whether the opcode is `MUL`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_mul: T,

    /// Flag indicating whether the opcode is `MULT`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_mult: T,

    /// Flag indicating whether the opcode is `MULTU`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_multu: T,

    pub is_real: T,

    /// Access to hi register
    pub op_hi_access: MemoryReadWriteCols<T>,

    /// Flag indicating whether the hi_access record is real.
    pub hi_record_is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for MulChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Mul".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        MulCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.mul_events.len(),
            None,
            <MulChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let padded_nb_rows = <MulChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MUL_COLS);
        let nb_rows = input.mul_events.len();
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_MUL_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_MUL_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MulCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.mul_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events, &input.program);
                    } else {
                        // Padding row: force the register reader's b/c memory-access
                        // multiplicities to zero (see cpuchip-migration-register-reader-gotchas
                        // memory).
                        cols.instruction.imm_b = F::ONE;
                        cols.instruction.imm_c = F::ONE;
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_MUL_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.mul_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .mul_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_MUL_COLS];
                    let cols: &mut MulCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect::<Vec<_>>());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.mul_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl MulChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &CompAluEvent,
        cols: &mut MulCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);

        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

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

        cols.hi_record_is_real = F::from_bool(event.hi_record_is_real);
        if event.hi_record_is_real {
            cols.op_hi_access.populate(MemoryRecordEnum::Write(event.hi_record), blu);
        }

        // Only MULT treats its operands as signed 32-bit values; MUL only needs the product's low
        // 32 bits (identical whether the inputs are read as signed or unsigned), and MULTU treats
        // both operands as unsigned.
        let is_signed = event.opcode == Opcode::MULT;
        cols.mul_operation.populate(blu, event.b, event.c, is_signed);

        cols.a = Word(event.a.to_le_bytes().map(F::from_canonical_u8));
        cols.is_real = F::ONE;
        cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
        cols.is_mult = F::from_bool(event.opcode == Opcode::MULT);
        cols.is_multu = F::from_bool(event.opcode == Opcode::MULTU);
    }
}

impl<F> BaseAir<F> for MulChip {
    fn width(&self) -> usize {
        NUM_MUL_COLS
    }
}

impl<AB> Air<AB> for MulChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MulCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        let is_real: AB::Expr = local.is_real.into();

        // Constrain the multiplication operation over `op_b`/`op_c`. Only `MULT` sign-extends.
        let (lo, hi) = MulOperation::<AB::F>::eval(
            builder,
            local.reader.op_b_val(),
            local.reader.op_c_val(),
            local.mul_operation,
            local.is_mult.into(),
            is_real.clone(),
        );

        // Compare the product's low word with the result.
        for i in 0..WORD_SIZE {
            builder.assert_eq(lo[i], local.a[i]);
        }

        // Check that the boolean values are indeed boolean values.
        {
            let booleans =
                [local.is_mul, local.is_mult, local.is_multu, local.is_real, local.hi_record_is_real];
            for boolean in booleans.iter() {
                builder.assert_bool(*boolean);
            }
        }

        // Exactly one of the op codes must be on.
        builder.when(is_real.clone()).assert_one(local.is_mul + local.is_mult + local.is_multu);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        builder.send_program(local.pc, local.instruction, is_real.clone());

        eval_register_reader(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            &local.instruction,
            local.a.map(|x| is_real.clone() * Into::<AB::Expr>::into(x)),
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
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

        // Bind `is_real`'s selector to the row's actual fetched opcode, so a row can't claim the
        // wrong multiply variant while still passing the program lookup.
        builder
            .when(is_real.clone() * local.is_mul)
            .assert_eq(local.instruction.opcode, AB::F::from_canonical_u32(Opcode::MUL as u32));
        builder
            .when(is_real.clone() * local.is_mult)
            .assert_eq(local.instruction.opcode, AB::F::from_canonical_u32(Opcode::MULT as u32));
        builder
            .when(is_real * local.is_multu)
            .assert_eq(local.instruction.opcode, AB::F::from_canonical_u32(Opcode::MULTU as u32));

        // Write the HI register, the register can only be Register::HI（33）.
        builder.eval_memory_access(
            clk_high,
            clk_low + AB::F::from_canonical_u32(MemoryAccessPosition::HI as u32),
            AB::F::from_canonical_u32(33),
            &local.op_hi_access,
            local.hi_record_is_real,
        );

        // Check hi_record_is_real.
        // hi_record_is_real can only be set for MULT and MULTU instruction when is_real = 1.
        builder.when_not(local.is_real).assert_zero(local.hi_record_is_real);
        builder.when(local.hi_record_is_real).assert_one(local.is_mult + local.is_multu);
        // A real MULT/MULTU retirement must always write HI.
        builder
            .when(Into::<AB::Expr>::into(local.is_real) * (local.is_mult + local.is_multu))
            .assert_one(local.hi_record_is_real);
        builder.when(local.hi_record_is_real).assert_word_eq(hi, *local.op_hi_access.value());
    }
}

#[cfg(test)]
mod tests {
    // use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::CompAluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::MulChip;

    #[test]
    fn generate_trace_mul() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::MUL,
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

        // Fill mul_events with 10 MUL events.
        let mut mul_events: Vec<CompAluEvent> = Vec::new();
        for _ in 0..10 {
            mul_events.push(CompAluEvent::new(
                0,
                Opcode::MUL,
                0x80004000,
                0x80000000,
                0xffff8000,
            ));
        }
        shard.mul_events = mul_events;
        let chip = MulChip::default();
        let _trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
    }
}
