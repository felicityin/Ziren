//! Implementation to check that b * c = product.
//!
//! We first extend the operands to 64 bits. We sign-extend them if the op code is signed. Then we
//! calculate the un-carried product and propagate the carry. Finally, we check that the appropriate
//! bits of the product match the result.
//!
//! b_64 = sign_extend(b) if signed operation else b
//! c_64 = sign_extend(c) if signed operation else c
//!
//! m = []
//! # 64-bit integers have 8 limbs.
//! # Calculate un-carried product.
//! for i in 0..8:
//!     for j in 0..8:
//!         if i + j < 8:
//!             m[i + j] += b_64[i] * c_64[j]
//!
//! # Propagate carry
//! for i in 0..8:
//!     x = m[i]
//!     if i > 0:
//!         x += carry[i - 1]
//!     carry[i] = x / 256
//!     m[i] = x % 256
//!
//! assert_eq(a, m[0..4])
//!
//! if mult or multu:
//!     assert_eq(hi, m[4..8])

mod utils;

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
    ByteOpcode, ExecutionRecord, Opcode, Program, UNUSED_PC,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{
    air::{MachineAir, PublicValues, ZKM_PROOF_NUM_PV_ELTS},
    word::Word,
};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    adapter::InstructionCols,
    adapter::{
        clk_high_expr, clk_low_expr, eval_cpu_state, eval_register_reader, eval_state_chain, CpuState, RegisterReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    alu::mul::utils::get_msb,
    memory::{MemoryCols, MemoryReadWriteCols},
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `MulChip`.
pub const NUM_MUL_COLS: usize = size_of::<MulCols<u8>>();

/// The number of digits in the product is at most the sum of the number of digits in the
/// multiplicands.
pub const PRODUCT_SIZE: usize = 8;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// The mask for a byte.
pub const BYTE_MASK: u8 = 0xff;

/// A chip that implements multiplication for the opcode MUL, MULT and MULTU.
///
/// As with `AddChip`, not every row is a real retired instruction: `DivRem`'s
/// `c * quotient` check and `misc/others`'s MADD/MADDU/MSUB/MSUBU dependency checks reuse this
/// chip's arithmetic circuit for internal MULT/MULTU checks at the `UNUSED_PC` sentinel. `MUL`
/// itself is never synthetic (no chip depends on it). `is_real_instruction` distinguishes real
/// retirements from synthetic dependency rows.
#[derive(Default)]
pub struct MulChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct MulCols<T: Copy> {
    /// The current shard and clk. Only meaningful when `is_real_instruction` is set.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    #[cfg_attr(feature = "picus", picus(input))]
    pub pc: T,
    pub next_pc: T,

    /// The raw fetched instruction. Only meaningful when `is_real_instruction` is set.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`/`c`. Only meaningful when `is_real_instruction` is
    /// set.
    pub reader: RegisterReader<T>,

    /// Whether this row is a real, retired MUL/MULT/MULTU instruction (as opposed to an
    /// internal dependency check from another chip, or padding).
    pub is_real_instruction: T,

    /// The upper bits of the output operand.
    pub hi: Word<T>,

    /// The output operand.
    pub a: Word<T>,

    /// The first input operand.
    pub b: Word<T>,

    /// The second input operand.
    pub c: Word<T>,

    /// Trace.
    pub carry: [T; PRODUCT_SIZE],

    /// An array storing the product of `b * c` after the carry propagation.
    pub product: [T; PRODUCT_SIZE],

    /// The most significant bit of `b`.
    pub b_msb: T,

    /// The most significant bit of `c`.
    pub c_msb: T,

    /// The sign extension of `b`.
    pub b_sign_extend: T,

    /// The sign extension of `c`.
    pub c_sign_extend: T,

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
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        // Default: not a real instruction, so the register reader's b/c memory accesses have
        // zero multiplicity unless overwritten by a real fetched instruction's actual immediate
        // flags just below.
        cols.instruction.imm_b = F::ONE;
        cols.instruction.imm_c = F::ONE;

        let is_real_instruction = event.pc != UNUSED_PC;
        cols.is_real_instruction = F::from_bool(is_real_instruction);
        if is_real_instruction {
            cols.state.populate(blu, event.shard, event.clk);

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
        }

        cols.hi_record_is_real = F::from_bool(event.hi_record_is_real);
        if event.hi_record_is_real {
            // For madd[u]/msub[u] instructions, pass in a dummy byte lookup vector.  This madd[u]/msub[u]
            // instruction chip also has a op_hi_access field that will be populated and that will contribute
            // to the byte lookup dependencies.
            cols.op_hi_access.populate(MemoryRecordEnum::Write(event.hi_record), blu);
        }

        let hi_word = event.hi.to_le_bytes();
        let a_word = event.a.to_le_bytes();
        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        let mut b = b_word.to_vec();
        let mut c = c_word.to_vec();

        // Handle b and c's signs.
        {
            let b_msb = get_msb(b_word);
            cols.b_msb = F::from_canonical_u8(b_msb);
            let c_msb = get_msb(c_word);
            cols.c_msb = F::from_canonical_u8(c_msb);

            // If b is signed and it is negative, sign extend b.
            if event.opcode == Opcode::MULT && b_msb == 1 {
                cols.b_sign_extend = F::ONE;
                b.resize(PRODUCT_SIZE, BYTE_MASK);
            }

            // If c is signed and it is negative, sign extend c.
            if event.opcode == Opcode::MULT && c_msb == 1 {
                cols.c_sign_extend = F::ONE;
                c.resize(PRODUCT_SIZE, BYTE_MASK);
            }

            // Insert the MSB lookup events.
            {
                let words = [b_word, c_word];
                let mut blu_events: Vec<ByteLookupEvent> = vec![];
                for word in words.iter() {
                    let most_significant_byte = word[WORD_SIZE - 1];
                    blu_events.push(ByteLookupEvent {
                        opcode: ByteOpcode::MSB,
                        a1: get_msb(*word) as u16,
                        a2: 0,
                        b: most_significant_byte,
                        c: 0,
                    });
                }
                blu.add_byte_lookup_events(blu_events);
            }
        }

        let mut product = [0u32; PRODUCT_SIZE];
        for i in 0..b.len() {
            for j in 0..c.len() {
                if i + j < PRODUCT_SIZE {
                    product[i + j] += (b[i] as u32) * (c[j] as u32);
                }
            }
        }

        // Calculate the correct product using the `product` array. We store the
        // correct carry value for verification.
        let base = (1 << BYTE_SIZE) as u32;
        let mut carry = [0u32; PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            carry[i] = product[i] / base;
            product[i] %= base;
            if i + 1 < PRODUCT_SIZE {
                product[i + 1] += carry[i];
            }
            cols.carry[i] = F::from_canonical_u32(carry[i]);
        }

        cols.product = product.map(F::from_canonical_u32);
        cols.hi = Word(hi_word.map(F::from_canonical_u8));
        cols.a = Word(a_word.map(F::from_canonical_u8));
        cols.b = Word(b_word.map(F::from_canonical_u8));
        cols.c = Word(c_word.map(F::from_canonical_u8));
        cols.is_real = F::ONE;
        cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
        cols.is_mult = F::from_bool(event.opcode == Opcode::MULT);
        cols.is_multu = F::from_bool(event.opcode == Opcode::MULTU);

        // Range check.
        {
            blu.add_u16_range_checks(&carry.map(|x| x as u16));
            blu.add_u8_range_checks(&product.map(|x| x as u8));
        }
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
        let base = AB::F::from_canonical_u32(1 << 8);

        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        let zero: AB::Expr = AB::F::ZERO.into();
        let one: AB::Expr = AB::F::ONE.into();
        let byte_mask = AB::F::from_canonical_u8(BYTE_MASK);

        // Calculate the MSBs.
        let (b_msb, c_msb) = {
            let msb_pairs =
                [(local.b_msb, local.b[WORD_SIZE - 1]), (local.c_msb, local.c[WORD_SIZE - 1])];
            let opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);
            for msb_pair in msb_pairs.iter() {
                let msb = msb_pair.0;
                let byte = msb_pair.1;
                builder.send_byte(opcode, msb, byte, zero.clone(), local.is_real);
            }
            (local.b_msb, local.c_msb)
        };

        // Calculate whether to extend b and c's sign.
        let (b_sign_extend, c_sign_extend) = {
            let is_b_i32 = local.is_mult;
            let is_c_i32 = local.is_mult;

            builder.assert_eq(local.b_sign_extend, is_b_i32 * b_msb);
            builder.assert_eq(local.c_sign_extend, is_c_i32 * c_msb);
            (local.b_sign_extend, local.c_sign_extend)
        };

        // Sign extend local.b and local.c whenever appropriate.
        let (b, c) = {
            let mut b: Vec<AB::Expr> = vec![AB::F::ZERO.into(); PRODUCT_SIZE];
            let mut c: Vec<AB::Expr> = vec![AB::F::ZERO.into(); PRODUCT_SIZE];
            for i in 0..PRODUCT_SIZE {
                if i < WORD_SIZE {
                    b[i] = local.b[i].into();
                    c[i] = local.c[i].into();
                } else {
                    b[i] = b_sign_extend * byte_mask;
                    c[i] = c_sign_extend * byte_mask;
                }
            }
            (b, c)
        };

        // Compute the uncarried product b(x) * c(x) = m(x).
        let mut m: Vec<AB::Expr> = vec![AB::F::ZERO.into(); PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            for j in 0..PRODUCT_SIZE {
                if i + j < PRODUCT_SIZE {
                    m[i + j] = m[i + j].clone() + b[i].clone() * c[j].clone();
                }
            }
        }

        // Propagate carry.
        let product = {
            for i in 0..PRODUCT_SIZE {
                if i == 0 {
                    builder.assert_eq(m[i].clone(), local.carry[i] * base + local.product[i]);
                } else {
                    builder.assert_eq(
                        local.product[i] - local.carry[i - 1] + local.carry[i] * base,
                        m[i].clone(),
                    );
                }
            }
            local.product
        };

        // Compare the product's appropriate bytes with that of the result.
        {
            let has_hi = local.is_mult + local.is_multu;
            for i in 0..WORD_SIZE {
                builder.assert_eq(product[i], local.a[i]);
                builder.when(has_hi.clone()).assert_eq(product[i + WORD_SIZE], local.hi[i]);
            }
        }

        // Check that the boolean values are indeed boolean values.
        {
            let booleans = [
                local.b_msb,
                local.c_msb,
                local.b_sign_extend,
                local.c_sign_extend,
                local.is_mul,
                local.is_mult,
                local.is_multu,
                local.is_real,
                local.hi_record_is_real,
                local.is_real_instruction,
            ];
            for boolean in booleans.iter() {
                builder.assert_bool(*boolean);
            }
        }

        // If signed extended, the MSB better be 1.
        builder.when(local.b_sign_extend).assert_eq(local.b_msb, one.clone());
        builder.when(local.c_sign_extend).assert_eq(local.c_msb, one.clone());

        // Calculate the opcode.
        let opcode = {
            // Exactly one of the op codes must be on.
            builder.when(local.is_real).assert_one(local.is_mul + local.is_mult + local.is_multu);

            let mul: AB::Expr = AB::F::from_canonical_u32(Opcode::MUL as u32).into();
            let mult: AB::Expr = AB::F::from_canonical_u32(Opcode::MULT as u32).into();
            let multu: AB::Expr = AB::F::from_canonical_u32(Opcode::MULTU as u32).into();
            local.is_mul * mul + local.is_mult * mult + local.is_multu * multu
        };

        // Range check.
        {
            // Ensure that the carry is at most 2^16. This ensures that
            // product_before_carry_propagation - carry * base + last_carry never overflows or
            // underflows enough to "wrap" around to create a second solution.
            builder.slice_range_check_u16(&local.carry, local.is_real);

            builder.slice_range_check_u8(&local.product, local.is_real);
        }

        builder.when_not(local.is_real).assert_zero(local.is_real_instruction);
        // MUL is never synthetic: nothing ever produces a synthetic MUL row, so a real MUL row's
        // `is_real_instruction` must track `is_real` exactly.
        builder.when(local.is_mul).assert_eq(local.is_real_instruction, local.is_real);
        let is_real_instruction: AB::Expr = local.is_real_instruction.into();

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk = clk_low_expr::<AB>(&local.state);

        builder.send_program(local.pc, local.instruction, is_real_instruction.clone());

        eval_register_reader(
            builder,
            &local.reader,
            local.state.clk_high,
            clk.clone(),
            &local.instruction,
            // Gated by `is_real_instruction`: `register.rs`'s `assert_word_eq(op_a_value,
            // reader.op_a_val())` fires unconditionally whenever `op_a_0` is unset, which it is
            // by default on synthetic rows (their `instruction` column is never populated).
            local.a.map(|x| is_real_instruction.clone() * Into::<AB::Expr>::into(x)),
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real_instruction.clone(),
        );

        eval_cpu_state(
            builder,
            &local.state,
            public_values.execution_shard,
            clk.clone(),
            is_real_instruction.clone(),
        );

        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk_high_expr::<AB>(&local.state),
            clk.clone(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            is_real_instruction.clone(),
        );

        builder
            .when(is_real_instruction.clone())
            .assert_word_eq(local.reader.op_b_val(), local.b.map(Into::into));
        builder
            .when(is_real_instruction.clone())
            .assert_word_eq(local.reader.op_c_val(), local.c.map(Into::into));

        // Bind `is_real_instruction` to the row's actual fetched opcode, so a real-instruction
        // row can't claim the wrong multiply variant while still passing the program lookup.
        // These gates are plain assertions (not interaction values/multiplicities), so the
        // product `is_real_instruction * is_X` (degree 2) is fine here.
        builder
            .when(is_real_instruction.clone() * local.is_mul)
            .assert_eq(local.instruction.opcode, AB::F::from_canonical_u32(Opcode::MUL as u32));
        builder
            .when(is_real_instruction.clone() * local.is_mult)
            .assert_eq(local.instruction.opcode, AB::F::from_canonical_u32(Opcode::MULT as u32));
        builder
            .when(is_real_instruction.clone() * local.is_multu)
            .assert_eq(local.instruction.opcode, AB::F::from_canonical_u32(Opcode::MULTU as u32));

        // ---- Synthetic dependency path: matches whichever chip generated this internal check via
        // `send_alu`/`send_alu_with_hi` (always at the `UNUSED_PC` sentinel, shard/clk zero). ----
        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.next_pc,
            local.next_pc + AB::Expr::from_canonical_u32(4),
            AB::Expr::zero(),
            opcode,
            local.a,
            local.b,
            local.c,
            local.hi,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.hi_record_is_real,
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_real - is_real_instruction.clone(),
        );

        // Write the HI register, the register can only be Register::HI（33）.
        builder.eval_memory_access(
            local.state.clk_high,
            clk + AB::F::from_canonical_u32(MemoryAccessPosition::HI as u32),
            AB::F::from_canonical_u32(33),
            &local.op_hi_access,
            local.hi_record_is_real,
        );

        // Check hi_record_is_real.
        // hi_record_is_real can only be set for MULT and MULTU instruction when is_real = 1.
        builder.when_not(local.is_real).assert_zero(local.hi_record_is_real);
        builder.when(local.hi_record_is_real).assert_one(local.is_mult + local.is_multu);
        // A real MULT/MULTU retirement must always write HI; only synthetic dependency rows
        // (or padding) may skip it.
        builder
            .when(is_real_instruction * (local.is_mult + local.is_multu))
            .assert_one(local.hi_record_is_real);
        builder.when(local.hi_record_is_real).assert_word_eq(local.hi, *local.op_hi_access.value());
        builder.when(local.is_mul).assert_word_zero(local.hi);
    }
}

#[cfg(test)]
mod tests {
    // use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::CompAluEvent, ExecutionRecord, Opcode, UNUSED_PC};
    use zkm_hypercube::air::MachineAir;
    // use zkm_stark::{
    //     air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig,
    // };

    use super::MulChip;

    #[test]
    fn generate_trace_mul() {
        let mut shard = ExecutionRecord::default();

        // Fill mul_events with 10 MUL events.
        let mut mul_events: Vec<CompAluEvent> = Vec::new();
        for _ in 0..10 {
            mul_events.push(CompAluEvent::new(
                UNUSED_PC,
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

    #[test]
    #[ignore = "no zkm-hypercube single-chip prove/verify utility yet (old FRI-backed uni_stark_prove/verify removed)"]
    fn prove_koalabear() {
        // let config = KoalaBearPoseidon2::new();
        // let mut challenger = config.challenger();
        //
        // let mut shard = ExecutionRecord::default();
        // let mut mul_events: Vec<CompAluEvent> = Vec::new();
        //
        // let mul_instructions: Vec<(Opcode, u32, u32, u32)> = vec![
        //     (Opcode::MUL, 0x00001200, 0x00007e00, 0xb6db6db7),
        //     (Opcode::MUL, 0x00001240, 0x00007fc0, 0xb6db6db7),
        //     (Opcode::MUL, 0x00000000, 0x00000000, 0x00000000),
        //     (Opcode::MUL, 0x00000001, 0x00000001, 0x00000001),
        //     (Opcode::MUL, 0x00000015, 0x00000003, 0x00000007),
        //     (Opcode::MUL, 0x00000000, 0x00000000, 0xffff8000),
        //     (Opcode::MUL, 0x00000000, 0x80000000, 0x00000000),
        //     (Opcode::MUL, 0x00000000, 0x80000000, 0xffff8000),
        //     (Opcode::MUL, 0x0000ff7f, 0xaaaaaaab, 0x0002fe7d),
        //     (Opcode::MUL, 0x0000ff7f, 0x0002fe7d, 0xaaaaaaab),
        //     (Opcode::MUL, 0x00000000, 0xff000000, 0xff000000),
        //     (Opcode::MUL, 0x00000001, 0xffffffff, 0xffffffff),
        //     (Opcode::MUL, 0xffffffff, 0xffffffff, 0x00000001),
        //     (Opcode::MUL, 0xffffffff, 0x00000001, 0xffffffff),
        // ];
        // for t in mul_instructions.iter() {
        //     mul_events.push(CompAluEvent::new(0, t.0, t.1, t.2, t.3));
        // }
        //
        // // Append more events until we have 1000 tests.
        // for _ in 0..(1000 - mul_instructions.len()) {
        //     mul_events.push(CompAluEvent::new(0, Opcode::MUL, 1, 1, 1));
        // }
        //
        // shard.mul_events = mul_events;
        // let chip = MulChip::default();
        // let trace: RowMajorMatrix<KoalaBear> =
        //     chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        // let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        //
        // let mut challenger = config.challenger();
        // verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
