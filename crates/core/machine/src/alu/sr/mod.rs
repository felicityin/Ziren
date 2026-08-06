//! Logical And Arithmetic Right Shift Verification.
//!
//! Implements verification for a = b >> c, decomposing the shift into bit and byte components:
//!
//! 1. num_bits_to_shift = c % 8: Bit-level shift, achieved by using ShrCarry.
//! 2. num_bytes_to_shift = c // 8: Byte-level shift, shifting entire bytes or words in b.
//!
//! The right shift is verified by reformulating it as (b >> c) = (b >> (num_bytes_to_shift * 8)) >>
//! num_bits_to_shift.
//!
//! The correct leading bits of logical and arithmetic right shifts are verified by sign extending b
//! to 64 bits.
//!
//! c = take the least significant 5 bits of c
//! num_bytes_to_shift = c // 8
//! num_bits_to_shift = c % 8
//!
//! # Sign extend b to 64 bits if SRA.
//! if opcode == SRA:
//!    b = sign_extend_32_bits_to_64_bits(b)
//! else:
//!    b = zero_extend_32_bits_to_64_bits(b)
//!
//!
//! # Byte shift. Leave the num_bytes_to_shift most significant bytes of b 0 for simplicity as it
//! # doesn't affect the correctness of the result.
//! result = [0; LONG_WORD_SIZE]
//! for i in range(LONG_WORD_SIZE - num_bytes_to_shift):
//!     result[i] = b[i + num_bytes_to_shift]
//!
//! # Bit shift.
//! carry_multiplier = 1 << (8 - num_bits_to_shift)
//! last_carry = 0
//! for i in reversed(range(LONG_WORD_SIZE)):
//!     # Shifts a byte to the right and returns both the shifted byte and the bits that carried.
//!     (shifted_byte[i], carry) = shr_carry(result[i], num_bits_to_shift)
//!     result[i] = shifted_byte[i] + last_carry * carry_multiplier
//!     last_carry = carry
//!
//! # The 4 least significant bytes must match a. The 4 most significant bytes of result may be
//! # inaccurate.
//! assert a = result[0..WORD_SIZE]

pub(crate) mod utils;

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program, UNUSED_PC,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{
    air::MachineAir,
    word::Word,
};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    adapter::{
        clk_low_expr, eval_alu_type_reader, eval_cpu_state, eval_state_chain, AluTypeReader,
        CpuState, InstructionCols,
    },
    air::ZKMCoreAirBuilder,
    alu::sr::utils::{nb_bits_to_shift, nb_bytes_to_shift},
    bytes::utils::shr_carry,
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `ShiftRightChip`.
pub const NUM_SHIFT_RIGHT_COLS: usize = size_of::<ShiftRightCols<u8>>();

/// The number of bytes necessary to represent a 64-bit integer.
const LONG_WORD_SIZE: usize = 2 * WORD_SIZE;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// A chip that implements bitwise operations for the opcodes SRL, SRA, and ROR.
///
/// Every row is a real, retired instruction: `CloClzChip`/`ExtChip`/`InsChip` (the only other
/// chips with an internal SRL/ROR dependency) each verify their own copy locally via an embedded
/// `ShiftRightOperation`/`FixedShiftRightOperation` instead of a cross-chip lookup into this
/// chip. A real SRL/SRA/ROR whose destination is register 0 is routed to `AluX0Chip` instead (see
/// its doc comment), since its result is unobservable and discarding it soundly requires a
/// different (cheaper) register-write scheme than a real result does -- see `AluTypeReader`'s doc
/// comment. Every real row reaching *this* chip therefore has a genuine, non-zero destination
/// register, which is what lets it use the narrow `AluTypeReader` (see its doc comment) instead
/// of the generic `InstructionCols`+`RegisterReader` pair.
#[derive(Default)]
pub struct ShiftRightChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct ShiftRightCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: AluTypeReader<T>,

    /// A boolean array whose `i`th element indicates whether `num_bits_to_shift = i`.
    pub shift_by_n_bits: [T; BYTE_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],

    /// The result of "byte-shifting" the input operand `b` by `num_bytes_to_shift`.
    pub byte_shift_result: [T; LONG_WORD_SIZE],

    /// The result of "bit-shifting" the byte-shifted input by `num_bits_to_shift`.
    pub bit_shift_result: [T; LONG_WORD_SIZE],

    /// The carry output of `shrcarry` on each byte of `byte_shift_result`.
    pub shr_carry_output_carry: [T; LONG_WORD_SIZE],

    /// The shift byte output of `shrcarry` on each byte of `byte_shift_result`.
    pub shr_carry_output_shifted_byte: [T; LONG_WORD_SIZE],

    /// The most significant bit of `b`.
    pub b_msb: T,

    /// The least significant byte of `c`. Used to verify `shift_by_n_bits` and `shift_by_n_bytes`.
    pub c_least_sig_byte: [T; BYTE_SIZE],

    /// If the opcode is SRL.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_srl: T,

    /// If the opcode is ROR.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_ror: T,

    /// If the opcode is SRA.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_sra: T,
}

impl<F: PrimeField32> MachineAir<F> for ShiftRightChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShiftRight".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        ShiftRightCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.shift_right_events.len(),
            None,
            <ShiftRightChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let nb_rows = input.shift_right_events.len();
        let padded_nb_rows = <ShiftRightChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SHIFT_RIGHT_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_SHIFT_RIGHT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SHIFT_RIGHT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut ShiftRightCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.shift_right_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events, &input.program);
                    } else {
                        // Padding row: `shift_by_n_bits`/`shift_by_n_bytes` need a valid one-hot
                        // selection even here, since their "exactly one is set" constraints are
                        // unconditional (not gated by `is_real`). Everything else (including
                        // `adapter`'s register-access multiplicities) is already correctly zeroed
                        // by `is_real` (the sum of the opcode selectors, all zero here) -- unlike
                        // the generic `RegisterReader`, `AluTypeReader` needs no separate "force
                        // immediate flags" workaround for this.
                        cols.shift_by_n_bits[0] = F::ONE;
                        cols.shift_by_n_bytes[0] = F::ONE;
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_SHIFT_RIGHT_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.shift_right_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .shift_right_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_SHIFT_RIGHT_COLS];
                    let cols: &mut ShiftRightCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.shift_right_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl ShiftRightChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut ShiftRightCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        // Initialize cols with basic operands and flags derived from the current event.
        {
            cols.pc = F::from_canonical_u32(event.pc);
            cols.next_pc = F::from_canonical_u32(event.next_pc);

            cols.b_msb = F::from_canonical_u32((event.b >> 31) & 1);

            cols.is_srl = F::from_bool(event.opcode == Opcode::SRL);
            cols.is_sra = F::from_bool(event.opcode == Opcode::SRA);
            cols.is_ror = F::from_bool(event.opcode == Opcode::ROR);

            for i in 0..BYTE_SIZE {
                cols.c_least_sig_byte[i] = F::from_canonical_u32((event.c >> i) & 1);
            }

            // Insert the MSB lookup event.
            let most_significant_byte = event.b.to_le_bytes()[WORD_SIZE - 1];
            blu.add_byte_lookup_events(vec![ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a1: ((most_significant_byte >> 7) & 1) as u16,
                a2: 0,
                b: most_significant_byte,
                c: 0,
            }]);
        }

        debug_assert!(event.pc != UNUSED_PC, "every ShiftRight row is now a real instruction");
        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);
        cols.adapter.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
            event.c_record,
            instruction.imm_c,
        );

        let num_bytes_to_shift = nb_bytes_to_shift(event.c);
        let num_bits_to_shift = nb_bits_to_shift(event.c);

        // Byte shifting.
        let mut byte_shift_result = [0u8; LONG_WORD_SIZE];
        {
            for i in 0..WORD_SIZE {
                cols.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
            }
            let sign_extended_b = {
                if event.opcode == Opcode::SRA {
                    // Sign extension is necessary only for arithmetic right shift.
                    ((event.b as i32) as i64).to_le_bytes()
                } else if event.opcode == Opcode::ROR {
                    (((event.b as u64) << 32) | (event.b as u64)).to_le_bytes()
                } else {
                    (event.b as u64).to_le_bytes()
                }
            };

            for i in 0..LONG_WORD_SIZE {
                if i + num_bytes_to_shift < LONG_WORD_SIZE {
                    byte_shift_result[i] = sign_extended_b[i + num_bytes_to_shift];
                }
            }
            cols.byte_shift_result = byte_shift_result.map(F::from_canonical_u8);
        }

        // Bit shifting.
        {
            for i in 0..BYTE_SIZE {
                cols.shift_by_n_bits[i] = F::from_bool(num_bits_to_shift == i);
            }
            let carry_multiplier = 1 << (8 - num_bits_to_shift);
            let mut last_carry = 0u32;
            let mut bit_shift_result = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_carry = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_shifted_byte = [0u8; LONG_WORD_SIZE];
            for i in (0..LONG_WORD_SIZE).rev() {
                let (shift, carry) = shr_carry(byte_shift_result[i], num_bits_to_shift as u8);

                let byte_event = ByteLookupEvent {
                    opcode: ByteOpcode::ShrCarry,
                    a1: shift as u16,
                    a2: carry,
                    b: byte_shift_result[i],
                    c: num_bits_to_shift as u8,
                };
                blu.add_byte_lookup_event(byte_event);

                shr_carry_output_carry[i] = carry;
                shr_carry_output_shifted_byte[i] = shift;
                bit_shift_result[i] = ((shift as u32 + last_carry * carry_multiplier) & 0xff) as u8;
                last_carry = carry as u32;
            }
            cols.bit_shift_result = bit_shift_result.map(F::from_canonical_u8);
            cols.shr_carry_output_carry = shr_carry_output_carry.map(F::from_canonical_u8);
            cols.shr_carry_output_shifted_byte =
                shr_carry_output_shifted_byte.map(F::from_canonical_u8);
            for i in 0..WORD_SIZE {
                debug_assert_eq!(
                    cols.bit_shift_result[i],
                    F::from_canonical_u8(event.a.to_le_bytes()[i])
                );
            }
            // Range checks.
            blu.add_u8_range_checks(&byte_shift_result);
            blu.add_u8_range_checks(&bit_shift_result);
            blu.add_u8_range_checks(&shr_carry_output_carry);
            blu.add_u8_range_checks(&shr_carry_output_shifted_byte);
        }
    }
}

impl<F> BaseAir<F> for ShiftRightChip {
    fn width(&self) -> usize {
        NUM_SHIFT_RIGHT_COLS
    }
}

impl<AB> Air<AB> for ShiftRightChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShiftRightCols<AB::Var> = (*local).borrow();
        let zero: AB::Expr = AB::F::ZERO.into();
        let one: AB::Expr = AB::F::ONE.into();

        builder.assert_bool(local.is_srl);
        builder.assert_bool(local.is_sra);
        builder.assert_bool(local.is_ror);
        let is_real = local.is_srl + local.is_sra + local.is_ror;
        builder.assert_bool(is_real.clone());

        let op_b_val = local.adapter.op_b_val();
        let op_c_val = local.adapter.op_c_val();

        // Check that the MSB of most_significant_byte matches local.b_msb using lookup.
        {
            let byte = op_b_val[WORD_SIZE - 1];
            let opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);
            let msb = local.b_msb;
            builder.send_byte(opcode, msb, byte, zero.clone(), is_real.clone());
        }

        // Calculate the number of bits and bytes to shift by from c.
        {
            // The sum of c_least_sig_byte[i] * 2^i must match c[0].
            let mut c_byte_sum = AB::Expr::zero();
            for i in 0..BYTE_SIZE {
                let val: AB::Expr = AB::F::from_canonical_u32(1 << i).into();
                c_byte_sum = c_byte_sum.clone() + val * local.c_least_sig_byte[i];
            }
            builder.assert_eq(c_byte_sum, op_c_val[0]);

            // Number of bits to shift.

            // The 3-bit number represented by the 3 least significant bits of c equals the number
            // of bits to shift.
            let mut num_bits_to_shift = AB::Expr::zero();
            for i in 0..3 {
                num_bits_to_shift = num_bits_to_shift.clone()
                    + local.c_least_sig_byte[i] * AB::F::from_canonical_u32(1 << i);
            }
            for i in 0..BYTE_SIZE {
                builder
                    .when(local.shift_by_n_bits[i])
                    .assert_eq(num_bits_to_shift.clone(), AB::F::from_canonical_usize(i));
            }

            // Exactly one of the shift_by_n_bits must be 1.
            builder.assert_eq(
                local.shift_by_n_bits.iter().fold(zero.clone(), |acc, &x| acc + x),
                one.clone(),
            );

            // The 2-bit number represented by the 3rd and 4th least significant bits of c is the
            // number of bytes to shift.
            let num_bytes_to_shift = local.c_least_sig_byte[3]
                + local.c_least_sig_byte[4] * AB::F::from_canonical_u32(2);

            // If shift_by_n_bytes[i] = 1, then i = num_bytes_to_shift.
            for i in 0..WORD_SIZE {
                builder
                    .when(local.shift_by_n_bytes[i])
                    .assert_eq(num_bytes_to_shift.clone(), AB::F::from_canonical_usize(i));
            }

            // Exactly one of the shift_by_n_bytes must be 1.
            builder.assert_eq(
                local.shift_by_n_bytes.iter().fold(zero.clone(), |acc, &x| acc + x),
                one.clone(),
            );
        }

        // Byte shift the sign-extended b.
        {
            // The leading bytes of b should be 0xff if b's MSB is 1 & opcode = SRA, 0 otherwise.
            let mut sign_extended_b: Vec<AB::Expr> = vec![];
            for i in 0..WORD_SIZE {
                sign_extended_b.push(op_b_val[i].into());
            }
            for i in 0..WORD_SIZE {
                let leading_byte = local.is_sra * local.b_msb * AB::Expr::from_canonical_u8(0xff)
                    + local.is_ror * op_b_val[i].into();
                sign_extended_b.push(leading_byte.clone());
            }

            // Shift the bytes of sign_extended_b by num_bytes_to_shift.
            for num_bytes_to_shift in 0..WORD_SIZE {
                for i in 0..(LONG_WORD_SIZE - num_bytes_to_shift) {
                    builder.when(local.shift_by_n_bytes[num_bytes_to_shift]).assert_eq(
                        local.byte_shift_result[i],
                        sign_extended_b[i + num_bytes_to_shift].clone(),
                    );
                }
            }
        }

        // Bit shift the byte_shift_result using ShrCarry, and compare the result to a.
        {
            // The carry multiplier is 2^(8 - num_bits_to_shift).
            let mut carry_multiplier = AB::Expr::from_canonical_u8(0);
            for i in 0..BYTE_SIZE {
                carry_multiplier = carry_multiplier.clone()
                    + AB::Expr::from_canonical_u32(1u32 << (8 - i)) * local.shift_by_n_bits[i];
            }

            // The 3-bit number represented by the 3 least significant bits of c equals the number
            // of bits to shift.
            let mut num_bits_to_shift = AB::Expr::zero();
            for i in 0..3 {
                num_bits_to_shift = num_bits_to_shift.clone()
                    + local.c_least_sig_byte[i] * AB::F::from_canonical_u32(1 << i);
            }

            // Calculate ShrCarry.
            for i in (0..LONG_WORD_SIZE).rev() {
                builder.send_byte_pair(
                    AB::F::from_canonical_u32(ByteOpcode::ShrCarry as u32),
                    local.shr_carry_output_shifted_byte[i],
                    local.shr_carry_output_carry[i],
                    local.byte_shift_result[i],
                    num_bits_to_shift.clone(),
                    is_real.clone(),
                );
            }

            // Use the results of ShrCarry to calculate the bit shift result.
            for i in (0..LONG_WORD_SIZE).rev() {
                let mut v: AB::Expr = local.shr_carry_output_shifted_byte[i].into();
                if i + 1 < LONG_WORD_SIZE {
                    v = v.clone() + local.shr_carry_output_carry[i + 1] * carry_multiplier.clone();
                }
                builder.assert_eq(v, local.bit_shift_result[i]);
            }
        }

        // Check that the flags are indeed boolean.
        {
            builder.assert_bool(local.b_msb);
            for shift_by_n_byte in local.shift_by_n_bytes.iter() {
                builder.assert_bool(*shift_by_n_byte);
            }
            for shift_by_n_bit in local.shift_by_n_bits.iter() {
                builder.assert_bool(*shift_by_n_bit);
            }
            for bit in local.c_least_sig_byte.iter() {
                builder.assert_bool(*bit);
            }
        }

        // Range check bytes.
        {
            let long_words = [
                local.byte_shift_result,
                local.bit_shift_result,
                local.shr_carry_output_carry,
                local.shr_carry_output_shifted_byte,
            ];

            for long_word in long_words.iter() {
                builder.slice_range_check_u8(long_word, is_real.clone());
            }
        }

        // Use bit_shift_result[0..4] directly as the output operand `a`, eliminating the
        // redundant `a` column since a[i] == bit_shift_result[i] is always true.
        let a_word = Word([
            local.bit_shift_result[0],
            local.bit_shift_result[1],
            local.bit_shift_result[2],
            local.bit_shift_result[3],
        ]);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode` is a degree-1
        // linear combination of the one-hot selectors above (never a variable a malicious prover
        // could substitute -- the program lookup against `ProgramChip`'s preprocessed ROM is what
        // makes `op_a`/`op_b`/`op_c` trustworthy, so there's no separate opcode-binding check
        // needed), `op_a_0`/`imm_b` are compile-time constants (this chip only ever sees a
        // non-zero destination and a register `op_b` -- see `AluTypeReader`'s doc comment), and
        // `op_b` is zero-extended from the adapter's register-index column.
        let cpu_opcode = local.is_srl * Opcode::SRL.as_field::<AB::F>()
            + local.is_sra * Opcode::SRA.as_field::<AB::F>()
            + local.is_ror * Opcode::ROR.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: cpu_opcode,
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: local.adapter.imm_c.into(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        eval_alu_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            a_word.map(Into::into),
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
    // use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;
    // use zkm_stark::{
    //     air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig,
    // };

    use super::ShiftRightChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SRL,
                op_a: 5,
                op_b: 12,
                op_c: 1,
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
        shard.shift_right_events = vec![AluEvent::new(0, Opcode::SRL, 6, 12, 1)];
        let chip = ShiftRightChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    #[ignore = "no zkm-hypercube single-chip prove/verify utility yet (old FRI-backed uni_stark_prove/verify removed)"]
    fn prove_koalabear() {
        // let config = KoalaBearPoseidon2::new();
        // let mut challenger = config.challenger();
        //
        // let shifts = vec![
        //     (Opcode::SRL, 0xffff8000, 0xffff8000, 0),
        //     (Opcode::SRL, 0x7fffc000, 0xffff8000, 1),
        //     (Opcode::SRL, 0x01ffff00, 0xffff8000, 7),
        //     (Opcode::SRL, 0x0003fffe, 0xffff8000, 14),
        //     (Opcode::SRL, 0x0001ffff, 0xffff8001, 15),
        //     (Opcode::SRL, 0xffffffff, 0xffffffff, 0),
        //     (Opcode::SRL, 0x7fffffff, 0xffffffff, 1),
        //     (Opcode::SRL, 0x01ffffff, 0xffffffff, 7),
        //     (Opcode::SRL, 0x0003ffff, 0xffffffff, 14),
        //     (Opcode::SRL, 0x00000001, 0xffffffff, 31),
        //     (Opcode::SRL, 0x21212121, 0x21212121, 0),
        //     (Opcode::SRL, 0x10909090, 0x21212121, 1),
        //     (Opcode::SRL, 0x00424242, 0x21212121, 7),
        //     (Opcode::SRL, 0x00008484, 0x21212121, 14),
        //     (Opcode::SRL, 0x00000000, 0x21212121, 31),
        //     (Opcode::SRL, 0x21212121, 0x21212121, 0xffffffe0),
        //     (Opcode::SRL, 0x10909090, 0x21212121, 0xffffffe1),
        //     (Opcode::SRL, 0x00424242, 0x21212121, 0xffffffe7),
        //     (Opcode::SRL, 0x00008484, 0x21212121, 0xffffffee),
        //     (Opcode::SRL, 0x00000000, 0x21212121, 0xffffffff),
        //     (Opcode::SRA, 0x00000000, 0x00000000, 0),
        //     (Opcode::SRA, 0xc0000000, 0x80000000, 1),
        //     (Opcode::SRA, 0xff000000, 0x80000000, 7),
        //     (Opcode::SRA, 0xfffe0000, 0x80000000, 14),
        //     (Opcode::SRA, 0xffffffff, 0x80000001, 31),
        //     (Opcode::SRA, 0x7fffffff, 0x7fffffff, 0),
        //     (Opcode::SRA, 0x3fffffff, 0x7fffffff, 1),
        //     (Opcode::SRA, 0x00ffffff, 0x7fffffff, 7),
        //     (Opcode::SRA, 0x0001ffff, 0x7fffffff, 14),
        //     (Opcode::SRA, 0x00000000, 0x7fffffff, 31),
        //     (Opcode::SRA, 0x81818181, 0x81818181, 0),
        //     (Opcode::SRA, 0xc0c0c0c0, 0x81818181, 1),
        //     (Opcode::SRA, 0xff030303, 0x81818181, 7),
        //     (Opcode::SRA, 0xfffe0606, 0x81818181, 14),
        //     (Opcode::SRA, 0xffffffff, 0x81818181, 31),
        // ];
        // let mut shift_events: Vec<AluEvent> = Vec::new();
        // for t in shifts.iter() {
        //     shift_events.push(AluEvent::new(0, t.0, t.1, t.2, t.3));
        // }
        // let mut shard = ExecutionRecord::default();
        // shard.shift_right_events = shift_events;
        // let chip = ShiftRightChip::default();
        // let trace: RowMajorMatrix<KoalaBear> =
        //     chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        // let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        //
        // let mut challenger = config.challenger();
        // verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
