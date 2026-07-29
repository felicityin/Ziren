use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::word::Word;
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    air::ZKMCoreAirBuilder,
    alu::sr::utils::{nb_bits_to_shift, nb_bytes_to_shift},
    bytes::utils::shr_carry,
};

/// The number of bytes necessary to represent a 64-bit integer.
const LONG_WORD_SIZE: usize = 2 * WORD_SIZE;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// Computes and validates `b >> (c & 0x1f)`, either zero-extended (`SRL`) or rotated (`ROR`),
/// shared by every chip that needs a right shift/rotate by a witnessed (not
/// compile-time-constant) amount: `ShiftRightChip` (real SRL/SRA/ROR instructions), and -- as an
/// embedded, no-cross-chip-lookup copy -- `CloClzChip`'s/`ExtChip`'s/`InsChip`'s intermediate
/// shift steps.
///
/// Unlike `ShiftRightChip`, this operation has no `SRA` variant and no `b_msb`/opcode-selector
/// columns: every embedded use site is statically known to be either `SRL` or `ROR` (chosen via
/// the `is_rotate` parameter to `populate`/`eval`, a plain Rust bool, not a witness column), so
/// there's nothing to multiplex at proving time.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftRightOperation<T> {
    /// The least significant byte of `c`. Used to verify `shift_by_n_bits` and
    /// `shift_by_n_bytes`.
    pub c_least_sig_byte: [T; BYTE_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bits_to_shift = i`.
    pub shift_by_n_bits: [T; BYTE_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],

    /// The result of "byte-shifting" the input operand `b` by `num_bytes_to_shift`.
    pub byte_shift_result: [T; LONG_WORD_SIZE],

    /// The result of "bit-shifting" the byte-shifted input by `num_bits_to_shift`. The low word
    /// (`bit_shift_result[0..4]`) is the result -- there's no separate output column.
    pub bit_shift_result: [T; LONG_WORD_SIZE],

    /// The carry output of `shrcarry` on each byte of `byte_shift_result`.
    pub shr_carry_output_carry: [T; LONG_WORD_SIZE],

    /// The shift byte output of `shrcarry` on each byte of `byte_shift_result`.
    pub shr_carry_output_shifted_byte: [T; LONG_WORD_SIZE],
}

impl<T: Copy> ShiftRightOperation<T> {
    /// The result of `b >> c`, taken directly from the low word of `bit_shift_result`.
    pub fn value(&self) -> Word<T> {
        Word([
            self.bit_shift_result[0],
            self.bit_shift_result[1],
            self.bit_shift_result[2],
            self.bit_shift_result[3],
        ])
    }
}

impl<F: PrimeField32> ShiftRightOperation<F> {
    /// Populates the columns for `b >> c` (zero-extended if `!is_rotate`, rotated if `is_rotate`),
    /// returning the result.
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        b_u32: u32,
        c_u32: u32,
        is_rotate: bool,
    ) -> u32 {
        for i in 0..BYTE_SIZE {
            self.c_least_sig_byte[i] = F::from_canonical_u32((c_u32 >> i) & 1);
        }

        let num_bytes_to_shift = nb_bytes_to_shift(c_u32);
        let num_bits_to_shift = nb_bits_to_shift(c_u32);

        let mut byte_shift_result = [0u8; LONG_WORD_SIZE];
        {
            for i in 0..WORD_SIZE {
                self.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
            }
            let extended_b = if is_rotate {
                (((b_u32 as u64) << 32) | (b_u32 as u64)).to_le_bytes()
            } else {
                (b_u32 as u64).to_le_bytes()
            };

            for i in 0..LONG_WORD_SIZE {
                if i + num_bytes_to_shift < LONG_WORD_SIZE {
                    byte_shift_result[i] = extended_b[i + num_bytes_to_shift];
                }
            }
            self.byte_shift_result = byte_shift_result.map(F::from_canonical_u8);
        }

        {
            for i in 0..BYTE_SIZE {
                self.shift_by_n_bits[i] = F::from_bool(num_bits_to_shift == i);
            }
            let carry_multiplier = 1 << (8 - num_bits_to_shift);
            let mut last_carry = 0u32;
            let mut bit_shift_result = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_carry = [0u8; LONG_WORD_SIZE];
            let mut shr_carry_output_shifted_byte = [0u8; LONG_WORD_SIZE];
            for i in (0..LONG_WORD_SIZE).rev() {
                let (shift, carry) = shr_carry(byte_shift_result[i], num_bits_to_shift as u8);

                blu.add_byte_lookup_event(ByteLookupEvent {
                    opcode: ByteOpcode::ShrCarry,
                    a1: shift as u16,
                    a2: carry,
                    b: byte_shift_result[i],
                    c: num_bits_to_shift as u8,
                });

                shr_carry_output_carry[i] = carry;
                shr_carry_output_shifted_byte[i] = shift;
                bit_shift_result[i] = ((shift as u32 + last_carry * carry_multiplier) & 0xff) as u8;
                last_carry = carry as u32;
            }
            self.bit_shift_result = bit_shift_result.map(F::from_canonical_u8);
            self.shr_carry_output_carry = shr_carry_output_carry.map(F::from_canonical_u8);
            self.shr_carry_output_shifted_byte =
                shr_carry_output_shifted_byte.map(F::from_canonical_u8);

            blu.add_u8_range_checks(&byte_shift_result);
            blu.add_u8_range_checks(&bit_shift_result);
            blu.add_u8_range_checks(&shr_carry_output_carry);
            blu.add_u8_range_checks(&shr_carry_output_shifted_byte);
        }

        u32::from_le_bytes(core::array::from_fn(|i| self.bit_shift_result[i].as_canonical_u32() as u8))
    }
}

impl<F: Field> ShiftRightOperation<F> {
    /// Evaluates `b >> c` (zero-extended if `!is_rotate`, rotated if `is_rotate`), returning the
    /// result word (`cols.value()`).
    pub fn eval<AB: ZKMCoreAirBuilder>(
        builder: &mut AB,
        b: Word<AB::Var>,
        c: Word<AB::Var>,
        cols: ShiftRightOperation<AB::Var>,
        is_rotate: bool,
        is_real: AB::Expr,
    ) -> Word<AB::Var> {
        let zero: AB::Expr = AB::F::ZERO.into();
        let one: AB::Expr = AB::F::ONE.into();

        {
            let mut c_byte_sum = AB::Expr::zero();
            for i in 0..BYTE_SIZE {
                let val: AB::Expr = AB::F::from_canonical_u32(1 << i).into();
                c_byte_sum = c_byte_sum.clone() + val * cols.c_least_sig_byte[i];
            }
            builder.when(is_real.clone()).assert_eq(c_byte_sum, c[0]);

            let mut num_bits_to_shift = AB::Expr::zero();
            for i in 0..3 {
                num_bits_to_shift = num_bits_to_shift.clone()
                    + cols.c_least_sig_byte[i] * AB::F::from_canonical_u32(1 << i);
            }
            for i in 0..BYTE_SIZE {
                builder
                    .when(is_real.clone())
                    .when(cols.shift_by_n_bits[i])
                    .assert_eq(num_bits_to_shift.clone(), AB::F::from_canonical_usize(i));
            }
            builder.when(is_real.clone()).assert_eq(
                cols.shift_by_n_bits.iter().fold(zero.clone(), |acc, &x| acc + x),
                one.clone(),
            );

            let num_bytes_to_shift =
                cols.c_least_sig_byte[3] + cols.c_least_sig_byte[4] * AB::F::from_canonical_u32(2);
            for i in 0..WORD_SIZE {
                builder
                    .when(is_real.clone())
                    .when(cols.shift_by_n_bytes[i])
                    .assert_eq(num_bytes_to_shift.clone(), AB::F::from_canonical_usize(i));
            }
            builder.when(is_real.clone()).assert_eq(
                cols.shift_by_n_bytes.iter().fold(zero.clone(), |acc, &x| acc + x),
                one.clone(),
            );
        }

        {
            let mut extended_b: Vec<AB::Expr> = vec![];
            for i in 0..WORD_SIZE {
                extended_b.push(b[i].into());
            }
            for i in 0..WORD_SIZE {
                // `SRL` zero-extends (leading byte 0); `ROR` wraps `b` into the upper 32 bits.
                extended_b.push(if is_rotate { b[i].into() } else { zero.clone() });
            }

            for num_bytes_to_shift in 0..WORD_SIZE {
                let mut shifting = builder.when(
                    is_real.clone() * Into::<AB::Expr>::into(cols.shift_by_n_bytes[num_bytes_to_shift]),
                );
                for i in 0..(LONG_WORD_SIZE - num_bytes_to_shift) {
                    shifting.assert_eq(
                        cols.byte_shift_result[i],
                        extended_b[i + num_bytes_to_shift].clone(),
                    );
                }
            }
        }

        {
            let mut carry_multiplier = AB::Expr::from_canonical_u8(0);
            for i in 0..BYTE_SIZE {
                carry_multiplier = carry_multiplier.clone()
                    + AB::Expr::from_canonical_u32(1u32 << (8 - i)) * cols.shift_by_n_bits[i];
            }

            let mut num_bits_to_shift = AB::Expr::zero();
            for i in 0..3 {
                num_bits_to_shift = num_bits_to_shift.clone()
                    + cols.c_least_sig_byte[i] * AB::F::from_canonical_u32(1 << i);
            }

            for i in (0..LONG_WORD_SIZE).rev() {
                builder.send_byte_pair(
                    AB::F::from_canonical_u32(ByteOpcode::ShrCarry as u32),
                    cols.shr_carry_output_shifted_byte[i],
                    cols.shr_carry_output_carry[i],
                    cols.byte_shift_result[i],
                    num_bits_to_shift.clone(),
                    is_real.clone(),
                );
            }

            for i in (0..LONG_WORD_SIZE).rev() {
                let mut v: AB::Expr = cols.shr_carry_output_shifted_byte[i].into();
                if i + 1 < LONG_WORD_SIZE {
                    v = v.clone() + cols.shr_carry_output_carry[i + 1] * carry_multiplier.clone();
                }
                builder.when(is_real.clone()).assert_eq(v, cols.bit_shift_result[i]);
            }
        }

        for bit in cols.c_least_sig_byte.iter() {
            builder.when(is_real.clone()).assert_bool(*bit);
        }
        for shift in cols.shift_by_n_bits.iter() {
            builder.when(is_real.clone()).assert_bool(*shift);
        }
        for shift in cols.shift_by_n_bytes.iter() {
            builder.when(is_real.clone()).assert_bool(*shift);
        }

        let long_words = [
            cols.byte_shift_result,
            cols.bit_shift_result,
            cols.shr_carry_output_carry,
            cols.shr_carry_output_shifted_byte,
        ];
        for long_word in long_words.iter() {
            builder.slice_range_check_u8(long_word, is_real.clone());
        }

        cols.value()
    }
}
