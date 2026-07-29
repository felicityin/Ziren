use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::events::ByteRecord;
use zkm_derive::AlignedBorrow;
use zkm_hypercube::word::Word;
use zkm_primitives::consts::WORD_SIZE;

use crate::air::ZKMCoreAirBuilder;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// Computes and validates `b << (c & 0x1f)`, shared by every chip that needs a left shift by a
/// witnessed (not compile-time-constant) amount: `ShiftLeft` (real SLL/SLLI instructions), and --
/// as an embedded, no-cross-chip-lookup copy -- `ExtChip`'s/`InsChip`'s intermediate shift steps.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShiftLeftOperation<T> {
    /// The result of `b << c`.
    pub value: Word<T>,

    /// The least significant byte of `c`. Used to verify `shift_by_n_bits` and
    /// `shift_by_n_bytes`.
    pub c_least_sig_byte: [T; BYTE_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bits_to_shift = i`.
    pub shift_by_n_bits: [T; BYTE_SIZE],

    /// The number to multiply to shift `b` by `num_bits_to_shift`. (i.e., `2^num_bits_to_shift`)
    pub bit_shift_multiplier: T,

    /// The result of multiplying `b` by `bit_shift_multiplier`.
    pub bit_shift_result: [T; WORD_SIZE],

    /// The carry propagated when multiplying `b` by `bit_shift_multiplier`.
    pub bit_shift_result_carry: [T; WORD_SIZE],

    /// A boolean array whose `i`th element indicates whether `num_bytes_to_shift = i`.
    pub shift_by_n_bytes: [T; WORD_SIZE],
}

impl<F: PrimeField32> ShiftLeftOperation<F> {
    /// Populates the columns for `b << c`, returning the result.
    pub fn populate(&mut self, blu: &mut impl ByteRecord, b_u32: u32, c_u32: u32) -> u32 {
        let b = b_u32.to_le_bytes();

        for i in 0..BYTE_SIZE {
            self.c_least_sig_byte[i] = F::from_canonical_u32((c_u32 >> i) & 1);
        }

        let num_bits_to_shift = c_u32 as usize % BYTE_SIZE;
        for i in 0..BYTE_SIZE {
            self.shift_by_n_bits[i] = F::from_bool(num_bits_to_shift == i);
        }

        let bit_shift_multiplier = 1u32 << num_bits_to_shift;
        self.bit_shift_multiplier = F::from_canonical_u32(bit_shift_multiplier);

        let mut carry = 0u32;
        let base = 1u32 << BYTE_SIZE;
        let mut bit_shift_result = [0u8; WORD_SIZE];
        let mut bit_shift_result_carry = [0u8; WORD_SIZE];
        for i in 0..WORD_SIZE {
            let v = b[i] as u32 * bit_shift_multiplier + carry;
            carry = v / base;
            bit_shift_result[i] = (v % base) as u8;
            bit_shift_result_carry[i] = carry as u8;
        }
        self.bit_shift_result = bit_shift_result.map(F::from_canonical_u8);
        self.bit_shift_result_carry = bit_shift_result_carry.map(F::from_canonical_u8);

        let num_bytes_to_shift = (c_u32 & 0b11111) as usize / BYTE_SIZE;
        for i in 0..WORD_SIZE {
            self.shift_by_n_bytes[i] = F::from_bool(num_bytes_to_shift == i);
        }

        blu.add_u8_range_checks(&bit_shift_result);
        blu.add_u8_range_checks(&bit_shift_result_carry);

        let mut a = [0u8; WORD_SIZE];
        a[num_bytes_to_shift..WORD_SIZE]
            .copy_from_slice(&bit_shift_result[..(WORD_SIZE - num_bytes_to_shift)]);
        self.value = Word(a.map(F::from_canonical_u8));

        u32::from_le_bytes(a)
    }
}

impl<F: Field> ShiftLeftOperation<F> {
    /// Evaluates `b << c`, returning the result word (`cols.value`).
    pub fn eval<AB: ZKMCoreAirBuilder>(
        builder: &mut AB,
        b: Word<AB::Var>,
        c: Word<AB::Var>,
        cols: ShiftLeftOperation<AB::Var>,
        is_real: AB::Expr,
    ) -> Word<AB::Var> {
        let zero: AB::Expr = AB::F::ZERO.into();
        let one: AB::Expr = AB::F::ONE.into();
        let base: AB::Expr = AB::F::from_canonical_u32(1 << BYTE_SIZE).into();

        // Step 1: Perform the fine-grained bit shift (i.e., shifting b by c % 8 bits).
        let mut c_byte_sum = zero.clone();
        for i in 0..BYTE_SIZE {
            let val: AB::Expr = AB::F::from_canonical_u32(1 << i).into();
            c_byte_sum = c_byte_sum.clone() + val * cols.c_least_sig_byte[i];
        }
        builder.when(is_real.clone()).assert_eq(c_byte_sum, c[0]);

        let mut num_bits_to_shift = zero.clone();
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

        for i in 0..BYTE_SIZE {
            builder
                .when(is_real.clone())
                .when(cols.shift_by_n_bits[i])
                .assert_eq(cols.bit_shift_multiplier, AB::F::from_canonical_usize(1 << i));
        }

        for i in 0..WORD_SIZE {
            let mut v = b[i] * cols.bit_shift_multiplier
                - cols.bit_shift_result_carry[i] * base.clone();
            if i > 0 {
                v = v.clone() + cols.bit_shift_result_carry[i - 1].into();
            }
            builder.when(is_real.clone()).assert_eq(cols.bit_shift_result[i], v);
        }

        // Step 2: Perform the coarser byte shift (i.e., shifting b by c // 8 bits).
        let num_bytes_to_shift =
            cols.c_least_sig_byte[3] + cols.c_least_sig_byte[4] * AB::F::from_canonical_u32(2);

        for i in 0..WORD_SIZE {
            builder
                .when(is_real.clone())
                .when(cols.shift_by_n_bytes[i])
                .assert_eq(num_bytes_to_shift.clone(), AB::F::from_canonical_usize(i));
        }

        for num_bytes_to_shift in 0..WORD_SIZE {
            let mut shifting = builder
                .when(is_real.clone() * Into::<AB::Expr>::into(cols.shift_by_n_bytes[num_bytes_to_shift]));
            for i in 0..WORD_SIZE {
                if i < num_bytes_to_shift {
                    shifting.assert_eq(cols.value[i], zero.clone());
                } else {
                    shifting
                        .assert_eq(cols.value[i], cols.bit_shift_result[i - num_bytes_to_shift]);
                }
            }
        }

        // Step 3: Misc checks such as range checks & bool checks.
        for bit in cols.c_least_sig_byte.iter() {
            builder.when(is_real.clone()).assert_bool(*bit);
        }

        for shift in cols.shift_by_n_bits.iter() {
            builder.when(is_real.clone()).assert_bool(*shift);
        }
        builder.when(is_real.clone()).assert_eq(
            cols.shift_by_n_bits.iter().fold(zero.clone(), |acc, &x| acc + x),
            one.clone(),
        );

        builder.slice_range_check_u8(&cols.bit_shift_result, is_real.clone());
        builder.slice_range_check_u8(&cols.bit_shift_result_carry, is_real.clone());

        for shift in cols.shift_by_n_bytes.iter() {
            builder.when(is_real.clone()).assert_bool(*shift);
        }
        builder.when(is_real).assert_eq(
            cols.shift_by_n_bytes.iter().fold(zero.clone(), |acc, &x| acc + x),
            one,
        );

        cols.value
    }
}
