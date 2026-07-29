use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::word::Word;
use zkm_primitives::consts::WORD_SIZE;

use crate::air::ZKMCoreAirBuilder;

/// The number of bytes in the 64-bit product of two 32-bit words.
pub const PRODUCT_SIZE: usize = 2 * WORD_SIZE;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// The mask for a byte.
const BYTE_MASK: u8 = 0xff;

fn get_msb(a: [u8; WORD_SIZE]) -> u8 {
    (a[WORD_SIZE - 1] >> (BYTE_SIZE - 1)) & 1
}

/// Computes and validates the (sign-aware) 64-bit product `b * c` of two 32-bit words, shared by
/// every chip that needs a multiply's full lo/hi result: `MulChip` (real MUL/MULT/MULTU
/// instructions), and -- as an embedded, no-cross-chip-lookup copy -- `DivRemChip`'s `c *
/// quotient` overflow check and `MaddsubChip`'s MADD/MADDU/MSUB/MSUBU accumulate-multiply.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MulOperation<T> {
    /// Trace.
    pub carry: [T; PRODUCT_SIZE],

    /// The product of `b * c` after carry propagation; `product[0..4]` is the low word,
    /// `product[4..8]` the high word.
    pub product: [T; PRODUCT_SIZE],

    /// The most significant bit of `b`.
    pub b_msb: T,

    /// The most significant bit of `c`.
    pub c_msb: T,

    /// The sign extension of `b`.
    pub b_sign_extend: T,

    /// The sign extension of `c`.
    pub c_sign_extend: T,
}

impl<F: PrimeField32> MulOperation<F> {
    /// Populates the columns for the (possibly signed) product `b * c`, returning `(lo, hi)`.
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        b_u32: u32,
        c_u32: u32,
        is_signed: bool,
    ) -> (u32, u32) {
        let b_word = b_u32.to_le_bytes();
        let c_word = c_u32.to_le_bytes();

        let mut b = b_word.to_vec();
        let mut c = c_word.to_vec();

        let b_msb = get_msb(b_word);
        self.b_msb = F::from_canonical_u8(b_msb);
        let c_msb = get_msb(c_word);
        self.c_msb = F::from_canonical_u8(c_msb);

        // If b is signed and it is negative, sign extend b.
        if is_signed && b_msb == 1 {
            self.b_sign_extend = F::ONE;
            b.resize(PRODUCT_SIZE, BYTE_MASK);
        } else {
            self.b_sign_extend = F::ZERO;
        }

        // If c is signed and it is negative, sign extend c.
        if is_signed && c_msb == 1 {
            self.c_sign_extend = F::ONE;
            c.resize(PRODUCT_SIZE, BYTE_MASK);
        } else {
            self.c_sign_extend = F::ZERO;
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

        let mut product = [0u32; PRODUCT_SIZE];
        for i in 0..b.len() {
            for j in 0..c.len() {
                if i + j < PRODUCT_SIZE {
                    product[i + j] += (b[i] as u32) * (c[j] as u32);
                }
            }
        }

        // Calculate the correct product using the `product` array. We store the correct carry
        // value for verification.
        let base = (1 << BYTE_SIZE) as u32;
        let mut carry = [0u32; PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            carry[i] = product[i] / base;
            product[i] %= base;
            if i + 1 < PRODUCT_SIZE {
                product[i + 1] += carry[i];
            }
            self.carry[i] = F::from_canonical_u32(carry[i]);
        }

        self.product = product.map(F::from_canonical_u32);

        // Range check.
        blu.add_u16_range_checks(&carry.map(|x| x as u16));
        blu.add_u8_range_checks(&product.map(|x| x as u8));

        let lo = u32::from_le_bytes(core::array::from_fn(|i| product[i] as u8));
        let hi = u32::from_le_bytes(core::array::from_fn(|i| product[WORD_SIZE + i] as u8));
        (lo, hi)
    }
}

impl<F: Field> MulOperation<F> {
    /// Evaluates the (possibly signed) product `b * c`, returning `(lo, hi)` words built directly
    /// from the witnessed `product` bytes.
    ///
    /// Assumes `is_signed` is boolean -- the caller's own selector logic (e.g. `MulChip`'s
    /// `is_mult`, `DivRemChip`'s `is_div + is_mod`) must guarantee this.
    pub fn eval<AB: ZKMCoreAirBuilder>(
        builder: &mut AB,
        b_word: Word<AB::Var>,
        c_word: Word<AB::Var>,
        cols: MulOperation<AB::Var>,
        is_signed: AB::Expr,
        is_real: AB::Expr,
    ) -> (Word<AB::Var>, Word<AB::Var>) {
        let base = AB::F::from_canonical_u32(1 << 8);
        let one: AB::Expr = AB::F::ONE.into();
        let zero: AB::Expr = AB::F::ZERO.into();
        let byte_mask = AB::F::from_canonical_u8(BYTE_MASK);

        // Calculate the MSBs.
        let msb_opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);
        for (msb, byte) in
            [(cols.b_msb, b_word[WORD_SIZE - 1]), (cols.c_msb, c_word[WORD_SIZE - 1])]
        {
            builder.send_byte(msb_opcode, msb, byte, zero.clone(), is_real.clone());
        }

        // Calculate whether to extend b and c's sign.
        builder.when(is_real.clone()).assert_eq(cols.b_sign_extend, is_signed.clone() * cols.b_msb);
        builder.when(is_real.clone()).assert_eq(cols.c_sign_extend, is_signed * cols.c_msb);

        // Sign extend b and c whenever appropriate.
        let mut b: Vec<AB::Expr> = vec![zero.clone(); PRODUCT_SIZE];
        let mut c: Vec<AB::Expr> = vec![zero.clone(); PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            if i < WORD_SIZE {
                b[i] = b_word[i].into();
                c[i] = c_word[i].into();
            } else {
                b[i] = cols.b_sign_extend.into() * byte_mask;
                c[i] = cols.c_sign_extend.into() * byte_mask;
            }
        }

        // Compute the uncarried product b(x) * c(x) = m(x).
        let mut m: Vec<AB::Expr> = vec![zero.clone(); PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            for j in 0..PRODUCT_SIZE {
                if i + j < PRODUCT_SIZE {
                    m[i + j] = m[i + j].clone() + b[i].clone() * c[j].clone();
                }
            }
        }

        // Propagate carry. Gated by `is_real`: unlike `MulChip` (whose `b`/`c` come from a
        // dedicated register reader that's naturally zero when `is_real` is 0), a caller whose
        // `b`/`c` columns are shared with other opcode families (e.g. via a `union`) can have
        // them genuinely nonzero on a row where this particular product isn't the one being
        // proved -- gating on `is_real` unconditionally keeps this operation safe to embed either
        // way.
        for i in 0..PRODUCT_SIZE {
            if i == 0 {
                builder
                    .when(is_real.clone())
                    .assert_eq(m[i].clone(), cols.carry[i] * base + cols.product[i]);
            } else {
                builder.when(is_real.clone()).assert_eq(
                    cols.product[i] - cols.carry[i - 1] + cols.carry[i] * base,
                    m[i].clone(),
                );
            }
        }

        // Check that the boolean values are indeed boolean values. Gated by `is_real`: a caller
        // that shares these columns with other opcode families via a `union` has genuinely
        // arbitrary bytes here on a row where this operation isn't the active view, not just
        // zeros.
        for boolean in [cols.b_msb, cols.c_msb, cols.b_sign_extend, cols.c_sign_extend] {
            builder.when(is_real.clone()).assert_bool(boolean);
        }

        // If signed extended, the MSB better be 1.
        builder.when(is_real.clone()).when(cols.b_sign_extend).assert_eq(cols.b_msb, one.clone());
        builder.when(is_real.clone()).when(cols.c_sign_extend).assert_eq(cols.c_msb, one);

        // Range check.
        {
            // Ensure that the carry is at most 2^16. This ensures that
            // product_before_carry_propagation - carry * base + last_carry never overflows or
            // underflows enough to "wrap" around to create a second solution.
            builder.slice_range_check_u16(&cols.carry, is_real.clone());
            builder.slice_range_check_u8(&cols.product, is_real);
        }

        let lo = Word([cols.product[0], cols.product[1], cols.product[2], cols.product[3]]);
        let hi = Word([cols.product[4], cols.product[5], cols.product[6], cols.product[7]]);
        (lo, hi)
    }
}
