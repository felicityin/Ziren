use itertools::izip;
use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, Opcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{
    air::{BaseAirBuilder, ZKMAirBuilder},
    word::Word,
};

/// Computes the MIPS SLT/SLTU (signed/unsigned set-less-than) comparison `b < c`, shared between
/// `LtChip` (real register-form SLT/SLTU, plus internal dependency checks from `DivRem`/`Branch`
/// reusing this circuit) and `SltiChip` (SLTI/SLTIU). `b`/`c` are supplied by the caller (already
/// stored/derived elsewhere, e.g. via an adapter's register access) rather than duplicated here --
/// mirrors `AddOperation`'s convention of taking its operands as parameters.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LtOperation<T: Copy> {
    /// The boolean result `b < c`, as a word (only byte 0 is ever nonzero).
    pub a: Word<T>,

    /// Whether this is a signed (SLT) comparison.
    pub is_slt: T,
    /// Whether this is an unsigned (SLTU) comparison.
    pub is_sltu: T,

    /// Boolean flag to indicate which byte pair differs if the operands are not equal.
    pub byte_flags: [T; 4],

    /// The masking b[3] & 0x7F.
    pub b_masked: T,
    /// The masking c[3] & 0x7F.
    pub c_masked: T,
    /// An inverse of differing byte if c_comp != b_comp.
    pub not_eq_inv: T,

    /// The most significant bit of operand b.
    pub msb_b: T,
    /// The most significant bit of operand c.
    pub msb_c: T,
    /// The multiplication msb_b * is_slt.
    pub bit_b: T,
    /// The multiplication msb_c * is_slt.
    pub bit_c: T,

    /// The result of the intermediate SLTU operation `b_comp < c_comp`.
    pub sltu: T,
    /// A boolean flag for an intermediate comparison.
    pub is_comp_eq: T,
    /// A boolean flag for comparing the sign bits.
    pub is_sign_eq: T,
    /// The comparison bytes to be looked up.
    pub comparison_bytes: [T; 2],
}

impl<F: PrimeField32> LtOperation<F> {
    /// Populates the operation's columns given the opcode (`SLT` or `SLTU`) and the two operand
    /// values, returning the boolean result (0 or 1).
    pub fn populate(&mut self, blu: &mut impl ByteRecord, opcode: Opcode, b_u32: u32, c_u32: u32) -> u32 {
        let b = b_u32.to_le_bytes();
        let c = c_u32.to_le_bytes();

        self.is_slt = F::from_bool(opcode == Opcode::SLT);
        self.is_sltu = F::from_bool(opcode == Opcode::SLTU);

        // If this is SLT, mask the MSB of b & c before computing the comparison.
        let masked_b = b[3] & 0x7f;
        let masked_c = c[3] & 0x7f;
        self.b_masked = F::from_canonical_u8(masked_b);
        self.c_masked = F::from_canonical_u8(masked_c);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: masked_b as u16,
            a2: 0,
            b: b[3],
            c: 0x7f,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: masked_c as u16,
            a2: 0,
            b: c[3],
            c: 0x7f,
        });

        let mut b_comp = b;
        let mut c_comp = c;
        if opcode == Opcode::SLT {
            b_comp[3] = masked_b;
            c_comp[3] = masked_c;
        }
        self.sltu = F::from_bool(b_comp < c_comp);
        self.is_comp_eq = F::from_bool(b_comp == c_comp);

        // Set the byte equality flags.
        for (b_byte, c_byte, flag) in
            izip!(b_comp.iter().rev(), c_comp.iter().rev(), self.byte_flags.iter_mut().rev())
        {
            if c_byte != b_byte {
                *flag = F::ONE;
                self.sltu = F::from_bool(b_byte < c_byte);
                let b_byte = F::from_canonical_u8(*b_byte);
                let c_byte = F::from_canonical_u8(*c_byte);
                self.not_eq_inv = (b_byte - c_byte).inverse();
                self.comparison_bytes = [b_byte, c_byte];
                break;
            }
        }

        self.msb_b = F::from_canonical_u8((b[3] >> 7) & 1);
        self.msb_c = F::from_canonical_u8((c[3] >> 7) & 1);
        self.is_sign_eq = if opcode == Opcode::SLT {
            F::from_bool((b[3] >> 7) == (c[3] >> 7))
        } else {
            F::ONE
        };

        self.bit_b = self.msb_b * self.is_slt;
        self.bit_c = self.msb_c * self.is_slt;

        let result = self.bit_b * (F::ONE - self.bit_c) + self.is_sign_eq * self.sltu;
        self.a = Word([result, F::ZERO, F::ZERO, F::ZERO]);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: self.sltu.as_canonical_u32() as u16,
            a2: 0,
            b: self.comparison_bytes[0].as_canonical_u32() as u8,
            c: self.comparison_bytes[1].as_canonical_u32() as u8,
        });

        result.as_canonical_u32()
    }
}

impl<F: Field> LtOperation<F> {
    /// Evaluates the comparison, asserting `cols.a == (b < c)` under the signed/unsigned mode
    /// selected by `cols.is_slt`/`cols.is_sltu`. `is_real` gates every constraint here (both real,
    /// retired instructions and internal dependency-check rows reusing this circuit should pass
    /// `is_slt + is_sltu`-shaped gating; padding rows must pass zero).
    pub fn eval<AB: ZKMAirBuilder>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        cols: LtOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        // We can compute the signed set-less-than as follows:
        // SLT (signed) = b_s * (1 - c_s) + (b_s == c_s) * SLTU(b_<s, c_<s)
        // Source: Jolt 5.3: Set Less Than (https://people.cs.georgetown.edu/jthaler/Jolt-paper.pdf)

        // We will compute SLTU(b_comp, c_comp) where `b_comp` and `c_comp` are:
        // * if the operation is `SLTU`, `b_comp = b` and `c_comp = c`
        // * if the operation is `SLT`, `b_comp = b & 0x7FFFFFFF` and `c_comp = c & 0x7FFFFFFF`
        //
        // We will set booleans `b_bit` and `c_bit` so that:
        // * If the operation is `SLTU`, then `b_bit = 0` and `c_bit = 0`.
        // * If the operation is `SLT`, then `b_bit`, `c_bit` are the most significant bits of `b`
        //   and `c` respectively.
        //
        // Then, we will compute the answer as:
        // SLT = b_bit * (1 - c_bit) + (b_bit == c_bit) * SLTU(b_comp, c_comp)

        // First, we set up the values of `b_comp` and `c_comp`.
        let mut b_comp = b.clone();
        let mut c_comp = c.clone();

        b_comp[3] = b[3].clone() * cols.is_sltu + cols.b_masked * cols.is_slt;
        c_comp[3] = c[3].clone() * cols.is_sltu + cols.c_masked * cols.is_slt;

        // Constrain the `masked_b` and `masked_c` values via lookup.
        //
        // The values are given by `b_masked = b[3] & 0x7F` and `c_masked = c[3] & 0x7F`.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            cols.b_masked,
            b[3].clone(),
            AB::F::from_canonical_u8(0x7f),
            is_real.clone(),
        );
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            cols.c_masked,
            c[3].clone(),
            AB::F::from_canonical_u8(0x7f),
            is_real.clone(),
        );

        // Set the values of `b_bit` and `c_bit`.
        builder.assert_eq(cols.bit_b, cols.msb_b * cols.is_slt);
        builder.assert_eq(cols.bit_c, cols.msb_c * cols.is_slt);

        // Assert the correctness of `cols.msb_b` and `cols.msb_c` using the mask.
        let inv_128 = AB::F::from_canonical_u32(128).inverse();
        builder.assert_eq(cols.msb_b, (b[3].clone() - cols.b_masked) * inv_128);
        builder.assert_eq(cols.msb_c, (c[3].clone() - cols.c_masked) * inv_128);

        // Constrain that when is_sign_eq = (bit_b == bit_c).

        // assert the flag is a boolean.
        builder.assert_bool(cols.is_sign_eq);

        // assert the correction of the comparison.
        builder.when(cols.is_sign_eq).assert_eq(cols.bit_b, cols.bit_c);
        builder
            .when(is_real.clone())
            .when_not(cols.is_sign_eq)
            .assert_one(cols.bit_b + cols.bit_c);

        // Assert the final result `a` is correct.

        // Check that `a[0]` is set correctly.
        builder.assert_eq(
            cols.a[0],
            cols.bit_b * (AB::Expr::one() - cols.bit_c) + cols.is_sign_eq * cols.sltu,
        );
        // Check the 3 most significant bytes of `a` are zero.
        builder.assert_zero(cols.a[1]);
        builder.assert_zero(cols.a[2]);
        builder.assert_zero(cols.a[3]);

        // Verify that the byte equality flags are set correctly, i.e. all are boolean and only
        // at most a single byte flag is set.
        let sum_flags =
            cols.byte_flags[0] + cols.byte_flags[1] + cols.byte_flags[2] + cols.byte_flags[3];
        builder.assert_bool(cols.byte_flags[0]);
        builder.assert_bool(cols.byte_flags[1]);
        builder.assert_bool(cols.byte_flags[2]);
        builder.assert_bool(cols.byte_flags[3]);
        builder.assert_bool(sum_flags.clone());
        builder.when(is_real.clone()).assert_eq(AB::Expr::one() - cols.is_comp_eq, sum_flags);

        // Constrain `cols.sltu == SLTU(b_comp, c_comp)`.
        //
        // We define bytes `b_comp_byte` and `c_comp_byte` as follows: If `b_comp == c_comp`, then
        // `b_comp_byte = c_comp_byte = 0`. Otherwise, we set `b_comp_byte` and `c_comp_byte` to
        // the first differing byte (in most significant order). We will use the `cols.is_comp_eq`
        // flag to indicate whether the bytes are equal.

        // Check the equality flag is boolean.
        builder.assert_bool(cols.is_comp_eq);

        // Find the differing byte if `b_comp != c_comp` and assert equality in case the flag
        // `cols.is_comp_eq` is set to `1`.

        // A flag to indicate whether an equality check is necessary (this is for all bytes from
        // most significant until the first inequality.
        let mut is_inequality_visited = AB::Expr::zero();

        // Expressions for computing the comparison bytes.
        let mut b_comparison_byte = AB::Expr::zero();
        let mut c_comparison_byte = AB::Expr::zero();
        // Iterate over the bytes in reverse order and select the differing bytes using the byte
        // flag columns values.
        for (b_byte, c_byte, &flag) in
            izip!(b_comp.0.iter().rev(), c_comp.0.iter().rev(), cols.byte_flags.iter().rev())
        {
            // Once the byte flag was set to one, we turn off the quality check flag.
            // We can do this by calculating the sum of the flags since only `1` is set to `1`.
            is_inequality_visited = is_inequality_visited.clone() + flag.into();

            b_comparison_byte = b_comparison_byte.clone() + b_byte.clone() * flag;
            c_comparison_byte = c_comparison_byte.clone() + c_byte.clone() * flag;

            // If inequality is not visited, assert that the bytes are equal.
            builder
                .when_not(is_inequality_visited.clone())
                .assert_eq(b_byte.clone(), c_byte.clone());
            // If the numbers are assumed equal, inequality should not be visited.
            builder.when(cols.is_comp_eq).assert_zero(is_inequality_visited.clone());
        }
        // We need to verify that the comparison bytes are set correctly. This is only relevant in
        // the case where the bytes are not equal.

        // Constrain the row comparison byte values to be equal to the calculated ones.
        let (b_comp_byte, c_comp_byte) = (cols.comparison_bytes[0], cols.comparison_bytes[1]);
        builder.assert_eq(b_comp_byte, b_comparison_byte);
        builder.assert_eq(c_comp_byte, c_comparison_byte);

        // Using the values above, we can constrain the `cols.is_comp_eq` flag. We already asserted
        // in the loop that when `cols.is_comp_eq == 1` then all bytes are equal. It is left to
        // verify that when `cols.is_comp_eq == 0` the comparison bytes are indeed not equal.
        // This is done using the inverse hint `not_eq_inv`.
        builder
            .when_not(cols.is_comp_eq)
            .assert_eq(cols.not_eq_inv * (b_comp_byte - c_comp_byte), is_real.clone());

        // Now the value of `cols.sltu` is equal to the same value for the comparison bytes.
        //
        // Set `cols.sltu = SLTU(b_comp_byte, c_comp_byte)` via a lookup.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            cols.sltu,
            b_comp_byte,
            c_comp_byte,
            is_real.clone(),
        );

        // Constrain the operation flags.

        // Check that the operation flags are boolean.
        builder.assert_bool(cols.is_slt);
        builder.assert_bool(cols.is_sltu);
        // Check that at most one of the operation flags is set.
        //
        // *remark*: this is not strictly necessary since it's also covered by the bus multiplicity
        // but this is included here to make sure the condition is met.
        builder.assert_bool(cols.is_slt + cols.is_sltu);
    }
}
