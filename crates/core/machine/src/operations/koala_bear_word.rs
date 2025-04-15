use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_stark::{air::ZKMAirBuilder, BaseAirBuilder, Word};

/// A set of columns needed to range check a KoalaBear word.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct KoalaBearWordRangeChecker<T> {
    /// Most sig byte is less than 127.
    pub most_sig_byte_lt_127: T,
}

impl<F: PrimeField32> KoalaBearWordRangeChecker<F> {
    pub fn populate(&mut self, value: Word<F>, record: &mut impl ByteRecord) {
        let ms_byte_u8 = value[3].as_canonical_u32() as u8;
        self.most_sig_byte_lt_127 = F::from_bool(ms_byte_u8 < 127);

        // Add the byte lookup for the range check bit.
        record.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: if ms_byte_u8 < 127 { 1 } else { 0 },
            a2: 0,
            b: ms_byte_u8,
            c: 127,
        });
    }
}

impl<F: Field> KoalaBearWordRangeChecker<F> {
    pub fn range_check<AB: ZKMAirBuilder>(
        builder: &mut AB,
        value: Word<AB::Var>,
        cols: KoalaBearWordRangeChecker<AB::Var>,
        is_real: AB::Expr,
    ) {
        // Range check that value is less than koala bear modulus.  To do this, it is sufficient
        // to just do comparisons for the most significant byte. KoalaBear's modulus is (in big
        // endian binary) 01111111_00000000_00000000_00000001.  So we need to check the
        // following conditions:
        // 1) if most_sig_byte > 01111111, then fail.
        // 2) if most_sig_byte == 01111111, then value's lower sig bytes must all be 0.
        // 3) if most_sig_byte < 01111111, then pass.

        let ms_byte = value[3];

        // The range check bit is on if and only if the most significant byte of the word is < 127.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            cols.most_sig_byte_lt_127,
            ms_byte,
            AB::Expr::from_canonical_u8(127),
            is_real.clone(),
        );

        let mut is_real_builder = builder.when(is_real.clone());

        // If the range check bit is off, the most significant byte is >=127, so to be a valid KoalaBear
        // word we need the most significant byte to be =127.
        is_real_builder
            .when_not(cols.most_sig_byte_lt_127)
            .assert_eq(ms_byte, AB::Expr::from_canonical_u8(127));

        // Moreover, if the most significant byte =127, then the 3 other bytes must all be zero.
        let mut assert_zero_builder = is_real_builder.when_not(cols.most_sig_byte_lt_127);
        assert_zero_builder.assert_zero(value[0]);
        assert_zero_builder.assert_zero(value[1]);
        assert_zero_builder.assert_zero(value[2]);
    }
}
