use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{
    air::{BaseAirBuilder, ZKMAirBuilder},
    word::Word,
};

/// The KoalaBear modulus's top 16 bits (`P = TOP_LIMB * 2^16 + 1`).
const TOP_LIMB: u32 = 0x7F00;

/// Range-checks that a [`Word`] is a canonical KoalaBear field element, i.e. `< P` where
/// `P = TOP_LIMB * 2^16 + 1`. Grouping the top two bytes into one 16-bit `high16` turns this into
/// a single less-than-`TOP_LIMB` check (via one `U16Range` lookup into the shared, already-paid-
/// for byte table) instead of a most-significant-byte bit decomposition -- see
/// `WordAddressOperation`'s identical technique, which this mirrors.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct KoalaBearWordRangeChecker<T> {
    /// 1 iff the value's top 16 bits are strictly less than `TOP_LIMB`.
    pub high16_lt_top_limb: T,
}

impl<F: Field> KoalaBearWordRangeChecker<F> {
    pub fn populate(&mut self, blu: &mut impl ByteRecord, value: u32) {
        let high16 = value >> 16;
        let lt = high16 < TOP_LIMB;
        self.high16_lt_top_limb = F::from_bool(lt);
        if lt {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U16Range,
                a1: (TOP_LIMB - 1 - high16) as u16,
                a2: 0,
                b: 0,
                c: 0,
            });
        } else {
            debug_assert_eq!(high16, TOP_LIMB, "a real word can't reach the KoalaBear modulus");
            debug_assert_eq!(value & 0xFFFF, 0);
        }
    }

    pub fn range_check<AB: ZKMAirBuilder>(
        builder: &mut AB,
        value: Word<AB::Var>,
        cols: KoalaBearWordRangeChecker<AB::Var>,
        is_real: AB::Expr,
    ) {
        if builder.try_emit_koala_bear_word_range_summary(
            value.map(|limb| AB::Expr::zero() + limb),
            is_real.clone(),
        ) {
            return;
        }

        let high16 = value[2].into() + value[3].into() * AB::Expr::from_canonical_u32(256);
        let low16 = value[0].into() + value[1].into() * AB::Expr::from_canonical_u32(256);
        let lt = cols.high16_lt_top_limb;
        builder.when(is_real.clone()).assert_bool(lt);

        // `lt == 1`: prove `high16 < TOP_LIMB` via one range-check on the slack -- if a cheating
        // prover set `high16 >= TOP_LIMB`, the slack underflows in the field to a value far
        // outside `[0, 2^16)`, which has no matching row in the (shared, already-paid-for)
        // `U16Range` table. `lt` alone (not `is_real * lt`) is the multiplicity; the safety
        // constraint right after forces `lt == 0` on padding rows so it can't be exploited for a
        // free lookup there.
        builder.send_byte(
            ByteOpcode::U16Range.as_field::<AB::F>(),
            AB::Expr::from_canonical_u32(TOP_LIMB - 1) - high16.clone(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            lt,
        );
        builder.when(lt).assert_one(is_real.clone());

        // `lt == 0`: the only way `high16` isn't `< TOP_LIMB` while `value` still reduces below
        // `P` is `high16 == TOP_LIMB` exactly, which further requires the low 16 bits to be
        // exactly 0.
        builder
            .when(is_real.clone())
            .when_not(lt)
            .assert_eq(high16, AB::Expr::from_canonical_u32(TOP_LIMB));
        builder.when(is_real).when_not(lt).assert_zero(low16);
    }
}
