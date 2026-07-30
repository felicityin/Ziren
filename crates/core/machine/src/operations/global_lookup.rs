use p3_air::AirBuilder;
use p3_field::Field;
use p3_field::FieldAlgebra;
use p3_field::FieldExtensionAlgebra;
use p3_field::PrimeField32;
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::ZKMAirBuilder;
use zkm_hypercube::{
    septic_curve::{SepticCurve, CURVE_WITNESS_DUMMY_POINT_X, CURVE_WITNESS_DUMMY_POINT_Y},
    septic_extension::{SepticBlock, SepticExtension},
};

/// A set of columns needed to compute the global interaction elliptic curve digest.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct GlobalLookupOperation<T: Copy> {
    pub offset_bits: [T; 8],
    pub x_coordinate: SepticBlock<T>,
    pub y_coordinate: SepticBlock<T>,
    /// Little-endian byte decomposition of the y-coordinate's sign/magnitude range check value
    /// (see `eval_single_digest`) -- range-checked via the shared byte table (`U8Range`/`LTU`)
    /// instead of a full bit decomposition, since `is_receive`/`is_send`/`is_exception` (see their
    /// doc comments) guarantee it fits in `[0, 63 * 2^24)`.
    pub y6_byte_decomp: [T; 4],
}

impl<F: PrimeField32> GlobalLookupOperation<F> {
    pub fn get_digest(
        values: SepticBlock<u32>,
        is_receive: bool,
        kind: u8,
    ) -> (SepticCurve<F>, u8) {
        let x_start = SepticExtension::<F>::from_base_fn(|i| F::from_canonical_u32(values.0[i]))
            + SepticExtension::from_base(F::from_canonical_u32((kind as u32) << 16));
        let (point, offset) = SepticCurve::<F>::lift_x(x_start);
        if !is_receive {
            return (point.neg(), offset);
        }
        (point, offset)
    }

    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        values: SepticBlock<u32>,
        is_receive: bool,
        is_real: bool,
        kind: u8,
    ) {
        if is_real {
            let (point, offset) = Self::get_digest(values, is_receive, kind);
            for i in 0..8 {
                self.offset_bits[i] = F::from_canonical_u8((offset >> i) & 1);
            }
            self.x_coordinate = SepticBlock::<F>::from(point.x.0);
            self.y_coordinate = SepticBlock::<F>::from(point.y.0);
            let range_check_value = if is_receive {
                point.y.0[6].as_canonical_u32() - 1
            } else {
                F::ORDER_U32 - point.y.0[6].as_canonical_u32() - 1
            };
            assert!(range_check_value < 63 * (1 << 24));
            let bytes = range_check_value.to_le_bytes();
            self.y6_byte_decomp = core::array::from_fn(|i| F::from_canonical_u8(bytes[i]));
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U8Range,
                a1: 0,
                a2: 0,
                b: bytes[0],
                c: bytes[1],
            });
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U8Range,
                a1: 0,
                a2: 0,
                b: bytes[2],
                c: 0,
            });
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: bytes[3],
                c: 63,
            });
        } else {
            self.populate_dummy();
        }
    }

    pub fn populate_dummy(&mut self) {
        for i in 0..8 {
            self.offset_bits[i] = F::ZERO;
        }
        self.x_coordinate = SepticBlock::<F>::from_base_fn(|i| {
            F::from_canonical_u32(CURVE_WITNESS_DUMMY_POINT_X[i])
        });
        self.y_coordinate = SepticBlock::<F>::from_base_fn(|i| {
            F::from_canonical_u32(CURVE_WITNESS_DUMMY_POINT_Y[i])
        });
        self.y6_byte_decomp = [F::ZERO; 4];
    }
}

impl<F: Field> GlobalLookupOperation<F> {
    /// Constrain that the elliptic curve point for the global interaction is correctly derived.
    pub fn eval_single_digest<AB: ZKMAirBuilder + p3_air::PairBuilder>(
        builder: &mut AB,
        values: [AB::Expr; 7],
        cols: GlobalLookupOperation<AB::Var>,
        is_receive: AB::Expr,
        is_send: AB::Expr,
        is_real: AB::Var,
        kind: AB::Var,
    ) {
        // Constrain that the `is_real` is boolean.
        builder.assert_bool(is_real);

        // Compute the offset and range check each bits, ensuring that the offset is a byte.
        let mut offset = AB::Expr::zero();
        for i in 0..8 {
            builder.assert_bool(cols.offset_bits[i]);
            offset = offset.clone() + cols.offset_bits[i] * AB::F::from_canonical_u32(1 << i);
        }

        // Range check the first element in the message to be a u16 so that we can encode the interaction kind in the upper 8 bits.
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            values[0].clone(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real,
        );

        let x = SepticExtension::<AB::Expr>::from_base_fn(|i| cols.x_coordinate[i].into());
        let y = SepticExtension::<AB::Expr>::from_base_fn(|i| cols.y_coordinate[i].into());

        // Constrain that x_coordinate is derived from (values, kind, offset) via the
        // map-to-curve function. This is the critical link between the tuple columns
        // (which participate in the cross-table lookup) and the witness curve point
        // (which is accumulated into the global digest).
        //
        // The map-to-curve computes:
        //   x[0] = values[0] + kind * 65536
        //   x[i] = values[i]              for i in 1..6
        //   x[6] = values[6] * 256 + offset
        builder.when(is_real).assert_eq(
            x.0[0].clone(),
            values[0].clone() + kind.into() * AB::Expr::from_canonical_u32(65536),
        );
        for i in 1..6 {
            builder.when(is_real).assert_eq(x.0[i].clone(), values[i].clone());
        }
        builder.when(is_real).assert_eq(
            x.0[6].clone(),
            values[6].clone() * AB::Expr::from_canonical_u32(256) + offset.clone(),
        );

        // Constrain that `(x, y)` is a valid point on the curve.
        let y2 = y.square();
        let x3_3zx_m3 = SepticCurve::<AB::Expr>::curve_formula(x);
        builder.assert_septic_ext_eq(y2, x3_3zx_m3);

        // Constrain that `0 <= y6_value < 63 * 2^24 < (p - 1) / 2`, via a 4-byte decomposition
        // range-checked against the shared byte table (`U8Range` on the low 3 bytes, `LTU` on the
        // top byte) instead of a full bit decomposition -- `is_receive`/`is_send`'s narrower bands
        // (see their doc comments) guarantee this always fits.
        let mut y6_value = AB::Expr::zero();
        for i in 0..4 {
            y6_value =
                y6_value.clone() + cols.y6_byte_decomp[i] * AB::F::from_canonical_u32(1 << (8 * i));
        }
        builder.send_byte(
            ByteOpcode::U8Range.as_field::<AB::F>(),
            AB::Expr::zero(),
            cols.y6_byte_decomp[0].into(),
            cols.y6_byte_decomp[1].into(),
            is_real,
        );
        builder.send_byte(
            ByteOpcode::U8Range.as_field::<AB::F>(),
            AB::Expr::zero(),
            cols.y6_byte_decomp[2].into(),
            AB::Expr::zero(),
            is_real,
        );
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            cols.y6_byte_decomp[3].into(),
            AB::Expr::from_canonical_u32(63),
            is_real,
        );

        // Constrain that y has correct sign.
        // If it's a receive: `1 <= y_6 <= 63 * 2^24`, so `0 <= y_6 - 1 = y6_value < 63 * 2^24`.
        // If it's a send: `p - 63 * 2^24 <= y_6 <= p - 1`, so `0 <= p - 1 - y_6 = y6_value < 63 * 2^24`.
        builder.when(is_receive).assert_eq(y.0[6].clone(), AB::Expr::one() + y6_value.clone());
        builder.when(is_send).assert_zero(y.0[6].clone() + AB::Expr::one() + y6_value);
    }
}
