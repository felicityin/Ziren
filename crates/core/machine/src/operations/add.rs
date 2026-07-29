#[cfg(feature = "picus")]
use core::borrow::Borrow;
#[cfg(feature = "picus")]
use core::mem::{size_of, transmute};

use zkm_core_executor::events::ByteRecord;
use zkm_primitives::consts::WORD_SIZE;
use zkm_hypercube::{air::ZKMAirBuilder, word::Word};

use p3_field::{Field, FieldAlgebra};
use slop_air::AirBuilder;
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusProjection;

use crate::air::WordAirBuilder;
#[cfg(feature = "picus")]
use crate::utils::indices_arr;

/// A set of columns needed to compute the add of two words.
///
/// The carry bits are not witnessed as their own columns: `eval` derives each one algebraically
/// from `a`, `b`, and `value` (dividing by the limb base, which is invertible mod p), then asserts
/// it's boolean. This is sound because a correctly computed `value` makes that division come out
/// to the true 0/1 carry; a forged `value` generically doesn't land on {0, 1}.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddOperation<T> {
    /// The result of `a + b`.
    pub value: Word<T>,
}

#[cfg(feature = "picus")]
const NUM_ADD_OPERATION_SUMMARY_COLS: usize = size_of::<AddOperationSummaryCols<u8>>();

#[cfg(feature = "picus")]
const ADD_OPERATION_SUMMARY_COL_MAP: AddOperationSummaryCols<usize> =
    make_add_operation_summary_col_map();

#[cfg(feature = "picus")]
const fn make_add_operation_summary_col_map() -> AddOperationSummaryCols<usize> {
    let indices_arr = indices_arr::<NUM_ADD_OPERATION_SUMMARY_COLS>();
    unsafe {
        transmute::<[usize; NUM_ADD_OPERATION_SUMMARY_COLS], AddOperationSummaryCols<usize>>(
            indices_arr,
        )
    }
}

/// Hidden witness layout used when Picus emits the exact two-word add AIR as a
/// local auxiliary module.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
#[cfg(feature = "picus")]
struct AddOperationSummaryCols<T> {
    pub a: Word<T>,
    pub b: Word<T>,
    pub is_real: T,
    pub cols: AddOperation<T>,
}

#[cfg(feature = "picus")]
#[cfg_attr(feature = "picus", derive(PicusProjection))]
#[cfg_attr(feature = "picus", picus_projection(
    source = AddOperationSummaryCols<u8>,
    col_map = ADD_OPERATION_SUMMARY_COL_MAP
))]
#[allow(dead_code)]
struct AddOperationSummaryProjection {
    #[cfg_attr(feature = "picus", picus(input, path = a))]
    pub a: Word<u8>,
    #[cfg_attr(feature = "picus", picus(input, path = b))]
    pub b: Word<u8>,
    #[cfg_attr(feature = "picus", picus(output, path = cols.value))]
    pub value: Word<u8>,
}

impl<F: Field> AddOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, a_u32: u32, b_u32: u32) -> u32 {
        let expected = a_u32.wrapping_add(b_u32);
        self.value = Word::from(expected);

        // Range check
        {
            record.add_u8_range_checks(&a_u32.to_le_bytes());
            record.add_u8_range_checks(&b_u32.to_le_bytes());
            record.add_u8_range_checks(&expected.to_le_bytes());
        }
        expected
    }

    fn eval_exact<AB: ZKMAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        b: Word<AB::Var>,
        cols: AddOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        let base_inv: AB::Expr = AB::F::from_canonical_u32(256).inverse().into();
        let mut builder_is_real = builder.when(is_real.clone());

        // The set of constraints are:
        //  - carry is initialized to zero
        //  - 256 * carry_next + value[i] = a[i] + b[i] + carry
        //  - carry is boolean
        // `carry` is derived algebraically (dividing by the limb base, which is invertible mod p)
        // instead of being witnessed as its own column -- see the doc comment on the struct.
        let mut carry = AB::Expr::zero();
        for i in 0..WORD_SIZE {
            carry = (Into::<AB::Expr>::into(a[i]) + Into::<AB::Expr>::into(b[i])
                - Into::<AB::Expr>::into(cols.value[i])
                + carry)
                * base_inv.clone();
            builder_is_real.assert_bool(carry.clone());
        }
        builder_is_real.assert_bool(is_real.clone());

        // Range check each byte.
        {
            builder.slice_range_check_u8(&a.0, is_real.clone());
            builder.slice_range_check_u8(&b.0, is_real.clone());
            builder.slice_range_check_u8(&cols.value.0, is_real);
        }
    }

    pub fn eval<AB: ZKMAirBuilder>(
        builder: &mut AB,
        a: Word<AB::Var>,
        b: Word<AB::Var>,
        cols: AddOperation<AB::Var>,
        is_real: AB::Expr,
    ) {
        #[cfg(feature = "picus")]
        {
            let mut current_inputs: Vec<AB::Expr> = Vec::with_capacity(WORD_SIZE * 2);
            for limb in a.0 {
                current_inputs.push(limb.into());
            }
            for limb in b.0 {
                current_inputs.push(limb.into());
            }

            let current_outputs: Vec<AB::Expr> =
                cols.value.0.iter().map(|limb| (*limb).into()).collect();

            if builder.is_known_one(&is_real)
                && builder.try_emit_projected_summary_with_hidden_consts(
                    "AddOperation",
                    &AddOperationSummaryProjection::picus_projection_info(),
                    &current_inputs,
                    &current_outputs,
                    size_of::<AddOperationSummaryCols<u8>>(),
                    &[(ADD_OPERATION_SUMMARY_COL_MAP.is_real, 1)],
                    |builder, source_row| {
                        let source: &AddOperationSummaryCols<AB::Var> = (*source_row).borrow();
                        Self::eval_exact(
                            builder,
                            source.a,
                            source.b,
                            source.cols,
                            source.is_real.into(),
                        );
                    },
                )
            {
                return;
            }
        }

        Self::eval_exact(builder, a, b, cols, is_real);
    }
}
