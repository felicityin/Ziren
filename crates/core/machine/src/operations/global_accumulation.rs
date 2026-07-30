use crate::operations::GlobalLookupOperation;
use p3_air::AirBuilder;
use p3_field::Field;
use p3_field::FieldAlgebra;
use p3_field::FieldExtensionAlgebra;
use p3_field::PrimeField32;
use std::iter::once;
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::SepticExtensionAirBuilder;
use zkm_hypercube::air::{AirLookup, LookupScope};
use zkm_hypercube::lookup::LookupKind;
use zkm_hypercube::septic_curve::SepticCurveComplete;
use zkm_hypercube::air::ZKMAirBuilder;
use zkm_hypercube::{
    septic_curve::SepticCurve,
    septic_extension::{SepticBlock, SepticExtension},
};

/// A set of columns needed to compute the global lookup elliptic curve digest.
/// It is critical that this struct is at the end of the main trace, as the permutation constraints will be dependent on this fact.
/// It is also critical the cumulative sum is at the end of this struct, for the same reason.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct GlobalAccumulationOperation<T, const N: usize> {
    pub initial_digest: [SepticBlock<T>; 2],
    pub cumulative_sum: [[SepticBlock<T>; 2]; N],
}

impl<T: Default, const N: usize> Default for GlobalAccumulationOperation<T, N> {
    fn default() -> Self {
        Self {
            initial_digest: core::array::from_fn(|_| SepticBlock::<T>::default()),
            cumulative_sum: core::array::from_fn(|_| {
                [SepticBlock::<T>::default(), SepticBlock::<T>::default()]
            }),
        }
    }
}

impl<F: PrimeField32, const N: usize> GlobalAccumulationOperation<F, N> {
    pub fn populate(
        &mut self,
        initial_digest: &mut SepticCurve<F>,
        global_lookup_cols: [GlobalLookupOperation<F>; N],
        is_real: [F; N],
    ) {
        self.initial_digest[0] = SepticBlock::from(initial_digest.x.0);
        self.initial_digest[1] = SepticBlock::from(initial_digest.y.0);

        for i in 0..N {
            let point_cur = SepticCurve {
                x: SepticExtension(global_lookup_cols[i].x_coordinate.0),
                y: SepticExtension(global_lookup_cols[i].y_coordinate.0),
            };
            assert!(is_real[i] == F::ONE || is_real[i] == F::ZERO);
            let sum_point = if is_real[i] == F::ONE {
                point_cur.add_incomplete(*initial_digest)
            } else {
                // Even off the real chain, this must be a genuine curve addition (not a no-op
                // repeat) so `sum_checker_x` holds unconditionally -- see `eval_accumulation`.
                initial_digest.add_incomplete(SepticCurve::<F>::dummy())
            };
            self.cumulative_sum[i][0] = SepticBlock::from(sum_point.x.0);
            self.cumulative_sum[i][1] = SepticBlock::from(sum_point.y.0);
            *initial_digest = sum_point;
        }
    }

    /// Populates a padding row: `start_digest` and each of the `N` steps' `dummy`-point additions
    /// are a genuine curve addition (not a no-op repeat), so `sum_checker_x` holds unconditionally
    /// in `eval_accumulation` without needing a witnessed column -- see its doc comment. The
    /// interaction multiplicity is zero on these rows regardless, so which valid point is used
    /// here doesn't matter beyond satisfying the local constraints.
    pub fn populate_dummy(&mut self, start_digest: SepticCurve<F>, dummy: SepticCurve<F>) {
        self.initial_digest[0] = SepticBlock::from(start_digest.x.0);
        self.initial_digest[1] = SepticBlock::from(start_digest.y.0);
        let mut current = start_digest;
        for i in 0..N {
            current = current.add_incomplete(dummy);
            self.cumulative_sum[i][0] = SepticBlock::from(current.x.0);
            self.cumulative_sum[i][1] = SepticBlock::from(current.y.0);
        }
    }

    pub fn populate_real(&mut self, sums: &[SepticCurveComplete<F>]) {
        debug_assert_eq!(sums.len(), N + 1);
        let sums = sums.iter().map(|complete_point| complete_point.point()).collect::<Vec<_>>();
        self.initial_digest[0] = SepticBlock::from(sums[0].x.0);
        self.initial_digest[1] = SepticBlock::from(sums[0].y.0);
        for i in 0..N {
            self.cumulative_sum[i][0] = SepticBlock::from(sums[i + 1].x.0);
            self.cumulative_sum[i][1] = SepticBlock::from(sums[i + 1].y.0);
        }
    }
}

impl<F: Field, const N: usize> GlobalAccumulationOperation<F, N> {
    pub fn eval_accumulation<AB: ZKMAirBuilder>(
        builder: &mut AB,
        global_lookup_cols: [GlobalLookupOperation<AB::Var>; N],
        local_is_real: [AB::Var; N],
        index: AB::Var,
        local_accumulation: GlobalAccumulationOperation<AB::Var, N>,
    ) {
        // First, constrain the control flow regarding `is_real`.
        // Constrain that all `is_real` values are boolean.
        for i in 0..N {
            builder.assert_bool(local_is_real[i]);
        }

        // Constrain that `is_real = 0` implies the next `is_real` values are all zero.
        for i in 0..N - 1 {
            // `is_real[i] == 0` implies `is_real[i + 1] == 0`.
            builder.when_not(local_is_real[i]).assert_zero(local_is_real[i + 1]);
        }

        // Next, constrain the accumulation.
        let initial_digest = SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.initial_digest[0][i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.initial_digest[1][i].into()
            }),
        };

        let assert_on_curve = |builder: &mut AB, point: SepticCurve<AB::Expr>| {
            builder.assert_septic_ext_eq(
                point.y.square(),
                SepticCurve::<AB::Expr>::curve_formula(point.x),
            );
        };

        let ith_cumulative_sum = |idx: usize| SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.cumulative_sum[idx][0].0[i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.cumulative_sum[idx][1].0[i].into()
            }),
        };

        let ith_point_to_add = |idx: usize| SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                global_lookup_cols[idx].x_coordinate.0[i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                global_lookup_cols[idx].y_coordinate.0[i].into()
            }),
        };

        // Receive this row's own claimed initial digest at `index`, matched by value against
        // whichever row sent it as its final digest at the same index -- or, for `index == 0`
        // and the genuinely last real row's `index + 1`, against the phantom send/receive pair
        // in `ExecutionRecord::eval_public_values` (mirrors SP1's `eval_global_sum`).
        builder.receive(
            AirLookup::new(
                once(index.into())
                    .chain(initial_digest.x.0.clone())
                    .chain(initial_digest.y.0.clone())
                    .collect(),
                local_is_real[0].into(),
                LookupKind::GlobalAccumulation,
            ),
            LookupScope::Local,
        );

        // Defense-in-depth: every witnessed running digest must stay on-curve even if the
        // incomplete Weierstrass addition edge case is triggered.
        assert_on_curve(builder, initial_digest.clone());

        // Constrain that a genuine curve addition (`current_sum + point_to_add == next_sum`) is
        // carried out on every row, real or not.
        for i in 0..N {
            let current_sum =
                if i == 0 { initial_digest.clone() } else { ith_cumulative_sum(i - 1) };
            let point_to_add = ith_point_to_add(i);
            let next_sum = ith_cumulative_sum(i);
            assert_on_curve(builder, next_sum.clone());

            // `sum_checker_x`/`_y` are both zero iff `current_sum + point_to_add == next_sum`.
            let sum_checker_x = SepticCurve::<AB::Expr>::sum_checker_x(
                current_sum.clone(),
                point_to_add.clone(),
                next_sum.clone(),
            );
            let sum_checker_y =
                SepticCurve::<AB::Expr>::sum_checker_y(current_sum, point_to_add, next_sum);
            // Enforced unconditionally, not gated by `is_real`: padding rows populate a genuine
            // dummy-point addition (see `populate_dummy`) rather than a no-op repeat, so this
            // holds honestly there too -- avoiding a witnessed column to reduce this degree-3
            // expression before gating it (which would otherwise push the gated constraint to
            // degree 4). The interaction multiplicity below is still `is_real`-gated, so a
            // padding row's resulting digest never needs to match anything.
            builder.assert_septic_ext_eq(sum_checker_x, SepticExtension::<AB::Expr>::zero());
            builder
                .when(local_is_real[i])
                .assert_septic_ext_eq(sum_checker_y, SepticExtension::<AB::Expr>::zero());
        }

        // Send this row's own final digest at `index + 1`, for whichever row receives it as its
        // initial digest at that index -- or, for the genuinely last real row, the phantom
        // receive in `ExecutionRecord::eval_public_values`.
        let final_digest = ith_cumulative_sum(N - 1);
        builder.send(
            AirLookup::new(
                once(index.into() + AB::Expr::one())
                    .chain(final_digest.x.0.clone())
                    .chain(final_digest.y.0.clone())
                    .collect(),
                local_is_real[N - 1].into(),
                LookupKind::GlobalAccumulation,
            ),
            LookupScope::Local,
        );
    }
}
