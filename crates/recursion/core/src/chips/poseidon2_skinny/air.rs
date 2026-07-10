//! The air module contains the AIR constraints for the Poseidon2 skinny chip.

use std::{array, borrow::Borrow};

use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;

use zkm_hypercube::air::{AirLookup, LookupScope};
use zkm_hypercube::lookup::LookupKind;

use crate::{builder::ZKMRecursionAirBuilder, chips::poseidon2_skinny::columns::Poseidon2};

use super::{
    columns::{preprocessed::Poseidon2PreprocessedCols, NUM_POSEIDON2_COLS},
    external_linear_layer, internal_linear_layer, Poseidon2SkinnyChip, NUM_INTERNAL_ROUNDS,
    NUM_ROUND_CONSTANTS, WIDTH,
};

impl<F, const DEGREE: usize> BaseAir<F> for Poseidon2SkinnyChip<DEGREE> {
    fn width(&self) -> usize {
        // We only support machines with degree 9.
        assert!(DEGREE >= 9);
        NUM_POSEIDON2_COLS
    }
}

impl<AB, const DEGREE: usize> Air<AB> for Poseidon2SkinnyChip<DEGREE>
where
    AB: ZKMRecursionAirBuilder + PairBuilder,
    AB::Var: 'static,
{
    fn eval(&self, builder: &mut AB) {
        // We only support machines with degree 9.
        assert!(DEGREE >= 9);

        let main = builder.main();
        let local_row = main.row_slice(0);
        let local_row: &Poseidon2<_> = (*local_row).borrow();
        let prepr = builder.preprocessed();
        let prep_local = prepr.row_slice(0);
        let prep_local: &Poseidon2PreprocessedCols<_> = (*prep_local).borrow();

        // Dummy constraints to normalize to DEGREE.
        let lhs = (0..DEGREE).map(|_| local_row.state_var[0].into()).product::<AB::Expr>();
        let rhs = (0..DEGREE).map(|_| local_row.state_var[0].into()).product::<AB::Expr>();
        builder.assert_eq(lhs, rhs);

        // Enforce the memory argument for the state words.
        (0..WIDTH).for_each(|i| {
            builder.send_single(
                prep_local.memory_preprocessed[i].addr,
                local_row.state_var[i],
                prep_local.memory_preprocessed[i].mult,
            )
        });

        // Chain the round state across rows of the same permutation invocation: every row but
        // the output row (the round-counter flags being all-zero doesn't distinguish an output
        // row from padding, hence the separate `is_real`) sends its `computed_next_state`
        // forward to `index + 1`; every row but the input row receives its predecessor's
        // `computed_next_state` as its own `state_var` from `index`. This replaces a physical
        // `next` row reference, which the zerocheck prover's single-row constraint-eval contexts
        // don't support.
        let round_counters = &prep_local.round_counters_preprocessed;
        let not_output: AB::Expr = round_counters.is_input_round.into()
            + round_counters.is_external_round.into()
            + round_counters.is_internal_round.into();
        let not_input: AB::Expr = prep_local.is_real.into() - round_counters.is_input_round.into();
        builder.send(
            AirLookup::new(
                std::iter::once(prep_local.index.into() + AB::Expr::one())
                    .chain(local_row.computed_next_state.iter().map(|x| (*x).into()))
                    .collect(),
                not_output,
                LookupKind::Poseidon2SkinnyState,
            ),
            LookupScope::Local,
        );
        builder.receive(
            AirLookup::new(
                std::iter::once(prep_local.index.into())
                    .chain(local_row.state_var.iter().map(|x| (*x).into()))
                    .collect(),
                not_input,
                LookupKind::Poseidon2SkinnyState,
            ),
            LookupScope::Local,
        );

        self.eval_input_round(builder, local_row, prep_local);

        self.eval_external_round(builder, local_row, prep_local);

        self.eval_internal_rounds(
            builder,
            local_row,
            prep_local.round_counters_preprocessed.round_constants,
            prep_local.round_counters_preprocessed.is_internal_round,
        );
    }
}

impl<const DEGREE: usize> Poseidon2SkinnyChip<DEGREE> {
    fn eval_input_round<AB: ZKMRecursionAirBuilder>(
        &self,
        builder: &mut AB,
        local_row: &Poseidon2<AB::Var>,
        prep_local: &Poseidon2PreprocessedCols<AB::Var>,
    ) {
        let mut state: [AB::Expr; WIDTH] = array::from_fn(|i| local_row.state_var[i].into());

        // Apply the linear layer.
        external_linear_layer(&mut state);

        for i in 0..WIDTH {
            builder
                .when(prep_local.round_counters_preprocessed.is_input_round)
                .assert_eq(local_row.computed_next_state[i], state[i].clone());
        }
    }

    fn eval_external_round<AB: ZKMRecursionAirBuilder>(
        &self,
        builder: &mut AB,
        local_row: &Poseidon2<AB::Var>,
        prep_local: &Poseidon2PreprocessedCols<AB::Var>,
    ) {
        let local_state = local_row.state_var;

        // Add the round constants.
        let add_rc: [AB::Expr; WIDTH] = core::array::from_fn(|i| {
            local_state[i].into() + prep_local.round_counters_preprocessed.round_constants[i]
        });

        // Apply the sboxes.
        // See `populate_external_round` for why we don't have columns for the sbox output here.
        let mut sbox_deg_3: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::zero());
        for i in 0..WIDTH {
            sbox_deg_3[i] = add_rc[i].clone() * add_rc[i].clone() * add_rc[i].clone();
            // sbox_deg_7[i] = sbox_deg_3.clone() * sbox_deg_3.clone() * add_rc[i].clone();
        }

        // Apply the linear layer.
        let mut state = sbox_deg_3;
        external_linear_layer(&mut state);

        for i in 0..WIDTH {
            builder
                .when(prep_local.round_counters_preprocessed.is_external_round)
                .assert_eq(local_row.computed_next_state[i], state[i].clone());
        }
    }

    fn eval_internal_rounds<AB: ZKMRecursionAirBuilder>(
        &self,
        builder: &mut AB,
        local_row: &Poseidon2<AB::Var>,
        round_constants: [AB::Var; NUM_ROUND_CONSTANTS],
        is_internal_row: AB::Var,
    ) {
        let local_state = local_row.state_var;

        let s0 = local_row.internal_rounds_s0;
        let mut state: [AB::Expr; WIDTH] = core::array::from_fn(|i| local_state[i].into());
        for r in 0..NUM_INTERNAL_ROUNDS {
            // Add the round constant.
            let add_rc =
                if r == 0 { state[0].clone() } else { s0[r - 1].into() } + round_constants[r];

            let sbox_deg_3 = add_rc.clone() * add_rc.clone() * add_rc.clone();
            // // See `populate_internal_rounds` for why we don't have columns for the sbox output
            // // here.
            // let sbox_deg_7 = sbox_deg_3.clone() * sbox_deg_3.clone() * add_rc.clone();

            // Apply the linear layer.
            // See `populate_internal_rounds` for why we don't have columns for the new state here.
            state[0] = sbox_deg_3.clone();
            internal_linear_layer(&mut state);

            if r < NUM_INTERNAL_ROUNDS - 1 {
                builder.when(is_internal_row).assert_eq(s0[r], state[0].clone());
            }
        }

        for i in 0..WIDTH {
            builder.when(is_internal_row).assert_eq(local_row.computed_next_state[i], state[i].clone())
        }
    }
}
