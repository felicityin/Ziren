use std::marker::PhantomData;

use slop_jagged::{
    BranchingProgram, JaggedLittlePolynomialVerifierParams, JaggedSumcheckEvalProof,
};
use slop_multilinear::{Mle, Point};
use zkm_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, Felt, SymbolicExt, SymbolicFelt},
};

use crate::{
    challenger::{CanObserveVariable, FieldChallengerVariable},
    sumcheck::verify_sumcheck,
    symbolic::IntoSymbolic,
    CircuitConfig,
};

impl<C: CircuitConfig> IntoSymbolic<C> for JaggedLittlePolynomialVerifierParams<Felt<C::F>> {
    type Output = JaggedLittlePolynomialVerifierParams<SymbolicFelt<C::F>>;

    fn as_symbolic(&self) -> Self::Output {
        JaggedLittlePolynomialVerifierParams {
            col_prefix_sums: self
                .col_prefix_sums
                .iter()
                .map(|x| <Point<Felt<C::F>> as IntoSymbolic<C>>::as_symbolic(x))
                .collect::<Vec<_>>(),
        }
    }
}

pub trait RecursiveJaggedEvalConfig<C: CircuitConfig, Challenger>: Sized {
    type JaggedEvalProof;

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn jagged_evaluation(
        &self,
        builder: &mut Builder<C>,
        params: &JaggedLittlePolynomialVerifierParams<Felt<C::F>>,
        z_row: Point<Ext<C::F, C::EF>>,
        z_col: Point<Ext<C::F, C::EF>>,
        z_trace: Point<Ext<C::F, C::EF>>,
        proof: &Self::JaggedEvalProof,
        challenger: &mut Challenger,
    ) -> (SymbolicExt<C::F, C::EF>, Vec<Felt<C::F>>);
}

#[derive(Debug, Clone)]
pub struct RecursiveJaggedEvalSumcheckConfig<Challenger>(pub PhantomData<Challenger>);

impl<C: CircuitConfig<F = p3_koala_bear::KoalaBear>, Challenger>
    RecursiveJaggedEvalConfig<C, Challenger> for RecursiveJaggedEvalSumcheckConfig<Challenger>
where
    Challenger: FieldChallengerVariable<C, C::Bit> + CanObserveVariable<C, Felt<C::F>>,
{
    type JaggedEvalProof = JaggedSumcheckEvalProof<Ext<C::F, C::EF>>;

    fn jagged_evaluation(
        &self,
        builder: &mut Builder<C>,
        params: &JaggedLittlePolynomialVerifierParams<Felt<C::F>>,
        z_row: Point<Ext<C::F, C::EF>>,
        z_col: Point<Ext<C::F, C::EF>>,
        z_trace: Point<Ext<C::F, C::EF>>,
        proof: &Self::JaggedEvalProof,
        challenger: &mut Challenger,
    ) -> (SymbolicExt<C::F, C::EF>, Vec<Felt<C::F>>) {
        let z_row = <Point<Ext<C::F, C::EF>> as IntoSymbolic<C>>::as_symbolic(&z_row);
        let z_col = <Point<Ext<C::F, C::EF>> as IntoSymbolic<C>>::as_symbolic(&z_col);
        let z_trace = <Point<Ext<C::F, C::EF>> as IntoSymbolic<C>>::as_symbolic(&z_trace);

        let JaggedSumcheckEvalProof { partial_sumcheck_proof } = proof;
        // Calculate the partial lagrange from z_col point.
        let z_col_partial_lagrange = Mle::blocking_partial_lagrange(&z_col);
        let z_col_partial_lagrange = z_col_partial_lagrange.guts().as_slice();

        // Calculate the jagged eval from the branching program eval claims.
        let jagged_eval = partial_sumcheck_proof.claimed_sum;

        challenger.observe_ext_element(builder, jagged_eval);

        builder.assert_ext_eq(jagged_eval, partial_sumcheck_proof.claimed_sum);

        // Verify the jagged eval proof.
        builder.cycle_tracker_v2_enter("jagged eval - verify sumcheck".to_string());
        verify_sumcheck(builder, challenger, partial_sumcheck_proof);
        builder.cycle_tracker_v2_exit();
        let proof_point = <Point<Ext<C::F, C::EF>> as IntoSymbolic<C>>::as_symbolic(
            &partial_sumcheck_proof.point_and_eval.0,
        );
        let (first_half_z_index, second_half_z_index) =
            proof_point.split_at(proof_point.dimension() / 2);
        assert!(first_half_z_index.len() == second_half_z_index.len());

        // Compute the jagged eval sc expected eval and assert it matches the proof's eval.
        let current_column_prefix_sums = params.col_prefix_sums.iter();
        let next_column_prefix_sums = params.col_prefix_sums.iter().skip(1);
        let mut prefix_sum_felts = Vec::new();
        builder.cycle_tracker_v2_enter("jagged eval - calculate expected eval".to_string());
        let mut jagged_eval_sc_expected_eval = current_column_prefix_sums
            .zip(next_column_prefix_sums)
            .zip(z_col_partial_lagrange.iter())
            .map(|((current_column_prefix_sum, next_column_prefix_sum), z_col_eq_val)| {
                assert!(current_column_prefix_sum.dimension() <= 30);
                assert!(next_column_prefix_sum.dimension() <= 30);

                let mut merged_prefix_sum = current_column_prefix_sum.clone();
                merged_prefix_sum.extend(next_column_prefix_sum);

                let (full_lagrange_eval, felt) = C::prefix_sum_checks(
                    builder,
                    merged_prefix_sum.to_vec(),
                    partial_sumcheck_proof.point_and_eval.0.to_vec(),
                );
                prefix_sum_felts.push(felt);
                *z_col_eq_val * full_lagrange_eval
            })
            .sum::<SymbolicExt<C::F, C::EF>>();
        builder.cycle_tracker_v2_exit();
        let branching_program = BranchingProgram::new(z_row.clone(), z_trace.clone());
        jagged_eval_sc_expected_eval *=
            branching_program.eval(&first_half_z_index, &second_half_z_index);

        builder
            .assert_ext_eq(jagged_eval_sc_expected_eval, partial_sumcheck_proof.point_and_eval.1);

        (jagged_eval.into(), prefix_sum_felts)
    }
}
