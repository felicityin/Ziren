use std::iter::repeat_n;

use itertools::{izip, Itertools};
use p3_field::{ExtensionField, Field, FieldAlgebra};
use slop_jagged::{JaggedLittlePolynomialVerifierParams, JaggedSumcheckEvalProof};
use slop_multilinear::{Mle, MleEval, Point};
use slop_sumcheck::PartialSumcheckProof;

use crate::{
    basefold::{
        stacked::{RecursiveStackedPcsProof, RecursiveStackedPcsVerifier},
        RecursiveBasefoldProof, RecursiveBasefoldVerifier,
    },
    challenger::{CanObserveVariable, FieldChallengerVariable},
    hash::FieldHasherVariable,
    sumcheck::{evaluate_mle_ext, verify_sumcheck},
    CircuitConfig,
};
use zkm_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, Felt, SymbolicExt, SymbolicFelt},
};

use super::jagged_eval::{RecursiveJaggedEvalConfig, RecursiveJaggedEvalSumcheckConfig};

pub struct JaggedPcsProofVariable<F: Field, EF: ExtensionField<F>, Proof, Digest> {
    pub params: JaggedLittlePolynomialVerifierParams<Felt<F>>,
    pub sumcheck_proof: PartialSumcheckProof<Ext<F, EF>>,
    pub jagged_eval_proof: JaggedSumcheckEvalProof<Ext<F, EF>>,
    pub pcs_proof: RecursiveStackedPcsProof<Proof, F, EF>,
    pub column_counts: Vec<Vec<usize>>,
    pub row_counts: Vec<Vec<Felt<F>>>,
    pub original_commitments: Vec<Digest>,
    pub expected_eval: Ext<F, EF>,
}

#[derive(Clone)]
pub struct RecursiveJaggedPcsVerifier<C: CircuitConfig, HV: FieldHasherVariable<C>, Challenger> {
    pub stacked_pcs_verifier:
        RecursiveStackedPcsVerifier<RecursiveBasefoldVerifier<C, HV, Challenger>>,
    pub max_log_row_count: usize,
    pub jagged_evaluator: RecursiveJaggedEvalSumcheckConfig<Challenger>,
}

impl<C: CircuitConfig<F = p3_koala_bear::KoalaBear>, HV: FieldHasherVariable<C>, Challenger>
    RecursiveJaggedPcsVerifier<C, HV, Challenger>
where
    Challenger: FieldChallengerVariable<C, C::Bit> + CanObserveVariable<C, HV::DigestVariable>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn verify_trusted_evaluations(
        &self,
        builder: &mut Builder<C>,
        commitments: &[HV::DigestVariable],
        point: Point<Ext<C::F, C::EF>>,
        evaluation_claims: &[MleEval<Ext<C::F, C::EF>>],
        proof: &JaggedPcsProofVariable<
            C::F,
            C::EF,
            RecursiveBasefoldProof<C, HV>,
            HV::DigestVariable,
        >,
        insertion_points: &[usize],
        challenger: &mut Challenger,
    ) -> Vec<Felt<C::F>> {
        let JaggedPcsProofVariable {
            pcs_proof,
            sumcheck_proof,
            jagged_eval_proof,
            params,
            column_counts,
            original_commitments,
            expected_eval,
            ..
        } = proof;
        let num_col_variables = (params.col_prefix_sums.len() - 1).next_power_of_two().ilog2();

        let z_col =
            (0..num_col_variables).map(|_| challenger.sample_ext(builder)).collect::<Point<_>>();

        let z_row = point;

        // Collect the claims for the different polynomials.
        let mut column_claims = evaluation_claims.iter().flatten().copied().collect::<Vec<_>>();

        let added_columns: Vec<usize> =
            column_counts.iter().map(|cc| cc[cc.len() - 2] + 1).collect();
        // For each commit, the PCS needs a commitment to a vector of length a multiple of
        // 1 << self.pcs.log_stacking_height, and this is achieved by adding a single column of
        // zeroes as the last matrix of the commitment. We insert these "artificial" zeroes
        // into the evaluation claims.
        let zero_ext: Ext<C::F, C::EF> = builder.constant(C::EF::ZERO);
        for (insertion_point, num_added_columns) in
            insertion_points.iter().rev().zip(added_columns.iter().rev())
        {
            for _ in 0..*num_added_columns {
                column_claims.insert(*insertion_point, zero_ext);
            }
        }

        for (round_column_counts, round_row_counts, modified_commitment, original_commitment) in izip!(
            column_counts.iter(),
            proof.row_counts.iter(),
            commitments.iter(),
            original_commitments.iter()
        ) {
            let mut felts_vec: Vec<Felt<_>> =
                vec![builder.eval(C::F::from_canonical_usize(round_column_counts.len()))];
            for &count in round_row_counts {
                felts_vec.push(builder.eval(count));
            }

            for &count in round_column_counts {
                felts_vec.push(builder.eval(C::F::from_canonical_usize(count)));
            }
            let hash = HV::hash(builder, &felts_vec);
            let expected_commitment = HV::compress(builder, [*original_commitment, hash]);

            HV::assert_digest_eq(builder, expected_commitment, *modified_commitment);
        }

        // Pad the column claims to the next power of two.
        column_claims.resize(column_claims.len().next_power_of_two(), zero_ext);

        let column_mle = Mle::from(column_claims);
        let sumcheck_claim: Ext<C::F, C::EF> =
            evaluate_mle_ext(builder, column_mle, z_col.clone())[0];

        builder.assert_ext_eq(sumcheck_claim, sumcheck_proof.claimed_sum);

        builder.cycle_tracker_v2_enter("jagged - verify sumcheck".to_string());
        verify_sumcheck(builder, challenger, sumcheck_proof);
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("jagged - jagged-eval".to_string());
        let (jagged_eval, prefix_sum_felts) = self.jagged_evaluator.jagged_evaluation(
            builder,
            params,
            z_row,
            z_col,
            sumcheck_proof.point_and_eval.0.clone(),
            jagged_eval_proof,
            challenger,
        );
        builder.cycle_tracker_v2_exit();

        // Check the prefix_sum_felts against the row counts.
        let repeated_flattened_row_counts: Vec<Felt<C::F>> = proof
            .row_counts
            .iter()
            .flatten()
            .zip_eq(column_counts.iter().flatten())
            .flat_map(|(row, col)| repeat_n(*row, *col))
            .collect();

        let mut acc: Felt<_> = builder.constant(C::F::ZERO);

        for (row_count, expected) in
            repeated_flattened_row_counts.iter().zip_eq(prefix_sum_felts.iter())
        {
            builder.assert_felt_eq(acc, *expected);
            acc = builder.eval(acc + *row_count)
        }
        let mut final_area = SymbolicFelt::<C::F>::ZERO;
        let two: Felt<_> = builder.constant(C::F::TWO);
        for bit in proof.params.col_prefix_sums.iter().last().unwrap().iter() {
            final_area = *bit + two * final_area;
        }
        builder.assert_felt_eq(acc, final_area);

        // Compute the expected evaluation of the dense trace polynomial.
        builder.assert_ext_eq(jagged_eval * *expected_eval, sumcheck_proof.point_and_eval.1);

        // Verify the evaluation proof.
        let evaluation_point = sumcheck_proof.point_and_eval.0.clone();
        self.stacked_pcs_verifier.verify_untrusted_evaluation(
            builder,
            original_commitments,
            &evaluation_point,
            pcs_proof,
            SymbolicExt::from(*expected_eval),
            challenger,
        );
        prefix_sum_felts
    }
}

impl<C: CircuitConfig<F = p3_koala_bear::KoalaBear>, HV: FieldHasherVariable<C>, Challenger>
    RecursiveJaggedPcsVerifier<C, HV, Challenger>
where
    Challenger: FieldChallengerVariable<C, C::Bit> + CanObserveVariable<C, HV::DigestVariable>,
{
    /// Builds the circuit-side verifier config matching a native
    /// `zkm_hypercube::verifier::ShardVerifier::from_basefold_parameters` call with the same
    /// `fri_config`/`log_stacking_height`/`max_log_row_count`.
    pub fn from_basefold_parameters(
        fri_config: slop_basefold::FriConfig<C::F>,
        log_stacking_height: u32,
        max_log_row_count: usize,
    ) -> Self {
        let basefold_verifier = RecursiveBasefoldVerifier {
            fri_config,
            tcs: crate::basefold::tcs::RecursiveMerkleTreeTcs(std::marker::PhantomData),
            _marker: std::marker::PhantomData,
        };
        Self {
            stacked_pcs_verifier: RecursiveStackedPcsVerifier::new(
                basefold_verifier,
                log_stacking_height,
            ),
            max_log_row_count,
            jagged_evaluator: RecursiveJaggedEvalSumcheckConfig(std::marker::PhantomData),
        }
    }
}

pub struct RecursiveMachineJaggedPcsVerifier<
    'a,
    C: CircuitConfig,
    HV: FieldHasherVariable<C>,
    Challenger,
> {
    pub jagged_pcs_verifier: &'a RecursiveJaggedPcsVerifier<C, HV, Challenger>,
    pub column_counts_by_round: Vec<Vec<usize>>,
}

impl<
        'a,
        C: CircuitConfig<F = p3_koala_bear::KoalaBear>,
        HV: FieldHasherVariable<C>,
        Challenger,
    > RecursiveMachineJaggedPcsVerifier<'a, C, HV, Challenger>
where
    Challenger: FieldChallengerVariable<C, C::Bit> + CanObserveVariable<C, HV::DigestVariable>,
{
    pub fn new(
        jagged_pcs_verifier: &'a RecursiveJaggedPcsVerifier<C, HV, Challenger>,
        column_counts_by_round: Vec<Vec<usize>>,
    ) -> Self {
        Self { jagged_pcs_verifier, column_counts_by_round }
    }

    pub fn verify_trusted_evaluations(
        &self,
        builder: &mut Builder<C>,
        commitments: &[HV::DigestVariable],
        point: Point<Ext<C::F, C::EF>>,
        evaluation_claims: &[MleEval<Ext<C::F, C::EF>>],
        proof: &JaggedPcsProofVariable<
            C::F,
            C::EF,
            RecursiveBasefoldProof<C, HV>,
            HV::DigestVariable,
        >,
        challenger: &mut Challenger,
    ) -> Vec<Felt<C::F>> {
        let insertion_points = self
            .column_counts_by_round
            .iter()
            .scan(0, |state, y| {
                *state += y.iter().sum::<usize>();
                Some(*state)
            })
            .collect::<Vec<_>>();

        self.jagged_pcs_verifier.verify_trusted_evaluations(
            builder,
            commitments,
            point,
            evaluation_claims,
            proof,
            &insertion_points,
            challenger,
        )
    }
}
