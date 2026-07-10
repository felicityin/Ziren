use super::RecursiveMultilinearPcsVerifier;
use crate::{challenger::FieldChallengerVariable, sumcheck::evaluate_mle_ext};
use slop_commit::Rounds;
use slop_multilinear::{Mle, MleEval, Point};
use zkm_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, SymbolicExt},
};

#[derive(Clone)]
pub struct RecursiveStackedPcsVerifier<P> {
    pub recursive_pcs_verifier: P,
    pub log_stacking_height: u32,
}

pub struct RecursiveStackedPcsProof<PcsProof, F, EF> {
    pub batch_evaluations: Rounds<MleEval<Ext<F, EF>>>,
    pub pcs_proof: PcsProof,
}

impl<P: RecursiveMultilinearPcsVerifier> RecursiveStackedPcsVerifier<P>
where
    P::Circuit: zkm_recursion_compiler::ir::Config<F = p3_koala_bear::KoalaBear>,
{
    pub const fn new(recursive_pcs_verifier: P, log_stacking_height: u32) -> Self {
        Self { recursive_pcs_verifier, log_stacking_height }
    }

    pub fn verify_untrusted_evaluation(
        &self,
        builder: &mut Builder<P::Circuit>,
        commitments: &[P::Commitment],
        point: &Point<
            Ext<
                <P::Circuit as zkm_recursion_compiler::ir::Config>::F,
                <P::Circuit as zkm_recursion_compiler::ir::Config>::EF,
            >,
        >,
        proof: &RecursiveStackedPcsProof<
            P::Proof,
            <P::Circuit as zkm_recursion_compiler::ir::Config>::F,
            <P::Circuit as zkm_recursion_compiler::ir::Config>::EF,
        >,
        evaluation_claim: SymbolicExt<
            <P::Circuit as zkm_recursion_compiler::ir::Config>::F,
            <P::Circuit as zkm_recursion_compiler::ir::Config>::EF,
        >,
        challenger: &mut P::Challenger,
    ) {
        let claim_ext: Ext<_, _> = builder.eval(evaluation_claim);
        challenger.observe_ext_element(builder, claim_ext);
        let (batch_point, stack_point) =
            point.split_at(point.dimension() - self.log_stacking_height as usize);
        let batch_evaluations =
            proof.batch_evaluations.iter().flatten().cloned().collect::<Mle<_>>();

        builder.cycle_tracker_v2_enter("basefold - evaluate_mle_ext".to_string());
        let expected_evaluation = evaluate_mle_ext(builder, batch_evaluations, batch_point)[0];
        builder.assert_ext_eq(claim_ext, expected_evaluation);
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("basefold - verify_untrusted_evaluations".to_string());
        self.recursive_pcs_verifier.verify_untrusted_evaluations(
            builder,
            commitments,
            stack_point,
            &proof.batch_evaluations,
            &proof.pcs_proof,
            challenger,
        );
        builder.cycle_tracker_v2_exit();
    }
}
