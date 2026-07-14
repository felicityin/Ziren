use rayon::prelude::*;

use slop_algebra::{
    interpolate_univariate_polynomial, FieldAlgebra, ExtensionField, Field, UnivariatePolynomial,
};
use slop_alloc::CpuBackend;
use slop_multilinear::Mle;

use crate::{
    backend::{ComponentPolyEvalBackend, SumcheckPolyBackend},
    SumCheckPolyFirstRoundBackend, SumcheckPolyBase,
};

impl<F: FieldAlgebra> SumcheckPolyBase for Mle<F, CpuBackend> {
    #[inline]
    fn num_variables(&self) -> u32 {
        self.num_variables()
    }
}

impl<F, EF> ComponentPolyEvalBackend<Mle<F, CpuBackend>, EF> for CpuBackend
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn get_component_poly_evals(poly: &Mle<F, CpuBackend>) -> Vec<EF> {
        let eval: F = *poly.guts()[[0, 0]];
        vec![EF::from_base(eval)]
    }
}

impl<F> SumcheckPolyBackend<Mle<F, CpuBackend>, F> for CpuBackend
where
    F: Field,
{
    fn fix_last_variable(poly: Mle<F, CpuBackend>, alpha: F) -> Mle<F, CpuBackend> {
        poly.fix_last_variable(alpha)
    }

    fn sum_as_poly_in_last_variable(
        poly: &Mle<F, CpuBackend>,
        claim: Option<F>,
    ) -> UnivariatePolynomial<F> {
        // If the polynomial is 0-variate, the length of its guts is not divisible by 2, so we need
        // to handle this case separately.
        if poly.num_variables() == 0 {
            return UnivariatePolynomial::new(vec![*poly.guts()[[0, 0]], F::zero()]);
        }

        let claim = claim.expect("expected a claim for a non-zero-variate polynomial");

        assert_eq!(poly.num_polynomials(), 1);

        let eval_zero = poly.guts().as_slice().par_iter().step_by(2).copied().sum::<F>();
        let eval_one = claim - eval_zero;

        interpolate_univariate_polynomial(&[F::zero(), F::one()], &[eval_zero, eval_one])
    }
}

impl<F, EF> SumCheckPolyFirstRoundBackend<Mle<F, CpuBackend>, EF> for CpuBackend
where
    F: Field,
    EF: ExtensionField<F>,
{
    type NextRoundPoly = Mle<EF, CpuBackend>;

    fn fix_t_variables(poly: Mle<F, CpuBackend>, alpha: EF, t: usize) -> Self::NextRoundPoly {
        assert_eq!(t, 1);
        poly.fix_last_variable(alpha)
    }

    fn sum_as_poly_in_last_t_variables(
        poly: &Mle<F, CpuBackend>,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert_eq!(t, 1);
        assert!(poly.num_variables() > 0);
        let claim = claim.expect("expected a claim for a non-zero-variate polynomial");

        assert_eq!(poly.num_polynomials(), 1);

        let eval_zero =
            EF::from_base(poly.guts().as_slice().par_iter().step_by(2).copied().sum::<F>());
        let eval_one = claim - eval_zero;

        interpolate_univariate_polynomial(&[EF::zero(), EF::one()], &[eval_zero, eval_one])
    }
}
#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use slop_algebra::{extension::BinomialExtensionField, FieldExtensionAlgebra};
    use slop_challenger::DuplexChallenger;
    use slop_koala_bear::{
        my_kb_16_perm, KoalaPerm,
        KoalaBear,
    };

    use crate::{partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation};

    use super::*;

    #[test]
    fn test_single_mle_sumcheck() {
        let mut rng = thread_rng();

        let mle = Mle::<KoalaBear, CpuBackend>::rand(&mut rng, 1, 10);
        type EF = BinomialExtensionField<KoalaBear, 4>;

        let default_perm = my_kb_16_perm();
        let mut challenger =
            DuplexChallenger::<KoalaBear, KoalaPerm, 16, 8>::new(default_perm.clone());

        let claim = EF::from_base(mle.guts().as_slice().par_iter().copied().sum::<KoalaBear>());

        let (sumcheck_proof, _) = reduce_sumcheck_to_evaluation::<KoalaBear, EF, _>(
            vec![mle.clone()],
            &mut challenger,
            vec![claim],
            1,
            EF::one(),
        );

        // Verify the evaluation claim.
        let (point, eval_claim) = sumcheck_proof.point_and_eval.clone();
        let evaluation = mle.eval_at(&point)[0];
        assert_eq!(evaluation, eval_claim);

        // Verify the proof.
        let mut challenger = DuplexChallenger::<KoalaBear, KoalaPerm, 16, 8>::new(default_perm);
        partially_verify_sumcheck_proof(&sumcheck_proof, &mut challenger, 10, 1).unwrap()
    }
}
