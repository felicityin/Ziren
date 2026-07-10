pub mod witness;

use crate::{
    challenger::{CanObserveVariable, FieldChallengerVariable},
    symbolic::IntoSymbolic,
    CircuitConfig,
};
use p3_field::FieldAlgebra;
use slop_algebra::UnivariatePolynomial;
use slop_alloc::{buffer, Buffer};
use slop_multilinear::{partial_lagrange_blocking, Mle, MleEval, Point};
use slop_sumcheck::PartialSumcheckProof;
use slop_tensor::{Dimensions, Tensor};
use zkm_recursion_compiler::{
    ir::Felt,
    prelude::{Builder, Ext, SymbolicExt},
};

pub fn evaluate_mle_ext_batch<C: CircuitConfig>(
    builder: &mut Builder<C>,
    mles: Vec<Mle<Ext<C::F, C::EF>>>,
    point: Point<Ext<C::F, C::EF>>,
) -> Vec<MleEval<Ext<C::F, C::EF>>> {
    let point_symbolic = <Point<Ext<C::F, C::EF>> as IntoSymbolic<C>>::as_symbolic(&point);
    let partial_lagrange = partial_lagrange_blocking(&point_symbolic);
    let mut result = Vec::new();
    for mle in &mles {
        let mle = mle.guts();
        let mut sizes = mle.sizes().to_vec();
        sizes.remove(0);
        let dimensions = Dimensions::try_from(sizes).unwrap();
        let mut dst = Tensor { storage: buffer![], dimensions };
        let total_len = dst.total_len();
        let dot_products = mle
            .as_buffer()
            .chunks_exact(mle.strides()[0])
            .zip(partial_lagrange.as_buffer().iter())
            .map(|(chunk, scalar)| chunk.iter().map(|a| *scalar * *a).collect())
            .fold(
                vec![SymbolicExt::<C::F, C::EF>::ZERO; total_len],
                |mut a, b: Vec<SymbolicExt<_, _>>| {
                    a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b);
                    a
                },
            );
        let dot_products = dot_products.into_iter().map(|x| builder.eval(x)).collect::<Buffer<_>>();
        dst.storage = dot_products;
        result.push(MleEval::new(dst));
    }

    result
}

pub fn evaluate_mle_ext<C: CircuitConfig>(
    builder: &mut Builder<C>,
    mle: Mle<Ext<C::F, C::EF>>,
    point: Point<Ext<C::F, C::EF>>,
) -> MleEval<Ext<C::F, C::EF>> {
    evaluate_mle_ext_batch(builder, vec![mle], point).pop().unwrap()
}

pub fn verify_sumcheck<C: CircuitConfig, Bit, Challenger>(
    builder: &mut Builder<C>,
    challenger: &mut Challenger,
    proof: &PartialSumcheckProof<Ext<C::F, C::EF>>,
) where
    Challenger: FieldChallengerVariable<C, Bit> + CanObserveVariable<C, Felt<C::F>>,
{
    let num_variables = proof.univariate_polys.len();
    let mut alpha_point: Point<SymbolicExt<C::F, C::EF>> = Point::default();

    assert_eq!(num_variables, proof.point_and_eval.0.dimension());

    let first_poly = proof.univariate_polys[0].clone();
    let first_poly_symbolic: UnivariatePolynomial<SymbolicExt<C::F, C::EF>> =
        UnivariatePolynomial {
            coefficients: first_poly
                .coefficients
                .clone()
                .into_iter()
                .map(|c| c.into())
                .collect::<Vec<_>>(),
        };
    builder.assert_ext_eq(first_poly_symbolic.eval_one_plus_eval_zero(), proof.claimed_sum);

    let coeffs: Vec<Felt<C::F>> =
        first_poly.coefficients.iter().flat_map(|x| C::ext2felt(builder, *x)).collect::<Vec<_>>();

    challenger.observe_slice(builder, coeffs);

    let mut previous_poly = first_poly_symbolic;
    for poly in proof.univariate_polys.iter().skip(1) {
        let alpha = challenger.sample_ext(builder);
        alpha_point.add_dimension(alpha.into());
        let poly_symbolic: UnivariatePolynomial<SymbolicExt<C::F, C::EF>> = UnivariatePolynomial {
            coefficients: poly
                .coefficients
                .clone()
                .into_iter()
                .map(|c| c.into())
                .collect::<Vec<_>>(),
        };
        let expected_eval = previous_poly.eval_at_point(alpha.into());
        builder.assert_ext_eq(expected_eval, poly_symbolic.eval_one_plus_eval_zero());

        let coeffs: Vec<Felt<C::F>> =
            poly.coefficients.iter().flat_map(|x| C::ext2felt(builder, *x)).collect::<Vec<_>>();
        challenger.observe_slice(builder, coeffs);
        previous_poly = poly_symbolic;
    }

    let alpha = challenger.sample_ext(builder);
    alpha_point.add_dimension(alpha.into());

    alpha_point.iter().zip(proof.point_and_eval.0.iter()).for_each(|(d, p)| {
        builder.assert_ext_eq(*d, *p);
    });

    builder.assert_ext_eq(previous_poly.eval_at_point(alpha.into()), proof.point_and_eval.1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        challenger::DuplexChallengerVariable,
        witness::{WitnessBlock, Witnessable},
    };
    use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
    use rand::{rngs::OsRng, thread_rng};
    use slop_multilinear::{full_geq, Mle};
    use slop_sumcheck::reduce_sumcheck_to_evaluation;
    use zkm_recursion_compiler::{
        circuit::{AsmBuilder, AsmConfig, CircuitV2Builder},
        ir::{Ext, SymbolicExt},
    };
    use zkm_stark::{inner_perm, InnerChallenge, InnerChallenger, InnerVal};

    use crate::utils::tests::run_test_recursion;

    type F = InnerVal;
    type EF = InnerChallenge;
    type C = AsmConfig<F, EF>;

    #[test]
    fn test_sumcheck() {
        let mut rng = thread_rng();

        let mle = Mle::<F>::rand(&mut rng, 1, 10);

        let mut challenger = InnerChallenger::new(inner_perm());

        let claim = EF::from_base(mle.guts().as_slice().iter().copied().sum::<F>());

        let (sumcheck_proof, _) = reduce_sumcheck_to_evaluation::<F, EF, _>(
            vec![mle.clone()],
            &mut challenger,
            vec![claim],
            1,
            EF::ONE,
        );

        let (point, eval_claim) = sumcheck_proof.point_and_eval.clone();
        let evaluation = mle.eval_at(&point)[0];
        assert_eq!(evaluation, eval_claim);

        let mut builder = AsmBuilder::<F, EF>::default();

        let sumcheck_proof_variable = sumcheck_proof.read(&mut builder);

        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);
        verify_sumcheck(&mut builder, &mut challenger_variable, &sumcheck_proof_variable);

        let mut witness_stream = Vec::<WitnessBlock<C>>::new();
        Witnessable::<C>::write(&sumcheck_proof, &mut witness_stream);

        run_test_recursion(builder.into_operations(), witness_stream);
    }

    #[test]
    #[should_panic]
    fn test_sumcheck_failure() {
        let mut rng = thread_rng();

        let mle = Mle::<F>::rand(&mut rng, 1, 10);

        let mut challenger = InnerChallenger::new(inner_perm());

        let claim = EF::from_base(mle.guts().as_slice().iter().copied().sum::<F>());

        let (mut sumcheck_proof, _) = reduce_sumcheck_to_evaluation::<F, EF, _>(
            vec![mle.clone()],
            &mut challenger,
            vec![claim],
            1,
            EF::ONE,
        );

        let (point, eval_claim) = sumcheck_proof.point_and_eval.clone();
        let evaluation = mle.eval_at(&point)[0];
        assert_eq!(evaluation, eval_claim);

        // modify the first polynomial to make the sumcheck fail
        sumcheck_proof.univariate_polys[0].coefficients[0] = EF::ONE;

        let mut builder = AsmBuilder::<F, EF>::default();

        let sumcheck_proof_variable = sumcheck_proof.read(&mut builder);

        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);
        verify_sumcheck(&mut builder, &mut challenger_variable, &sumcheck_proof_variable);

        let mut witness_stream = Vec::<WitnessBlock<C>>::new();
        Witnessable::<C>::write(&sumcheck_proof, &mut witness_stream);

        run_test_recursion(builder.into_operations(), witness_stream);
    }

    #[test]
    fn test_eval_at_point() {
        let mut rng = OsRng;
        let mut builder = AsmBuilder::<F, EF>::default();
        let exts = builder.hint_exts_v2(3);
        let point = builder.hint_ext_v2();
        let univariate_poly =
            UnivariatePolynomial { coefficients: vec![exts[0], exts[1], exts[2]] };
        let univariate_poly_symbolic: UnivariatePolynomial<SymbolicExt<F, EF>> =
            UnivariatePolynomial {
                coefficients: univariate_poly.coefficients.iter().map(|c| (*c).into()).collect(),
            };
        let expected_eval = univariate_poly_symbolic.eval_at_point(point.into());
        builder.assert_ext_eq(expected_eval, exts[0] + exts[1] * point + exts[2] * point * point);

        let coeffs = (0..3).map(|_| rand::Rng::gen::<F>(&mut rng)).collect::<Vec<_>>();
        let point_val: F = rand::Rng::gen(&mut rng);
        let witness_stream: Vec<WitnessBlock<C>> =
            [vec![coeffs[0].into(), coeffs[1].into(), coeffs[2].into()], vec![point_val.into()]]
                .concat();

        run_test_recursion(builder.into_operations(), witness_stream);
    }

    #[test]
    fn test_eq_eval() {
        let mut builder = AsmBuilder::<F, EF>::default();
        let vec_1: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let vec_2: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let point_1 = Point::from(vec_1);
        let point_2 = Point::from(vec_2);
        let eq_eval = Mle::full_lagrange_eval(&point_1, &point_2);
        let one: Ext<F, EF> = builder.constant(EF::ONE);
        builder.assert_ext_eq(eq_eval, one);

        let witness_stream: Vec<WitnessBlock<C>> =
            [vec![F::ZERO.into(), F::ONE.into()], vec![F::ZERO.into(), F::ONE.into()]].concat();

        run_test_recursion(builder.into_operations(), witness_stream);
    }

    #[test]
    fn test_full_geq() {
        let mut builder = AsmBuilder::<F, EF>::default();
        let vec_1: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let vec_2: Vec<SymbolicExt<F, EF>> =
            builder.hint_exts_v2(2).iter().copied().map(|x| x.into()).collect::<Vec<_>>();
        let point_1 = Point::from(vec_1);
        let point_2 = Point::from(vec_2);
        let geq_eval = full_geq(&point_1, &point_2);
        let one: Ext<F, EF> = builder.constant(EF::ONE);
        builder.assert_ext_eq(geq_eval, one);

        let witness_stream: Vec<WitnessBlock<C>> =
            [vec![F::ZERO.into(), F::ONE.into()], vec![F::ONE.into(), F::ZERO.into()]].concat();

        run_test_recursion(builder.into_operations(), witness_stream);
    }
}
