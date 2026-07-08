use rayon::prelude::*;
use slop_algebra::{FieldExtensionAlgebra, FieldAlgebra};
use slop_alloc::{buffer, Buffer, CpuBackend};
use slop_tensor::{dot_along_dim, Dimensions, Tensor};

use crate::{partial_eq_with_basis, partial_lagrange, Basis, Point};

/// Evaluates the MLE at a given point.
pub fn eval_mle_at_point<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    point: &Point<EF, CpuBackend>,
) -> Tensor<EF, CpuBackend>
where
    F: FieldAlgebra + Sync + 'static,
    EF: FieldExtensionAlgebra<F> + Send + Sync + 'static,
{
    // Compute the eq(b, point) polynomial.
    let partial_lagrange = partial_lagrange(point);
    // Evaluate the mle via a dot product with the partial lagrange polynomial.
    dot_along_dim(mle, &partial_lagrange, 0)
}

/// Evaluates the MLE at a given eq polynomial.
pub fn eval_mle_at_eq<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    eq: &Tensor<EF, CpuBackend>,
) -> Tensor<EF, CpuBackend>
where
    F: FieldAlgebra + Sync + 'static,
    EF: FieldExtensionAlgebra<F> + Send + Sync + 'static,
{
    // Evaluate the mle via a dot product with the eq polynomial.
    dot_along_dim(mle, eq, 0)
}

/// Returns a tensor of zero evaluations.
pub fn zero_evaluations<F: FieldAlgebra>(num_polynomials: usize) -> Tensor<F, CpuBackend> {
    Tensor::zeros_in([num_polynomials], CpuBackend)
}

pub fn eval_mle_at_point_with_basis<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    point: &Point<EF, CpuBackend>,
    basis: Basis,
) -> Tensor<EF, CpuBackend>
where
    F: FieldAlgebra + Sync,
    EF: FieldExtensionAlgebra<F> + Send + Sync,
{
    let partial_lagrange = partial_eq_with_basis(point, basis);
    let mut sizes = mle.sizes().to_vec();
    sizes.remove(0);
    let dimensions = Dimensions::try_from(sizes).unwrap();
    let mut dst = Tensor { storage: buffer![], dimensions };
    let total_len = dst.total_len();
    let dot_products = mle
        .as_buffer()
        .par_chunks_exact(mle.strides()[0])
        .zip(partial_lagrange.as_buffer().par_iter())
        .map(|(chunk, scalar)| chunk.iter().map(|a| scalar.clone() * a.clone()).collect())
        .reduce(
            || vec![EF::zero(); total_len],
            |mut a, b| {
                a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += b.clone());
                a
            },
        );

    let dot_products = Buffer::from(dot_products);
    dst.storage = dot_products;
    dst
}

/// Alias for `eval_mle_at_point` for backwards compatibility.
pub fn eval_mle_at_point_blocking<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    point: &Point<EF, CpuBackend>,
) -> Tensor<EF, CpuBackend>
where
    F: FieldAlgebra + Sync + 'static,
    EF: FieldExtensionAlgebra<F> + Send + Sync + 'static,
{
    eval_mle_at_point(mle, point)
}

/// Alias for `eval_mle_at_point_with_basis` for backwards compatibility.
pub fn eval_mle_at_point_blocking_with_basis<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    point: &Point<EF, CpuBackend>,
    basis: Basis,
) -> Tensor<EF, CpuBackend>
where
    F: FieldAlgebra + Sync,
    EF: FieldExtensionAlgebra<F> + Send + Sync,
{
    eval_mle_at_point_with_basis(mle, point, basis)
}

/// Interpreting the internal vector of `mle` as the monomial-basis coefficients of a multilinear
/// polynomial, evaluate that multilinear at `point`.
pub fn eval_monomial_basis_mle_at_point<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    point: &Point<EF, CpuBackend>,
) -> Tensor<EF, CpuBackend>
where
    F: FieldAlgebra + Sync,
    EF: FieldExtensionAlgebra<F> + Send + Sync,
{
    eval_mle_at_point_with_basis(mle, point, Basis::Monomial)
}

/// Alias for backwards compatibility.
pub fn eval_monomial_basis_mle_at_point_blocking<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    point: &Point<EF, CpuBackend>,
) -> Tensor<EF, CpuBackend>
where
    F: FieldAlgebra + Sync,
    EF: FieldExtensionAlgebra<F> + Send + Sync,
{
    eval_monomial_basis_mle_at_point(mle, point)
}
