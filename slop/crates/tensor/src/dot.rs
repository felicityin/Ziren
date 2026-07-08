use rayon::prelude::*;

use slop_algebra::{FieldExtensionAlgebra, FieldAlgebra};
use slop_alloc::{buffer, Buffer, CpuBackend};

use crate::{Dimensions, Tensor};

/// Compute the dot product of a tensor with a scalar tensor along a given dimension.
///
/// This scalar tensor is assumed to be a `1D` tensor, which is any tensor of a shape
/// `[len, 1, 1, 1,..]`.
pub fn dot_along_dim<T, U>(
    src: &Tensor<T, CpuBackend>,
    scalars: &Tensor<U, CpuBackend>,
    dim: usize,
) -> Tensor<U, CpuBackend>
where
    T: FieldAlgebra + 'static + Sync,
    U: FieldExtensionAlgebra<T> + 'static + Send + Sync,
{
    let mut sizes = src.sizes().to_vec();
    sizes.remove(dim);
    let dimensions = Dimensions::try_from(sizes).unwrap();
    let mut dst = Tensor { storage: buffer![], dimensions };
    let max_scalar_dim = *scalars.sizes().iter().max().unwrap();
    assert_eq!(max_scalar_dim, scalars.total_len(), "The scalar tensor must be a 1D tensor");
    match dim {
        0 => {
            assert!(
                src.sizes().len() <= 2,
                "Only 1D and 2D dimensional tensors are supported for dim 0"
            );
            let total_len = dst.total_len();
            let dot_products = src
                .as_buffer()
                .par_chunks_exact(src.strides()[0])
                .zip(scalars.as_buffer().par_iter())
                .map(|(chunk, scalar)| chunk.iter().map(|a| scalar.clone() * a.clone()).collect())
                .reduce(
                    || vec![U::zero(); total_len],
                    |mut a, b| {
                        a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += b.clone());
                        a
                    },
                );

            let dot_products = Buffer::from(dot_products);
            dst.storage = dot_products;
        }
        dim if dim == src.sizes().len() - 1 => {
            let mut dst_storage = Vec::<U>::with_capacity(dst.total_len());
            src.as_buffer()
                .par_chunks_exact(src.strides()[dim - 1])
                .map(|chunk| {
                    scalars
                        .as_buffer()
                        .iter()
                        .zip(chunk.iter())
                        .map(|(a, b)| a.clone() * b.clone())
                        .sum::<U>()
                })
                .collect_into_vec(&mut dst_storage);
            dst.storage = Buffer::from(dst_storage);
        }
        _ => {
            panic!("Unsupported dot product dimension {} for tensor sizes: {:?}", dim, src.sizes())
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use slop_algebra::FieldAlgebra;
    use slop_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn test_dot_along_dim_0() {
        let mut rng = rand::thread_rng();
        let tensor = Tensor::<BabyBear, CpuBackend>::rand(&mut rng, [1500, 10]);
        let scalars = Tensor::<BabyBear, CpuBackend>::rand(&mut rng, [1500]);
        let dot = dot_along_dim(&tensor, &scalars, 0);
        for j in 0..10 {
            let mut dot_product = BabyBear::zero();
            for i in 0..1500 {
                dot_product += *scalars[[i]] * *tensor[[i, j]];
            }
            assert_eq!(*dot[[j]], dot_product);
        }
    }

    #[test]
    fn test_dot_along_dim_last() {
        let mut rng = rand::thread_rng();
        let tensor = Tensor::<BabyBear, CpuBackend>::rand(&mut rng, [10, 1500, 10]);
        let scalars = Tensor::<BabyBear, CpuBackend>::rand(&mut rng, [10]);
        let dot = dot_along_dim(&tensor, &scalars, 2);
        for k in 0..10 {
            for i in 0..1500 {
                let mut dot_product = BabyBear::zero();
                for j in 0..10 {
                    dot_product += *scalars[[j]] * *tensor[[k, i, j]];
                }
                assert_eq!(*dot[[k, i]], dot_product);
            }
        }
    }
}
