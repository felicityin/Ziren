use rayon::prelude::*;

use slop_algebra::Field;
use slop_alloc::{buffer, Buffer, CpuBackend};

use crate::{Dimensions, Tensor};

/// Computes the sum of the tensor along a dimension.
pub fn sum_tensor_dim<T: Field>(src: &Tensor<T, CpuBackend>, dim: usize) -> Tensor<T, CpuBackend> {
    let mut sizes = src.sizes().to_vec();
    sizes.remove(dim);
    let dimensions = Dimensions::try_from(sizes).unwrap();
    let mut dst = Tensor { storage: buffer![], dimensions };
    assert_eq!(dim, 0, "Only sum along the first dimension is supported");
    let total_len = dst.total_len();
    let dim_stride = src.strides()[dim];

    let sums = src
        .as_buffer()
        .par_chunks_exact(dim_stride)
        .fold(
            || vec![T::zero(); total_len],
            |mut acc, item| {
                acc.iter_mut().zip(item).for_each(|(a, b)| *a += *b);
                acc
            },
        )
        .reduce(
            || vec![T::zero(); total_len],
            |mut a, b| {
                a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += *b);
                a
            },
        );

    let sums = Buffer::from(sums);
    dst.storage = sums;
    dst
}

impl<T: Field> Tensor<T, CpuBackend> {
    /// Computes the sum of the tensor along a dimension.
    pub fn sum(&self, dim: usize) -> Tensor<T, CpuBackend> {
        sum_tensor_dim(self, dim)
    }
}

#[cfg(test)]
mod tests {
    use slop_algebra::FieldAlgebra;
    use slop_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn test_sum() {
        let mut rng = rand::thread_rng();

        let sizes = [3, 4];

        let a = Tensor::<BabyBear>::rand(&mut rng, sizes);
        let b = a.sum(0);
        for j in 0..sizes[1] {
            let mut sum = BabyBear::zero();
            for i in 0..sizes[0] {
                sum += *a[[i, j]];
            }
            assert_eq!(sum, *b[[j]]);
        }

        let sizes = [3, 4, 5];

        let a = Tensor::<BabyBear>::rand(&mut rng, sizes);
        let b = a.sum(0);
        for j in 0..sizes[1] {
            for k in 0..sizes[2] {
                let mut sum = BabyBear::zero();
                for i in 0..sizes[0] {
                    sum += *a[[i, j, k]];
                }
                assert_eq!(sum, *b[[j, k]]);
            }
        }
    }
}
