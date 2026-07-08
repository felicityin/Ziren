use slop_algebra::FieldAlgebra;
use slop_alloc::CpuBackend;

use crate::Tensor;

/// Add a scalar value to all elements of a tensor in place.
pub fn add_assign<T: FieldAlgebra>(lhs: &mut Tensor<T, CpuBackend>, rhs: T) {
    let lhs = lhs.as_mut_slice();
    for elem in lhs.iter_mut() {
        *elem += rhs.clone();
    }
}
