use slop_algebra::FieldAlgebra;
use slop_alloc::{Backend, CpuBackend};
use slop_tensor::Tensor;

/// Backend trait for basic MLE operations that need to work across CPU and GPU.
pub trait MleBaseBackend<F: FieldAlgebra>: Backend {
    /// Returns the number of polynomials in the batch.
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize;

    /// Returns the number of variables in the polynomials.
    fn num_variables(guts: &Tensor<F, Self>) -> u32;

    /// Number of non-zero entries in the MLE.
    fn num_non_zero_entries(guts: &Tensor<F, Self>) -> usize;

    /// Creates an uninitialized MLE tensor.
    fn uninit_mle(&self, num_polynomials: usize, num_non_zero_entries: usize) -> Tensor<F, Self>;
}

impl<F: FieldAlgebra> MleBaseBackend<F> for CpuBackend {
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize {
        guts.sizes()[1]
    }

    fn num_variables(guts: &Tensor<F, Self>) -> u32 {
        guts.sizes()[0].next_power_of_two().ilog2()
    }

    fn num_non_zero_entries(guts: &Tensor<F, Self>) -> usize {
        guts.sizes()[0]
    }

    fn uninit_mle(&self, num_polynomials: usize, num_non_zero_entries: usize) -> Tensor<F, Self> {
        Tensor::with_sizes_in([num_non_zero_entries, num_polynomials], *self)
    }
}

// Convenience functions that delegate to the CpuBackend implementation

/// Returns the number of polynomials in the batch.
pub fn num_polynomials<F: FieldAlgebra>(guts: &Tensor<F, CpuBackend>) -> usize {
    CpuBackend::num_polynomials(guts)
}

/// Returns the number of variables in the polynomials.
pub fn num_variables<F: FieldAlgebra>(guts: &Tensor<F, CpuBackend>) -> u32 {
    CpuBackend::num_variables(guts)
}

/// Number of non-zero entries in the MLE.
pub fn num_non_zero_entries<F: FieldAlgebra>(guts: &Tensor<F, CpuBackend>) -> usize {
    CpuBackend::num_non_zero_entries(guts)
}

pub fn uninit_mle<F: FieldAlgebra>(
    num_polynomials: usize,
    num_non_zero_entries: usize,
) -> Tensor<F, CpuBackend> {
    CpuBackend.uninit_mle(num_polynomials, num_non_zero_entries)
}
