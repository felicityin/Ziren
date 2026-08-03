use backtrace::Backtrace;
use p3_field::Field;
use serde::{Deserialize, Serialize};
use shape::RecursionShape;
use zkm_hypercube::air::{MachineAir, MachineProgram};
use zkm_hypercube::septic_digest::SepticDigest;

use crate::*;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecursionProgram<F> {
    pub instructions: Vec<Instruction<F>>,
    pub total_memory: usize,
    #[serde(skip)]
    pub traces: Vec<Option<Backtrace>>,
    pub shape: Option<RecursionShape>,
}

impl<F: Field> MachineProgram<F> for RecursionProgram<F> {
    fn pc_start(&self) -> F {
        F::ZERO
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<F> {
        SepticDigest::<F>::zero()
    }
}

impl<F: Field> RecursionProgram<F> {
    /// The exact target row count for `air`'s trace, if a shape is configured. This is a real
    /// row count, not a log2 exponent: the underlying `Mle`/`Tensor` machinery accepts any row
    /// count (`slop/crates/multilinear/src/base.rs`'s `num_variables` computes
    /// `next_power_of_two().ilog2()` on demand rather than requiring the stored data to already
    /// have that many rows), and the jagged/stacked PCS's own alignment requirement applies to
    /// the aggregate committed area across all chips, not any individual chip's row count
    /// (`slop/crates/stacked/src/{prover,fixed_rate}.rs`). The shape mechanism exists to make
    /// different real programs converge onto a small, stable set of trace sizes for a
    /// consistent circuit/VK shape across runs, not because padding to a power of two is
    /// itself required.
    #[inline]
    pub fn fixed_num_rows<A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        self.shape
            .as_ref()
            .map(|shape| {
                shape
                    .inner
                    .get(&air.name())
                    .unwrap_or_else(|| panic!("Chip {} not found in specified shape", air.name()))
            })
            .copied()
    }

    pub fn shape_mut(&mut self) -> &mut Option<RecursionShape> {
        &mut self.shape
    }
}
