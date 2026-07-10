use std::mem::{size_of, transmute};

use zkm_core_machine::utils::indices_arr;
use zkm_derive::AlignedBorrow;

use crate::chips::poseidon2_skinny::{NUM_INTERNAL_ROUNDS, WIDTH};

pub mod preprocessed;

pub const NUM_POSEIDON2_COLS: usize = size_of::<Poseidon2<u8>>();
const fn make_col_map_degree9() -> Poseidon2<usize> {
    let indices_arr = indices_arr::<NUM_POSEIDON2_COLS>();
    unsafe { transmute::<[usize; NUM_POSEIDON2_COLS], Poseidon2<usize>>(indices_arr) }
}
pub const POSEIDON2_DEGREE9_COL_MAP: Poseidon2<usize> = make_col_map_degree9();

pub const NUM_INTERNAL_ROUNDS_S0: usize = NUM_INTERNAL_ROUNDS - 1;

/// Struct for the poseidon2 skinny non preprocessed column.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2<T: Copy> {
    pub state_var: [T; WIDTH],
    pub internal_rounds_s0: [T; NUM_INTERNAL_ROUNDS_S0],
    /// The state this row computes for its successor within the same permutation invocation
    /// (unused/unconstrained on the output row, which has no successor). Sent forward via an
    /// index-keyed lookup instead of a physical `next` row reference, since the zerocheck
    /// prover's single-row constraint-eval contexts don't support row adjacency.
    pub computed_next_state: [T; WIDTH],
    /// `state_var + round_constants` on an external-round row; zero elsewhere. Witnessed
    /// (instead of computed inline) so the sbox cubing below stays a plain degree-3 expression
    /// in trace columns, since gating a degree-3 expression directly would exceed the zerocheck
    /// prover's degree-3 constraint cap.
    pub add_rc: [T; WIDTH],
    /// `add_rc^3` on an external-round row; zero elsewhere (see `add_rc`).
    pub sbox_out: [T; WIDTH],
}
