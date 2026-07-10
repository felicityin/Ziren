use zkm_derive::AlignedBorrow;

use crate::chips::{
    mem::MemoryAccessColsChips,
    poseidon2_skinny::{NUM_ROUND_CONSTANTS, WIDTH},
};

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct RoundCountersPreprocessedCols<T: Copy> {
    pub is_input_round: T,
    pub is_external_round: T,
    pub is_internal_round: T,
    pub round_constants: [T; NUM_ROUND_CONSTANTS],
}

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct Poseidon2PreprocessedColsSkinny<T: Copy> {
    pub memory_preprocessed: [MemoryAccessColsChips<T>; WIDTH],
    pub round_counters_preprocessed: RoundCountersPreprocessedCols<T>,
    /// A shard-wide row counter, used to chain the round state across rows of the same
    /// permutation invocation via an index-keyed lookup instead of physical row adjacency.
    pub index: T,
    /// Distinguishes a real (possibly output-round) row from a padding row; the three
    /// `round_counters_preprocessed` flags alone can't tell an output row (all three zero) apart
    /// from padding (also all three zero).
    pub is_real: T,
}

pub type Poseidon2PreprocessedCols<T> = Poseidon2PreprocessedColsSkinny<T>;
