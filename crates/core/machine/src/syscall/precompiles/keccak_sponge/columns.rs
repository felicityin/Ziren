use core::mem::size_of;

use crate::adapter::CpuState;
use crate::memory::{MemoryReadCols, MemoryWriteCols};
use crate::operations::{IsZeroOperation, XorOperation};
use crate::syscall::precompiles::keccak_sponge::{
    KECCAK_GENERAL_OUTPUT_U32S, KECCAK_GENERAL_RATE_U32S, KECCAK_STATE_U32S,
};

use p3_keccak_air::KeccakCols;
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
use zkm_hypercube::word::Word;

/// KeccakSpongeCols is the column layout for the keccak sponge.
/// The number of rows equal to the number of block.
#[derive(AlignedBorrow)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub(crate) struct KeccakSpongeCols<T: Copy> {
    pub keccak: KeccakCols<T>,
    /// The round position (0..NUM_ROUNDS) of this row within its block's Keccak-f permutation.
    /// Cross-checked combinatorially against `keccak.step_flags` and used, together with
    /// `already_absorbed_u32s`, as the key for the `KeccakPermuteRound` interaction chain that
    /// replaces the old row-adjacent (`next`-row) permutation chaining.
    pub round_index: T,
    pub block_mem: [MemoryReadCols<T>; KECCAK_GENERAL_RATE_U32S],
    pub state: CpuState<T>,
    pub is_real: T,
    pub read_block: T,
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub input_address: T,
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub output_address: T,
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub input_len: T,
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub already_absorbed_u32s: T,
    pub is_absorbed: T,
    pub receive_syscall: T,
    pub write_output: T,
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub is_first_input_block: T,
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub is_final_input_block: T,
    /// `is_first_input_block == (already_absorbed_u32s == 0)`, checked combinatorially.
    pub is_absorbed_zero: IsZeroOperation<T>,
    /// `is_final_input_block == (already_absorbed_u32s == input_len - RATE)`, checked
    /// combinatorially.
    pub is_final_block_zero: IsZeroOperation<T>,
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub original_state: [Word<T>; KECCAK_STATE_U32S],
    pub xored_general_rate: [XorOperation<T>; KECCAK_GENERAL_RATE_U32S],
    pub input_length_mem: MemoryReadCols<T>,
    pub output_mem: [MemoryWriteCols<T>; KECCAK_GENERAL_OUTPUT_U32S],
}

pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();
