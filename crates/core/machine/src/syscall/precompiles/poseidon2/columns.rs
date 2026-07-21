use core::mem::size_of;

use crate::memory::MemoryWriteCols;
use crate::operations::poseidon2::{Poseidon2Operation, WIDTH};
use crate::operations::KoalaBearWordRangeChecker;
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;

/// Poseidon2MemCols is the column layout for the poseidon2 permutation.
#[derive(Debug, Clone, AlignedBorrow)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub(crate) struct Poseidon2MemCols<T: Copy> {
    pub poseidon2: Poseidon2Operation<T>,

    pub shard: T,
    pub clk: T,
    /// The clk's high limb (bits above the low 28-bit window), used only for
    /// `eval_memory_access_slice` -- `shard`/`clk` (the low 28 bits) stay as-is for
    /// `receive_syscall`, matching `SyscallChip`'s own (unwidened) within-shard interaction key.
    pub clk_high: T,
    pub state_addr: T,

    /// Memory columns for the state
    pub state_mem: [MemoryWriteCols<T>; WIDTH],

    /// Columns to KoalaBear range check the state
    pub pre_state_range_check_cols: [KoalaBearWordRangeChecker<T>; WIDTH],
    pub post_state_range_check_cols: [KoalaBearWordRangeChecker<T>; WIDTH],

    pub is_real: T,
}

pub(crate) const NUM_COLS: usize = size_of::<Poseidon2MemCols<u8>>();
