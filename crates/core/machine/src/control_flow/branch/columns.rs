use std::mem::size_of;
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
use zkm_hypercube::word::Word;

use crate::{
    adapter::{CpuState, ITypeImmutableReader},
    operations::{AddOperation, KoalaBearWordRangeChecker},
};

pub const NUM_BRANCH_COLS: usize = size_of::<BranchColumns<u8>>();

/// The column layout for branching.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct BranchColumns<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// Register operand access for `a`/`b`/`c`: `op_a`/`op_b` are both read-only (a branch
    /// compares two registers but never writes one), `op_c` is always the instruction's own
    /// encoded immediate offset (never a register) -- the Program lookup's instruction word is
    /// reconstructed from this adapter's fields directly, so no separate `InstructionCols` is
    /// needed (see `Air::eval`).
    pub reader: ITypeImmutableReader<T>,

    /// The current program counter.
    pub pc: T,

    /// The next program counter.
    pub next_pc: Word<T>,
    pub next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// The next next program counter: `next_pc + op_c` when branching (cross-checked against
    /// `add_operation.value` below), `next_pc + 4` otherwise.
    pub next_next_pc: Word<T>,

    /// Range check for next next program counter.
    pub next_next_pc_range_checker: KoalaBearWordRangeChecker<T>,

    /// Computes `next_pc + op_c` locally (no cross-chip lookup); only meaningful when
    /// `is_branching`, since `next_next_pc` takes a different formula (`next_pc + 4`) otherwise.
    pub add_operation: AddOperation<T>,

    /// Branch Instructions Selectors.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_beq: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_bne: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_bltz: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_blez: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_bgtz: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_bgez: T,

    /// The branching column is equal to:
    ///
    /// > is_beq & a_eq_b ||
    /// > is_bne & !a_eq_b ||
    /// > is_bltz & a_lt_0 ||
    /// > is_bgtz & a_gt_0 ||
    /// > is_blez & (a_lt_0  | a_eq_0) ||
    /// > is_bgez & (a_gt_0  | a_eq_0)
    pub is_branching: T,

    /// Whether a is greater than b.
    pub a_gt_b: T,

    /// Whether a is less than b.
    pub a_lt_b: T,
}
