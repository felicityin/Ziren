use std::mem::size_of;
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
use zkm_hypercube::air::PV_DIGEST_NUM_WORDS;

use crate::{
    adapter::CpuState,
    memory::{RegisterAccessCols, RegisterWriteAccessCols},
    operations::{IsZeroOperation, KoalaBearWordRangeChecker},
};

pub const NUM_SYSCALL_INSTR_COLS: usize = size_of::<SyscallInstrColumns<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct SyscallInstrColumns<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// `op_a`'s access (register `V0`, always register 2 -- a compile-time constant, never a
    /// witnessed index, since SYSCALL always hardcodes its operand registers; see this chip's doc
    /// comment). A read-modify-write: its witnessed `value` is masked/muxed depending on the
    /// syscall kind (see `eval_syscall`), so it needs `RegisterWriteAccessCols` rather than a
    /// directly-fed lookup value.
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// `op_b`'s access (register `A0`, always register 4).
    pub op_b_access: RegisterAccessCols<T>,
    /// `op_c`'s access (register `A1`, always register 5).
    pub op_c_access: RegisterAccessCols<T>,

    pub pc: T,
    pub next_pc: T,

    /// `next_pc + is_halt * (pc + 4)`, witnessed separately since `LookupKind::State`
    /// interaction values must stay affine in the trace columns -- mirrors
    /// `CpuChip::eval_state_chain`'s identical halt-sentinel mechanism.
    pub state_chain_next_pc: T,

    pub num_extra_cycles: T,

    /// Whether the current instruction is a halt instruction.
    pub is_halt: T,

    /// Whether the current syscall is linux syscall.
    pub is_sys_linux: T,

    /// IsZero check on prev_a_value[1] for bidirectional is_sys_linux.
    pub is_prev_a1_zero: IsZeroOperation<T>,

    pub syscall_id: T,

    pub is_enter_unconstrained: IsZeroOperation<T>,
    pub is_hint_len: IsZeroOperation<T>,
    pub is_halt_check: IsZeroOperation<T>,
    pub is_exit_group_check: IsZeroOperation<T>,
    pub is_commit: IsZeroOperation<T>,
    pub is_commit_deferred_proofs: IsZeroOperation<T>,

    pub index_bitmap: [T; PV_DIGEST_NUM_WORDS],

    /// KoalaBear range check for op_b_value.
    /// Active when send_to_table=1 (bug 4) OR is_halt=1 (exit code check).
    pub op_b_range_check: KoalaBearWordRangeChecker<T>,

    /// KoalaBear range check for op_c_value.
    /// Active when send_to_table=1 (bug 4) OR is_commit_deferred_proofs=1 (digest check).
    pub op_c_range_check: KoalaBearWordRangeChecker<T>,

    /// Stored boolean: 1 when op_b needs range check (send_to_table || is_halt).
    pub op_b_check: T,

    /// Stored boolean: 1 when op_c needs range check (send_to_table || is_commit_deferred_proofs).
    pub op_c_check: T,

    pub is_real: T,
}
