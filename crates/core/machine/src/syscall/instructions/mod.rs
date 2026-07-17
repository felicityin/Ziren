use columns::NUM_SYSCALL_INSTR_COLS;
use p3_air::BaseAir;

pub mod air;
pub mod columns;
pub mod trace;

/// A chip that implements the SYSCALL opcode: dispatches to precompile/linux-syscall tables,
/// derives `is_halt`/`num_extra_cycles`, and handles the HALT/COMMIT/COMMIT_DEFERRED_PROOFS
/// special cases.
///
/// Nothing ever emits a synthetic dependency row into `syscall_events` and this chip never
/// produces one either -- every row here is a real, retired instruction.
#[derive(Default)]
pub struct SyscallInstrsChip;

impl<F> BaseAir<F> for SyscallInstrsChip {
    fn width(&self) -> usize {
        NUM_SYSCALL_INSTR_COLS
    }
}
