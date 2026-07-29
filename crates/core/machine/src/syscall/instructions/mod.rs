use columns::NUM_SYSCALL_INSTR_COLS;
use p3_air::BaseAir;

pub mod air;
pub mod columns;
pub mod trace;

/// A chip that implements the SYSCALL opcode: dispatches to precompile/linux-syscall tables,
/// derives `is_halt`/`num_extra_cycles`, and handles the HALT/COMMIT/COMMIT_DEFERRED_PROOFS
/// special cases.
///
/// SYSCALL always hardcodes its operand registers (`op_a`=`V0`=2, `op_b`=`A0`=4, `op_c`=`A1`=5 --
/// see `Instruction::decode`), never a variable field the way most other opcodes' `op_a`/`op_b`/
/// `op_c` are -- so this chip needs no witnessed register-index columns at all (unlike
/// `RTypeReader`/`AluTypeReader`), just the cheap [`crate::memory::RegisterAccessCols`]/
/// [`crate::memory::RegisterWriteAccessCols`] value+timestamp tracking. `op_a` is never register
/// 0 (it's always `V0`), so this chip never needs `AluX0Chip` routing either.
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
