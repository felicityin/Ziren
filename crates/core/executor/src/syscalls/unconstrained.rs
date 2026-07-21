use crate::ExecutionError;

use super::{Syscall, SyscallCode, SyscallContext, SyscallRuntime};

pub(crate) struct EnterUnconstrainedSyscall;

impl<R: SyscallRuntime> Syscall<R> for EnterUnconstrainedSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        _: u32,
        _: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        ctx.rt.enter_unconstrained();
        Ok(Some(1))
    }
}

pub(crate) struct ExitUnconstrainedSyscall;

impl<R: SyscallRuntime> Syscall<R> for ExitUnconstrainedSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        _: u32,
        _: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        ctx.rt.exit_unconstrained();
        // The runtime's pc was just rolled back to right after the matching `enter_unconstrained`
        // call, so `next_pc` (which `execute_operation` reads off `ctx.next_pc` when this syscall
        // returns) must be re-derived from it rather than kept at whatever it was before exit.
        ctx.next_pc = ctx.rt.pc().wrapping_add(4);
        Ok(Some(0))
    }
}
