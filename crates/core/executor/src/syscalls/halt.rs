use super::{context::SyscallContext, Syscall, SyscallCode, SyscallRuntime};
use crate::ExecutionError;

pub(crate) struct HaltSyscall;

impl<R: SyscallRuntime> Syscall<R> for HaltSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        exit_code: u32,
        _: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        ctx.set_next_pc(0);
        ctx.set_exit_code(exit_code);
        Ok(None)
    }
}
