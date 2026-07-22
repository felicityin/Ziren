use super::{Syscall, SyscallCode, SyscallContext, SyscallRuntime};
use crate::ExecutionError;

pub(crate) struct HintLenSyscall;

impl<R: SyscallRuntime> Syscall<R> for HintLenSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        _arg1: u32,
        _arg2: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        Ok(Some(ctx.rt.resolve_hint_len()?))
    }
}

pub(crate) struct HintReadSyscall;

impl<R: SyscallRuntime> Syscall<R> for HintReadSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        ptr: u32,
        len: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        ctx.rt.resolve_hint_read(ptr, len)?;
        Ok(None)
    }
}
