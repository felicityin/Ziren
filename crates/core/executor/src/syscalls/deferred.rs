use super::{Syscall, SyscallCode, SyscallContext, SyscallRuntime};
use crate::ExecutionError;

pub(crate) struct CommitDeferredSyscall;

impl<R: SyscallRuntime> Syscall<R> for CommitDeferredSyscall {
    #[allow(clippy::mut_mut)]
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        word_idx: u32,
        word: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let record = ctx.record_mut();

        if word_idx as usize >= record.public_values.deferred_proofs_digest.len() {
            return Err(ExecutionError::InvalidSyscallArgs());
        }
        record.public_values.deferred_proofs_digest[word_idx as usize] = word;

        Ok(None)
    }
}
