use super::{Syscall, SyscallCode, SyscallContext, SyscallRuntime};
use crate::ExecutionError;

pub(crate) struct CommitSyscall;

impl<R: SyscallRuntime> Syscall<R> for CommitSyscall {
    #[allow(clippy::mut_mut)]
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        word_idx: u32,
        public_values_digest_word: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let record = ctx.record_mut();

        if word_idx as usize >= record.public_values.committed_value_digest.len() {
            return Err(ExecutionError::InvalidSyscallArgs());
        }
        record.public_values.committed_value_digest[word_idx as usize] =
            public_values_digest_word;

        Ok(None)
    }
}
