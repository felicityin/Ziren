use crate::program::MAX_MEMORY;

use crate::ExecutionError;

use super::{Syscall, SyscallCode, SyscallContext, SyscallRuntime};

pub(crate) struct VerifySyscall;

impl<R: SyscallRuntime> Syscall<R> for VerifySyscall {
    #[allow(clippy::mut_mut)]
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        vkey_ptr: u32,
        pv_digest_ptr: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        // vkey_ptr is a pointer to [u32; 8] which contains the verification key.
        // pv_digest_ptr is a pointer to [u32; 8] which contains the public values digest.

        if !vkey_ptr.is_multiple_of(4) || !pv_digest_ptr.is_multiple_of(4) {
            return Err(ExecutionError::InvalidSyscallArgs());
        }

        if vkey_ptr as usize + 32 > MAX_MEMORY || pv_digest_ptr as usize + 32 > MAX_MEMORY {
            return Err(ExecutionError::InvalidSyscallArgs());
        }

        let vkey = (0..8).map(|i| ctx.word_unsafe(vkey_ptr + i * 4)).collect::<Vec<u32>>();

        let pv_digest = (0..8).map(|i| ctx.word_unsafe(pv_digest_ptr + i * 4)).collect::<Vec<u32>>();

        let vkey_bytes: [u32; 8] = vkey.try_into().unwrap();
        let pv_digest_bytes: [u32; 8] = pv_digest.try_into().unwrap();

        ctx.rt.verify_deferred_proof(vkey_bytes, pv_digest_bytes)?;

        Ok(None)
    }
}
