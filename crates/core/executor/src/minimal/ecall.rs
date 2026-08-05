//! `MinimalExecutor` syscall dispatch. See `minimal/mod.rs`'s module doc for exactly what's real
//! vs. a documented no-op fallback.

use super::MinimalExecutor;
use crate::{register::Register, syscalls::SyscallCode, vm, ExecutionError};
use zkm_primitives::consts::fd::{FD_HINT, FD_PUBLIC_VALUES, FD_STDERR, FD_STDOUT};

impl MinimalExecutor {
    /// Executes the `SYSCALL` at the current `pc`. Returns `next_pc` (the value the caller must
    /// still add 4 to for `next_next_pc`, mirroring `SyscallContext::next_pc`'s default of
    /// `self.pc.wrapping_add(4)` and `HaltSyscall`'s override to `0`).
    pub(super) fn execute_syscall(&mut self) -> Result<u32, ExecutionError> {
        let syscall_id = self.reg(Register::V0);
        let code = SyscallCode::from_u32(syscall_id);
        let arg1 = self.reg(Register::A0);
        let arg2 = self.reg(Register::A1);

        let mut next_pc = self.pc.wrapping_add(4);
        let mut extra_cycles = 0u32;
        let a0_result: Option<u32> = match code {
            SyscallCode::HALT => {
                let exit_code = arg1;
                next_pc = 0;
                if exit_code != 0 {
                    return Err(ExecutionError::HaltWithNonZeroExitCode(exit_code));
                }
                self.exited = true;
                None
            }
            SyscallCode::WRITE => {
                let fd = arg1;
                let write_buf = arg2;
                let nbytes = self.reg(Register::A2);
                // `byte_peek` doesn't bump timestamps (matching `Executor::word`'s untracked-peek
                // semantics, for golden-parity) but does log each byte's owning word as an oracle
                // entry (unlike `Executor::word`) -- `CoreVM` has no backing RAM to peek from
                // during replay, so this data must be in the log or it's unrecoverable. See
                // `word_peek`'s doc comment.
                let bytes: Vec<u8> = (0..nbytes).map(|i| self.byte_peek(write_buf + i)).collect();
                if fd == FD_PUBLIC_VALUES {
                    self.public_values_stream.extend_from_slice(&bytes);
                } else if fd == FD_STDOUT || fd == FD_STDERR {
                    // Skips the real cycle-tracker-command parsing / line-buffered
                    // print-on-newline machinery (`write.rs::update_io_buf`) -- affects only
                    // console output, not executor state.
                } else if fd == FD_HINT {
                    // Hint-stream plumbing (`HINT_READ`/`HINT_LEN`) not implemented yet.
                }
                None
            }
            SyscallCode::SYS_BRK => {
                // Real (not a no-op): Go-style runtime startup calls this before reaching any
                // interesting logic, and depends on a real (not garbage) resolved brk value.
                // Mirrors `SysBrkSyscall::execute`/`resolve_brk` exactly, including its real
                // quirk: the `BRK` register itself is never written back (every call re-derives
                // `initial_brk` fresh from `program.image`, so "current brk" is always just
                // `initial_brk`), and `A3` gets a real (if unused-by-us) `0` write for
                // register-value parity.
                const MAX_HEAP_SIZE: u32 = 0x4000_0000;
                let initial_brk = self
                    .program
                    .image
                    .get(&(Register::BRK as u32))
                    .copied()
                    .unwrap_or_else(|| self.reg(Register::BRK));
                let limit = initial_brk
                    .checked_add(MAX_HEAP_SIZE)
                    .ok_or(ExecutionError::InvalidSyscallArgs())?
                    .min(crate::program::MAX_MEMORY as u32);
                let v0 = arg1.max(initial_brk);
                if v0 > limit {
                    return Err(ExecutionError::InvalidSyscallArgs());
                }
                self.set_reg(Register::A3, 0);
                Some(v0)
            }
            SyscallCode::SHA_COMPRESS => {
                let w_ptr = arg1;
                let h_ptr = arg2;
                let h: [u32; 8] = std::array::from_fn(|i| self.mr(h_ptr + i as u32 * 4));
                let w: [u32; 64] = std::array::from_fn(|i| self.mr(w_ptr + i as u32 * 4));
                let out = vm::sha256_compress(h, &w);
                for (i, &v) in out.iter().enumerate() {
                    self.mw(h_ptr + i as u32 * 4, v);
                }
                extra_cycles = 1;
                None
            }
            SyscallCode::SHA_EXTEND => {
                let w_ptr = arg1;
                for i in 16..64u32 {
                    let w_i_minus_15 = self.mr(w_ptr + (i - 15) * 4);
                    let w_i_minus_2 = self.mr(w_ptr + (i - 2) * 4);
                    let w_i_minus_16 = self.mr(w_ptr + (i - 16) * 4);
                    let w_i_minus_7 = self.mr(w_ptr + (i - 7) * 4);
                    let w_i = vm::sha256_extend_word(
                        w_i_minus_15,
                        w_i_minus_2,
                        w_i_minus_16,
                        w_i_minus_7,
                    );
                    self.mw(w_ptr + i * 4, w_i);
                }
                extra_cycles = 48;
                None
            }
            SyscallCode::KECCAK_SPONGE => {
                let input_ptr = arg1;
                let result_ptr = arg2;
                let input_len_u32s = self.mr(result_ptr + 16 * 4);

                let input_values: Vec<u32> =
                    (0..input_len_u32s).map(|i| self.mr(input_ptr + i * 4)).collect();
                let input_u64_values: Vec<u64> = input_values
                    .chunks_exact(2)
                    .map(|pair| pair[0] as u64 + ((pair[1] as u64) << 32))
                    .collect();

                let mut state = [0u64; vm::KECCAK_STATE_SIZE_U64S];
                for block in input_u64_values.chunks_exact(vm::KECCAK_GENERAL_BLOCK_SIZE_U64S) {
                    vm::keccak_xor_block(&mut state, block);
                    vm::keccakf(&mut state);
                }

                for i in 0..vm::KECCAK_GENERAL_OUTPUT_U64S {
                    let least_sig = (state[i] & 0xFFFF_FFFF) as u32;
                    let most_sig = (state[i] >> 32) as u32;
                    self.mw(result_ptr + (2 * i) as u32 * 4, least_sig);
                    self.mw(result_ptr + (2 * i + 1) as u32 * 4, most_sig);
                }
                extra_cycles = 1;
                None
            }
            // Everything else (precompiles, other Linux shims, hints, unconstrained, VERIFY): a
            // documented no-op -- see the module doc on `minimal/mod.rs`.
            _ => None,
        };

        let a0 = a0_result.unwrap_or(syscall_id);
        self.set_reg(Register::V0, a0);
        self.clk += u64::from(extra_cycles);
        Ok(next_pc)
    }
}
