//! `MinimalExecutor` syscall dispatch. See `minimal/mod.rs`'s module doc for exactly what's real
//! vs. a documented no-op fallback.

use super::MinimalExecutor;
use crate::{
    events::{FieldOperation, MemoryAccessPosition},
    register::Register,
    syscalls::SyscallCode,
    vm, ExecutionError,
};
use zkm_curves::{
    edwards::{ed25519::Ed25519, WORDS_FIELD_ELEMENT},
    weierstrass::{
        bls12_381::{Bls12381, Bls12381BaseField},
        bn254::{Bn254, Bn254BaseField},
        secp256k1::Secp256k1,
        secp256r1::Secp256r1,
        FpOpField,
    },
    EllipticCurve, COMPRESSED_POINT_BYTES,
};
use zkm_primitives::consts::{
    bytes_to_words_le_vec, fd::{FD_HINT, FD_PUBLIC_VALUES, FD_STDERR, FD_STDOUT}, words_to_bytes_le_vec,
    WORD_SIZE,
};

impl MinimalExecutor {
    /// `p_ptr += q_ptr`, writing the sum back to `p_ptr`. `p` is peeked untracked (its old value
    /// is recoverable from the immediately-following `mw_slice`'s own log, per `slice_peek`'s doc
    /// comment); `q` is read for real, before `p` is overwritten, in case `p_ptr == q_ptr`.
    fn ec_add_dispatch<E: EllipticCurve>(&mut self, p_ptr: u32, q_ptr: u32) {
        let num_words = vm::ec_num_words::<E>();
        let p = self.slice_peek(p_ptr, num_words);
        let q = self.mr_slice(q_ptr, num_words);
        self.clk += 1;
        let result = vm::ec_add::<E>(&p, &q);
        self.mw_slice(p_ptr, &result);
    }

    /// `p_ptr *= 2`, writing the result back to `p_ptr`.
    fn ec_double_dispatch<E: EllipticCurve>(&mut self, p_ptr: u32) {
        let num_words = vm::ec_num_words::<E>();
        let p = self.slice_peek(p_ptr, num_words);
        let result = vm::ec_double::<E>(&p);
        self.mw_slice(p_ptr, &result);
    }

    /// Reads the `x` coordinate at `slice_ptr + num_limbs`, decompresses `y`, and writes `y` back
    /// to `slice_ptr`.
    fn ec_decompress_dispatch<E: EllipticCurve>(
        &mut self,
        slice_ptr: u32,
        sign_bit: u32,
    ) -> Result<(), ExecutionError> {
        let num_words_field_element = vm::ec_num_limb_words::<E>();
        let num_limbs = num_words_field_element * 4;
        let x_vec = self.mr_slice(slice_ptr + num_limbs as u32, num_words_field_element);
        let mut x_bytes_be = words_to_bytes_le_vec(&x_vec);
        x_bytes_be.reverse();
        let y_bytes =
            vm::ec_decompress::<E>(&x_bytes_be, sign_bit).map_err(ExecutionError::CurveError)?;
        let y_words = bytes_to_words_le_vec(&y_bytes);
        self.mw_slice(slice_ptr, &y_words);
        Ok(())
    }

    /// Reads the compressed `y` coordinate at `slice_ptr + COMPRESSED_POINT_BYTES`, decompresses
    /// `x`, and writes `x` back to `slice_ptr`.
    fn ed_decompress_dispatch(&mut self, slice_ptr: u32, sign: u32) -> Result<(), ExecutionError> {
        let y_vec = self.mr_slice(slice_ptr + COMPRESSED_POINT_BYTES as u32, WORDS_FIELD_ELEMENT);
        let y_bytes: [u8; COMPRESSED_POINT_BYTES] =
            words_to_bytes_le_vec(&y_vec).try_into().unwrap();
        let x_bytes =
            vm::ed25519_decompress(y_bytes, sign).map_err(ExecutionError::CurveError)?;
        let x_words = bytes_to_words_le_vec(&x_bytes);
        self.mw_slice(slice_ptr, &x_words);
        Ok(())
    }

    /// `x_ptr = x_ptr op y_ptr` (mod `P::MODULUS`), writing the result back to `x_ptr`. Same
    /// peek-x/read-y/write-x shape as `ec_add_dispatch`.
    fn fp_dispatch<P: FpOpField>(&mut self, x_ptr: u32, y_ptr: u32, op: FieldOperation) {
        let num_words = vm::fp_num_words::<P>();
        let x = self.slice_peek(x_ptr, num_words);
        let y = self.mr_slice(y_ptr, num_words);
        self.clk += 1;
        let result = vm::fp_op::<P>(&x, &y, op);
        self.mw_slice(x_ptr, &result);
    }

    /// `x_ptr = x_ptr op y_ptr` in `P`'s degree-2 extension field, writing the result back to
    /// `x_ptr`.
    fn fp2_addsub_dispatch<P: FpOpField>(&mut self, x_ptr: u32, y_ptr: u32, op: FieldOperation) {
        let num_words = vm::fp2_num_words::<P>();
        let x = self.slice_peek(x_ptr, num_words);
        let y = self.mr_slice(y_ptr, num_words);
        self.clk += 1;
        let result = vm::fp2_addsub::<P>(&x, &y, op);
        self.mw_slice(x_ptr, &result);
    }

    /// `x_ptr *= y_ptr` in `P`'s degree-2 extension field, writing the result back to `x_ptr`.
    fn fp2_mul_dispatch<P: FpOpField>(&mut self, x_ptr: u32, y_ptr: u32) {
        let num_words = vm::fp2_num_words::<P>();
        let x = self.slice_peek(x_ptr, num_words);
        let y = self.mr_slice(y_ptr, num_words);
        self.clk += 1;
        let result = vm::fp2_mul::<P>(&x, &y);
        self.mw_slice(x_ptr, &result);
    }

    /// `x_ptr = (x_ptr * y_ptr) mod modulus`, where `modulus` immediately follows `y_ptr` in
    /// memory, writing the result back to `x_ptr`.
    fn uint256_mul_dispatch(&mut self, x_ptr: u32, y_ptr: u32) {
        let x: [u32; 8] = self.slice_peek(x_ptr, WORDS_FIELD_ELEMENT).try_into().unwrap();
        let y: [u32; 8] = self.mr_slice(y_ptr, WORDS_FIELD_ELEMENT).try_into().unwrap();
        let modulus_ptr = y_ptr + WORDS_FIELD_ELEMENT as u32 * WORD_SIZE as u32;
        let modulus: [u32; 8] = self.mr_slice(modulus_ptr, WORDS_FIELD_ELEMENT).try_into().unwrap();
        self.clk += 1;
        let result = vm::uint256_mul(&x, &y, &modulus);
        self.mw_slice(x_ptr, &result);
    }

    /// `a_ptr * b_ptr`, writing the low 2048 bits to the address in `$a2` and the high 256 bits
    /// to the address in `$a3`.
    fn u256xu2048_mul_dispatch(&mut self, a_ptr: u32, b_ptr: u32) {
        let lo_ptr = self.reg_read_aux(Register::A2);
        let hi_ptr = self.reg_read_aux(Register::A3);
        let a: [u32; vm::U256_NUM_WORDS] = self.mr_slice(a_ptr, vm::U256_NUM_WORDS).try_into().unwrap();
        let b: [u32; vm::U2048_NUM_WORDS] = self.mr_slice(b_ptr, vm::U2048_NUM_WORDS).try_into().unwrap();
        self.clk += 1;
        let (lo, hi) = vm::u256xu2048_mul(&a, &b);
        self.mw_slice(lo_ptr, &lo);
        self.mw_slice(hi_ptr, &hi);
    }

    /// Permutes the 16-word Poseidon2 state at `state_ptr` in place.
    fn poseidon2_permute_dispatch(&mut self, state_ptr: u32) {
        let pre_state: [u32; vm::POSEIDON2_STATE_SIZE] =
            self.slice_peek(state_ptr, vm::POSEIDON2_STATE_SIZE).try_into().unwrap();
        let post_state = vm::poseidon2_permute(pre_state);
        self.mw_slice(state_ptr, &post_state);
    }

    /// Executes the `SYSCALL` at the current `pc`. Returns `next_pc` (the value the caller must
    /// still add 4 to for `next_next_pc`, mirroring `SyscallContext::next_pc`'s default of
    /// `self.pc.wrapping_add(4)` and `HaltSyscall`'s override to `0`).
    pub(super) fn execute_syscall(&mut self) -> Result<u32, ExecutionError> {
        // `V0` stays an untracked live peek here -- the real record comes from the write at the
        // bottom of this function (mirrors `TracingVM::execute_syscall`'s identical comment). `A1`
        // is read before `A0` (C before B), matching `Executor::execute_operation`'s exact order.
        let syscall_id = self.reg(Register::V0);
        let code = SyscallCode::from_u32(syscall_id);
        let arg2 = self.reg_read(Register::A1, MemoryAccessPosition::C);
        let arg1 = self.reg_read(Register::A0, MemoryAccessPosition::B);

        // Only `WRITE` (the hint-hook escape hatch) and `EXIT_UNCONSTRAINED` itself are allowed
        // inside an unconstrained block -- mirrors `Executor::execute_operation`'s identical
        // check exactly (`ExecutionError::InvalidSyscallUsage`, not a panic, to match its
        // catchable-error contract).
        if self.unconstrained && code != SyscallCode::WRITE && code != SyscallCode::EXIT_UNCONSTRAINED
        {
            return Err(ExecutionError::InvalidSyscallUsage(syscall_id as u64));
        }

        if !self.unconstrained {
            self.syscall_counts[code] += 1;
        }

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
                let initial_brk = self
                    .program
                    .image
                    .get(&(Register::BRK as u32))
                    .copied()
                    .unwrap_or_else(|| self.reg(Register::BRK));
                let v0 = vm::resolve_brk(initial_brk, initial_brk, arg1)?;
                self.set_reg_aux(Register::A3, 0);
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
            SyscallCode::SECP256K1_ADD => {
                self.ec_add_dispatch::<Secp256k1>(arg1, arg2);
                None
            }
            SyscallCode::SECP256R1_ADD => {
                self.ec_add_dispatch::<Secp256r1>(arg1, arg2);
                None
            }
            SyscallCode::BN254_ADD => {
                self.ec_add_dispatch::<Bn254>(arg1, arg2);
                None
            }
            SyscallCode::BLS12381_ADD => {
                self.ec_add_dispatch::<Bls12381>(arg1, arg2);
                None
            }
            SyscallCode::SECP256K1_DOUBLE => {
                self.ec_double_dispatch::<Secp256k1>(arg1);
                None
            }
            SyscallCode::SECP256R1_DOUBLE => {
                self.ec_double_dispatch::<Secp256r1>(arg1);
                None
            }
            SyscallCode::BN254_DOUBLE => {
                self.ec_double_dispatch::<Bn254>(arg1);
                None
            }
            SyscallCode::BLS12381_DOUBLE => {
                self.ec_double_dispatch::<Bls12381>(arg1);
                None
            }
            SyscallCode::SECP256K1_DECOMPRESS => {
                self.ec_decompress_dispatch::<Secp256k1>(arg1, arg2)?;
                None
            }
            SyscallCode::SECP256R1_DECOMPRESS => {
                self.ec_decompress_dispatch::<Secp256r1>(arg1, arg2)?;
                None
            }
            SyscallCode::BLS12381_DECOMPRESS => {
                self.ec_decompress_dispatch::<Bls12381>(arg1, arg2)?;
                None
            }
            SyscallCode::ED_ADD => {
                self.ec_add_dispatch::<Ed25519>(arg1, arg2);
                None
            }
            SyscallCode::ED_DECOMPRESS => {
                self.ed_decompress_dispatch(arg1, arg2)?;
                None
            }
            SyscallCode::BN254_FP_ADD => {
                self.fp_dispatch::<Bn254BaseField>(arg1, arg2, FieldOperation::Add);
                None
            }
            SyscallCode::BN254_FP_SUB => {
                self.fp_dispatch::<Bn254BaseField>(arg1, arg2, FieldOperation::Sub);
                None
            }
            SyscallCode::BN254_FP_MUL => {
                self.fp_dispatch::<Bn254BaseField>(arg1, arg2, FieldOperation::Mul);
                None
            }
            SyscallCode::BLS12381_FP_ADD => {
                self.fp_dispatch::<Bls12381BaseField>(arg1, arg2, FieldOperation::Add);
                None
            }
            SyscallCode::BLS12381_FP_SUB => {
                self.fp_dispatch::<Bls12381BaseField>(arg1, arg2, FieldOperation::Sub);
                None
            }
            SyscallCode::BLS12381_FP_MUL => {
                self.fp_dispatch::<Bls12381BaseField>(arg1, arg2, FieldOperation::Mul);
                None
            }
            SyscallCode::BN254_FP2_ADD => {
                self.fp2_addsub_dispatch::<Bn254BaseField>(arg1, arg2, FieldOperation::Add);
                None
            }
            SyscallCode::BN254_FP2_SUB => {
                self.fp2_addsub_dispatch::<Bn254BaseField>(arg1, arg2, FieldOperation::Sub);
                None
            }
            SyscallCode::BN254_FP2_MUL => {
                self.fp2_mul_dispatch::<Bn254BaseField>(arg1, arg2);
                None
            }
            SyscallCode::BLS12381_FP2_ADD => {
                self.fp2_addsub_dispatch::<Bls12381BaseField>(arg1, arg2, FieldOperation::Add);
                None
            }
            SyscallCode::BLS12381_FP2_SUB => {
                self.fp2_addsub_dispatch::<Bls12381BaseField>(arg1, arg2, FieldOperation::Sub);
                None
            }
            SyscallCode::BLS12381_FP2_MUL => {
                self.fp2_mul_dispatch::<Bls12381BaseField>(arg1, arg2);
                None
            }
            SyscallCode::UINT256_MUL => {
                self.uint256_mul_dispatch(arg1, arg2);
                None
            }
            SyscallCode::U256XU2048_MUL => {
                self.u256xu2048_mul_dispatch(arg1, arg2);
                None
            }
            SyscallCode::POSEIDON2_PERMUTE => {
                self.poseidon2_permute_dispatch(arg1);
                None
            }
            SyscallCode::SYS_MMAP | SyscallCode::SYS_MMAP2 => {
                let size = vm::align_size(arg2)?;
                let v0 = if arg1 == 0 {
                    let heap = self.reg(Register::HEAP);
                    self.set_reg_aux(Register::HEAP, heap.wrapping_add(size));
                    heap
                } else {
                    arg1
                };
                self.set_reg_aux(Register::A3, 0);
                Some(v0)
            }
            SyscallCode::SYS_CLONE => {
                self.set_reg_aux(Register::A3, 0);
                Some(1) // Simulate a successful clone operation.
            }
            SyscallCode::SYS_EXT_GROUP => {
                next_pc = 0;
                self.set_reg_aux(Register::A3, 0);
                Some(0)
            }
            SyscallCode::SYS_FCNTL => {
                let (v0, a3) = vm::fcntl_result(arg1, arg2);
                self.set_reg_aux(Register::A3, a3);
                Some(v0)
            }
            SyscallCode::SYS_READ => {
                let (v0, a3) = vm::read_result(arg1);
                self.set_reg_aux(Register::A3, a3);
                Some(v0)
            }
            SyscallCode::SYS_WRITE => {
                let fd = arg1;
                let write_buf = arg2;
                let nbytes = self.reg_read_aux(Register::A2);
                // Every byte's owning word is logged (via `byte_peek`) regardless of `fd`, same
                // as the `WRITE` syscall above -- `CoreVM` must pop a matching entry per byte to
                // stay in sync even for destinations whose content isn't otherwise preserved.
                let bytes: Vec<u8> = (0..nbytes).map(|i| self.byte_peek(write_buf + i)).collect();
                if fd == FD_PUBLIC_VALUES {
                    self.public_values_stream.extend_from_slice(&bytes);
                }
                self.set_reg_aux(Register::A3, 0);
                Some(nbytes)
            }
            SyscallCode::SYS_OPEN
            | SyscallCode::SYS_CLOSE
            | SyscallCode::SYS_RT_SIGACTION
            | SyscallCode::SYS_RT_SIGPROCMASK
            | SyscallCode::SYS_MADVISE
            | SyscallCode::SYS_GETTID
            | SyscallCode::SYS_SCHED_GETAFFINITY
            | SyscallCode::SYS_CLOCK_GETTIME
            | SyscallCode::SYS_NANOSLEEP
            | SyscallCode::SYS_PRLIMIT64
            | SyscallCode::SYS_SIGALTSTACK
            | SyscallCode::SYS_OPENAT
            | SyscallCode::SYS_FSTAT64
            | SyscallCode::SYS_MUNMAP => {
                self.set_reg_aux(Register::A3, 0);
                Some(0)
            }
            SyscallCode::ENTER_UNCONSTRAINED => Some(self.enter_unconstrained()),
            SyscallCode::EXIT_UNCONSTRAINED => {
                next_pc = self.exit_unconstrained();
                Some(0)
            }
            // Everything else (precompiles, hints, VERIFY): a documented no-op -- see the module
            // doc on `minimal/mod.rs`.
            _ => None,
        };

        let a0 = a0_result.unwrap_or(syscall_id);
        self.set_reg(Register::V0, a0, MemoryAccessPosition::A);
        self.clk += u64::from(extra_cycles);
        Ok(next_pc)
    }
}
