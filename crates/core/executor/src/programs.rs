//! MIPS ELFs used for testing.

#[allow(dead_code)]
#[allow(missing_docs)]
pub mod tests {
    use crate::{Instruction, Opcode, Program};

    use test_artifacts::{
        BLS12381_ADD_ELF, BLS12381_DECOMPRESS_ELF, BLS12381_DOUBLE_ELF, BLS12381_FP2_ADDSUB_ELF,
        BLS12381_FP2_MUL_ELF, BLS12381_FP_ELF, BN254_ADD_ELF, BN254_DOUBLE_ELF,
        BN254_FP2_ADDSUB_ELF, BN254_FP2_MUL_ELF, BN254_FP_ELF, ED_ADD_ELF, ED_DECOMPRESS_ELF,
        FIBONACCI_ELF, HELLO_WORLD_ELF, KECCAK_SPONGE_ELF, MAX_MEMORY_ELF, PANIC_ELF,
        POSEIDON2_PERMUTE_ELF, SECP256K1_ADD_ELF, SECP256K1_DECOMPRESS_ELF, SECP256K1_DOUBLE_ELF,
        SECP256R1_ADD_ELF, SECP256R1_DECOMPRESS_ELF, SECP256R1_DOUBLE_ELF, SHA3_CHAIN_ELF,
        SHA_COMPRESS_ELF, SHA_EXTEND_ELF, U256XU2048_MUL_ELF, UINT256_MUL_ELF, UNCONSTRAINED_ELF,
    };

    #[must_use]
    pub fn simple_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// A synthetic (non-ELF) program that reaches an explicit `HALT` syscall (id 0, exit code 0),
    /// isolating whether syscall handling itself (vs. real-ELF loading/bootstrap complexity)
    /// triggers a given bug. Mirrors `zkm_core_machine::programs::tests::halt_only_program`.
    #[must_use]
    pub fn halt_only_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 2, 0, 0, false, true), // v0 = 0 (HALT syscall id)
            Instruction::new(Opcode::ADD, 4, 0, 0, false, true), // a0 = 0 (exit code)
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// A synthetic (non-ELF) program exercising `HINT_LEN`/`HINT_READ`: reads the first `stdin`
    /// entry's length via `SYSHINTLEN`, then its two words via `SYSHINTREAD` followed by real
    /// `LW`s (forcing the hinted values to materialize into real memory touches, not just
    /// `hint_seed`), before halting. Callers must feed a single 8-byte `stdin` entry.
    #[must_use]
    #[allow(clippy::unreadable_literal)]
    pub fn hint_read_program() -> Program {
        use crate::syscalls::SyscallCode;

        const PTR: u32 = 0x27654320;
        let instructions = vec![
            // t0 (reg 8) = SYSHINTLEN() -- the stdin entry's length.
            Instruction::new(Opcode::ADD, 2, 0, SyscallCode::SYSHINTLEN as u32, false, true),
            Instruction::new(Opcode::ADD, 4, 0, 0, false, true),
            Instruction::new(Opcode::ADD, 5, 0, 0, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
            Instruction::new(Opcode::ADD, 8, 2, 0, false, true),
            // SYSHINTREAD(PTR, t0) -- seeds hint_seed[PTR..PTR+len).
            Instruction::new(Opcode::ADD, 4, 0, PTR, false, true),
            Instruction::new(Opcode::ADD, 5, 8, 0, false, true),
            Instruction::new(Opcode::ADD, 2, 0, SyscallCode::SYSHINTREAD as u32, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
            // Real loads to force both hinted words to materialize into memory.
            Instruction::new(Opcode::LW, 9, 0, PTR, false, true),
            Instruction::new(Opcode::LW, 10, 0, PTR + 4, false, true),
            // HALT.
            Instruction::new(Opcode::ADD, 2, 0, 0, false, true),
            Instruction::new(Opcode::ADD, 4, 0, 0, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// A synthetic (non-ELF) program invoking the built-in `fp_inverse` hook: writes a
    /// `[len=1 (BE u32) || element=3 || modulus=7]` request to `FD_FP_INV`, reads the spliced
    /// result back via `SYSHINTLEN`/`SYSHINTREAD`, and commits it (the modular inverse of 3 mod 7,
    /// i.e. 5, since `3*5 = 15 = 2*7 + 1`) as public values before halting.
    #[must_use]
    #[allow(clippy::unreadable_literal)]
    pub fn hook_fp_inverse_program() -> Program {
        use crate::{hook::FD_FP_INV, syscalls::SyscallCode};
        use zkm_primitives::consts::fd::FD_PUBLIC_VALUES;

        const REQ_PTR: u32 = 0x1000;
        const RESULT_PTR: u32 = 0x2000;
        let instructions = vec![
            // Request buffer: [len=1 (BE u32) || element=3 || modulus=7].
            Instruction::new(Opcode::ADD, 8, 0, 0x01000000, false, true),
            Instruction::new(Opcode::SW, 8, 0, REQ_PTR, false, true),
            Instruction::new(Opcode::ADD, 8, 0, 0x0000_0703, false, true),
            Instruction::new(Opcode::SW, 8, 0, REQ_PTR + 4, false, true),
            // WRITE(FD_FP_INV, REQ_PTR, 6) -- invokes the hook.
            Instruction::new(Opcode::ADD, 2, 0, SyscallCode::WRITE as u32, false, true),
            Instruction::new(Opcode::ADD, 4, 0, FD_FP_INV, false, true),
            Instruction::new(Opcode::ADD, 5, 0, REQ_PTR, false, true),
            Instruction::new(Opcode::ADD, 6, 0, 6, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
            // Read the hook's spliced result back.
            Instruction::new(Opcode::ADD, 2, 0, SyscallCode::SYSHINTLEN as u32, false, true),
            Instruction::new(Opcode::ADD, 4, 0, 0, false, true),
            Instruction::new(Opcode::ADD, 5, 0, 0, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
            Instruction::new(Opcode::ADD, 9, 2, 0, false, true),
            Instruction::new(Opcode::ADD, 4, 0, RESULT_PTR, false, true),
            Instruction::new(Opcode::ADD, 5, 9, 0, false, true),
            Instruction::new(Opcode::ADD, 2, 0, SyscallCode::SYSHINTREAD as u32, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
            // A real load to force the hinted word to materialize into memory: `WRITE`'s own byte
            // reads (below) are an untracked peek that, like `Executor::byte`, deliberately never
            // consults the hint side-table (see `MinimalExecutor::hint_seed`'s doc comment).
            Instruction::new(Opcode::LW, 10, 0, RESULT_PTR, false, true),
            // Commit the result word as public values.
            Instruction::new(Opcode::ADD, 2, 0, SyscallCode::WRITE as u32, false, true),
            Instruction::new(Opcode::ADD, 4, 0, FD_PUBLIC_VALUES, false, true),
            Instruction::new(Opcode::ADD, 5, 0, RESULT_PTR, false, true),
            Instruction::new(Opcode::ADD, 6, 0, 4, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
            // HALT.
            Instruction::new(Opcode::ADD, 2, 0, 0, false, true),
            Instruction::new(Opcode::ADD, 4, 0, 0, false, true),
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// Get the fibonacci program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn fibonacci_program() -> Program {
        Program::from(FIBONACCI_ELF).unwrap()
    }

    /// Get the max_memory program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn max_memory_program() -> Program {
        Program::from(MAX_MEMORY_ELF).unwrap()
    }

    /// Get the hello world program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn hello_world_program() -> Program {
        Program::from(HELLO_WORLD_ELF).unwrap()
    }

    /// Get the sha3-chain program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn sha3_chain_program() -> Program {
        Program::from(SHA3_CHAIN_ELF).unwrap()
    }

    /// Get the sha256-extend precompile program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn sha_extend_program() -> Program {
        Program::from(SHA_EXTEND_ELF).unwrap()
    }

    /// Get the sha256-compress precompile program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn sha_compress_program() -> Program {
        Program::from(SHA_COMPRESS_ELF).unwrap()
    }

    /// Get the bn254 fp add/sub/mul program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bn254_fp_program() -> Program {
        Program::from(BN254_FP_ELF).unwrap()
    }

    /// Get the bn254 fp2 add/sub program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bn254_fp2_addsub_program() -> Program {
        Program::from(BN254_FP2_ADDSUB_ELF).unwrap()
    }

    /// Get the bn254 fp2 mul program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bn254_fp2_mul_program() -> Program {
        Program::from(BN254_FP2_MUL_ELF).unwrap()
    }

    /// Get the bls12381 fp add/sub/mul program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bls12381_fp_program() -> Program {
        Program::from(BLS12381_FP_ELF).unwrap()
    }

    /// Get the bls12381 fp2 add/sub program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bls12381_fp2_addsub_program() -> Program {
        Program::from(BLS12381_FP2_ADDSUB_ELF).unwrap()
    }

    /// Get the bls12381 fp2 mul program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bls12381_fp2_mul_program() -> Program {
        Program::from(BLS12381_FP2_MUL_ELF).unwrap()
    }

    /// Get the ed25519 add program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn ed_add_program() -> Program {
        Program::from(ED_ADD_ELF).unwrap()
    }

    /// Get the ed25519 decompress program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn ed_decompress_program() -> Program {
        Program::from(ED_DECOMPRESS_ELF).unwrap()
    }

    /// Get the secp256k1 add program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn secp256k1_add_program() -> Program {
        Program::from(SECP256K1_ADD_ELF).unwrap()
    }

    /// Get the secp256k1 double program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn secp256k1_double_program() -> Program {
        Program::from(SECP256K1_DOUBLE_ELF).unwrap()
    }

    /// Get the secp256k1 decompress program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn secp256k1_decompress_program() -> Program {
        Program::from(SECP256K1_DECOMPRESS_ELF).unwrap()
    }

    /// Get the secp256r1 decompress program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn secp256r1_decompress_program() -> Program {
        Program::from(SECP256R1_DECOMPRESS_ELF).unwrap()
    }

    /// Get the bn254 add program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bn254_add_program() -> Program {
        Program::from(BN254_ADD_ELF).unwrap()
    }

    /// Get the bn254 double program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bn254_double_program() -> Program {
        Program::from(BN254_DOUBLE_ELF).unwrap()
    }

    /// Get the bls12381 add program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bls12381_add_program() -> Program {
        Program::from(BLS12381_ADD_ELF).unwrap()
    }

    /// Get the bls12381 double program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bls12381_double_program() -> Program {
        Program::from(BLS12381_DOUBLE_ELF).unwrap()
    }

    /// Get the bls12381 decompress program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn bls12381_decompress_program() -> Program {
        Program::from(BLS12381_DECOMPRESS_ELF).unwrap()
    }

    /// Get the secp256r1 add program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn secp256r1_add_program() -> Program {
        Program::from(SECP256R1_ADD_ELF).unwrap()
    }

    /// Get the secp256r1 double program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn secp256r1_double_program() -> Program {
        Program::from(SECP256R1_DOUBLE_ELF).unwrap()
    }

    /// Get the u256x2048 mul program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn u256xu2048_mul_program() -> Program {
        Program::from(U256XU2048_MUL_ELF).unwrap()
    }

    /// Get the uint256 mul program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn uint256_mul_program() -> Program {
        Program::from(UINT256_MUL_ELF).unwrap()
    }

    /// Get the poseidon2 permute program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn poseidon2_permute_program() -> Program {
        Program::from(POSEIDON2_PERMUTE_ELF).unwrap()
    }

    /// Get the SSZ withdrawals program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn ssz_withdrawals_program() -> Program {
        Program::from(KECCAK_SPONGE_ELF).unwrap()
    }

    /// Get the unconstrained program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn unconstrained_program() -> Program {
        Program::from(UNCONSTRAINED_ELF).unwrap()
    }

    /// Get the panic program.
    ///
    /// # Panics
    ///
    /// This function will panic if the program fails to load.
    #[must_use]
    pub fn panic_program() -> Program {
        Program::from(PANIC_ELF).unwrap()
    }

    #[must_use]
    #[allow(clippy::unreadable_literal)]
    pub fn simple_memory_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0x12348765, false, true),
            // SW and LW
            Instruction::new(Opcode::SW, 29, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LW, 28, 0, 0x27654320, false, true),
            // LBU
            Instruction::new(Opcode::LBU, 27, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LBU, 26, 0, 0x27654321, false, true),
            Instruction::new(Opcode::LBU, 25, 0, 0x27654322, false, true),
            Instruction::new(Opcode::LBU, 24, 0, 0x27654323, false, true),
            // LB
            Instruction::new(Opcode::LB, 23, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LB, 22, 0, 0x27654321, false, true),
            // LHU
            Instruction::new(Opcode::LHU, 21, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LHU, 20, 0, 0x27654322, false, true),
            // LH:
            Instruction::new(Opcode::LH, 19, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LH, 18, 0, 0x27654322, false, true),
            // SB
            Instruction::new(Opcode::ADD, 17, 0, 0x38276525, false, true),
            // Save the value 0x12348765 into address 0x43627530
            Instruction::new(Opcode::SW, 29, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627530, false, true),
            Instruction::new(Opcode::LW, 16, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627531, false, true),
            Instruction::new(Opcode::LW, 15, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627532, false, true),
            Instruction::new(Opcode::LW, 14, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SB, 17, 0, 0x43627533, false, true),
            Instruction::new(Opcode::LW, 13, 0, 0x43627530, false, true),
            // SH
            // Save the value 0x12348765 into address 0x43627530
            Instruction::new(Opcode::SW, 29, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SH, 17, 0, 0x43627530, false, true),
            Instruction::new(Opcode::LW, 12, 0, 0x43627530, false, true),
            Instruction::new(Opcode::SH, 17, 0, 0x43627532, false, true),
            Instruction::new(Opcode::LW, 11, 0, 0x43627530, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    pub fn other_memory_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, (1 << 20) + (1 << 15) + (1 << 6) - 1, false, true),
            Instruction::new(Opcode::ADD, 27, 0, 25, false, true),
            Instruction::new(
                Opcode::ADD,
                25,
                0,
                (1 << 28) + (1 << 12) + (1 << 18) - 1,
                false,
                true,
            ),
            Instruction::new(Opcode::ADD, 17, 0, 0x43627530, false, true),
            Instruction::new(Opcode::ADD, 22, 0, 22, false, true),
            Instruction::new(Opcode::ADD, 10, 0, 15, false, true),
            Instruction::new(Opcode::LWR, 29, 27, 1, false, true),
            Instruction::new(Opcode::LWL, 29, 27, 1, false, true),
            Instruction::new(Opcode::LL, 29, 27, 3, false, true),
            Instruction::new(Opcode::ADD, 15, 0, (1 << 20) + (1 << 15) + (1 << 6) - 1, false, true),
            Instruction::new(Opcode::SWL, 15, 22, 2, false, true),
            Instruction::new(Opcode::SWR, 15, 22, 2, false, true),
            Instruction::new(Opcode::SWL, 26, 10, 2, false, true),
            Instruction::new(Opcode::SWR, 26, 10, 2, false, true),
            Instruction::new(Opcode::LWR, 29, 27, 0, false, true),
            Instruction::new(Opcode::LWL, 29, 27, 0, false, true),
        ];
        Program::new(instructions, 0, 0)
    }
}
