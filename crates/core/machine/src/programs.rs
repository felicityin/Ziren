//! MIPS ELFs used for testing.

#[allow(dead_code)]
#[allow(missing_docs)]
pub mod tests {
    use zkm_core_executor::{Instruction, Opcode, Program};

    use test_artifacts::{
        FIBONACCI_ELF, HELLO_WORLD_ELF, KECCAK_SPONGE_ELF, MAX_MEMORY_ELF, PANIC_ELF,
        SECP256R1_ADD_ELF, SECP256R1_DOUBLE_ELF, SHA3_CHAIN_ELF, U256XU2048_MUL_ELF,
        UNCONSTRAINED_ELF,
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
    /// triggers a given bug.
    #[must_use]
    pub fn halt_only_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 2, 0, 0, false, true), // v0 = 0 (HALT syscall id)
            Instruction::new(Opcode::ADD, 4, 0, 0, false, true), // a0 = 0 (exit code)
            Instruction::new(Opcode::SYSCALL, 2, 4, 5, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// A straight-line program of `num_adds` repeated `ADD`s accumulating into one register, with
    /// no control flow -- used to force many small shards (via a small `ZKMCoreOpts::shard_size`)
    /// while keeping every shard's contents (and register-access pattern) as simple as possible.
    ///
    /// Starts at pc `4`, not `0`: `MinimalRunner::try_next_chunk` treats `pc == 0` as "already
    /// halted" (matching `Executor::execute`'s own `done` check), so a program starting at pc `0`
    /// never executes a single instruction under `prove_with_context`'s producer pipeline.
    #[must_use]
    pub fn many_adds_program(num_adds: usize) -> Program {
        let instructions =
            vec![Instruction::new(Opcode::ADD, 29, 29, 1, false, true); num_adds];
        Program::new(instructions, 4, 4)
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
            Instruction::new(Opcode::ADD, 27, 0, 3325, false, true),
            Instruction::new(
                Opcode::ADD,
                25,
                0,
                (1 << 28) + (1 << 12) + (1 << 18) - 1,
                false,
                true,
            ),
            Instruction::new(Opcode::ADD, 17, 0, 0x43627530, false, true),
            Instruction::new(Opcode::ADD, 22, 0, 80, false, true),
            Instruction::new(Opcode::ADD, 10, 0, 100, false, true),
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
