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

    /// A synthetic program exercising real register-form `add $zero, ...`/`sub $zero, ...` --
    /// i.e. `AluX0Chip`'s shape (see its doc comment). Real compiled code essentially never emits
    /// these (unlike `SYNC`/`Pref`, which real code does), so this is `AluX0Chip`'s only
    /// end-to-end coverage.
    #[must_use]
    pub fn add_sub_x0_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::ADD, 0, 30, 29, false, false),
            Instruction::new(Opcode::SUB, 0, 30, 29, false, false),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// A synthetic program exercising real register-form `slt $zero, ...`/`sltu $zero, ...` --
    /// i.e. `AluX0Chip`'s extended shape (see its doc comment) -- alongside real register-form
    /// SLT/SLTU (`LtChip`) and immediate-form SLTI/SLTIU (`SltiChip`). `slt`/`sltu`-to-`$zero` are
    /// as rare in real compiled code as `add $zero,...`, so this is `AluX0Chip`'s only end-to-end
    /// coverage for these two opcodes.
    #[must_use]
    pub fn slt_x0_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::SLT, 31, 29, 30, false, false),
            Instruction::new(Opcode::SLTU, 31, 30, 29, false, false),
            Instruction::new(Opcode::SLT, 31, 29, 10, false, true),
            Instruction::new(Opcode::SLTU, 31, 29, 10, false, true),
            Instruction::new(Opcode::SLT, 0, 29, 30, false, false),
            Instruction::new(Opcode::SLTU, 0, 30, 29, false, false),
        ];
        Program::new(instructions, 0, 0)
    }

    /// A synthetic program exercising real `sw $zero, ...` (`StoreWordChip`'s `op_a_0` pass-through
    /// -- a common real-code idiom for zeroing memory) and real `lw $zero, ...` (`LoadX0Chip`'s
    /// only shape, routed there instead of `LoadWordChip`). `lw $zero,...` is as rare in real
    /// compiled code as `add $zero,...`, so this is `LoadX0Chip`'s only end-to-end coverage.
    #[must_use]
    pub fn lw_sw_x0_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0x1000, false, true),
            Instruction::new(Opcode::ADD, 8, 0, 42, false, true),
            Instruction::new(Opcode::SW, 8, 29, 0, false, true),
            Instruction::new(Opcode::LW, 9, 29, 0, false, true),
            Instruction::new(Opcode::SW, 0, 29, 4, false, true),
            Instruction::new(Opcode::LW, 0, 29, 0, false, true),
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
