//! MIPS ELFs used for testing.

#[allow(dead_code)]
#[allow(missing_docs)]
pub mod tests {
    use crate::{Instruction, Opcode, Program};

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
            // 17: 0x38276525
            // Save the value 0x6525 into the lower 16 bits of address 0x43627530
            Instruction::new(Opcode::SH, 17, 0, 0x43627530, false, true),
            // 10: 0x12346525
            Instruction::new(Opcode::LW, 10, 0, 0x43627530, false, true),
            // Save the value 0x6525 into the lower 16 bits of address 0x43627532
            Instruction::new(Opcode::SH, 17, 0, 0x43627532, false, true),
            // 11: 0x65256525
            Instruction::new(Opcode::LW, 11, 0, 0x43627530, false, true),
        ];
        Program::new(instructions, 0, 0)
    }

    pub fn unaligned_memory_program() -> Program {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0x12345678, false, true),
            // 0x10000000: 0x78
            // 0x10000001: 0x56
            // 0x10000002: 0x34
            // 0x10000003: 0x12
            Instruction::new(Opcode::SW, 29, 0, 0x10000000, false, true),
            // ========== Test LWL ==========
            Instruction::new(Opcode::ADD, 28, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWL, 28, 0, 0x10000001, false, true),
            // Expected result: $28 = 0x5678ccdd
            Instruction::new(Opcode::ADD, 27, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWL, 27, 0, 0x10000002, false, true),
            // Expected result: $27 = 0x345678dd
            Instruction::new(Opcode::ADD, 26, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWL, 26, 0, 0x10000003, false, true),
            // Expected result: $26 = 0x12345678
            Instruction::new(Opcode::ADD, 25, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWL, 25, 0, 0x10000000, false, true),
            // Expected result: $25 = 0x78bbccdd

            // ========== Test LWR ==========
            Instruction::new(Opcode::ADD, 24, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWR, 24, 0, 0x10000001, false, true),
            // Expected result: $24 = 0xaa123456
            Instruction::new(Opcode::ADD, 23, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWR, 23, 0, 0x10000002, false, true),
            // Expected result: $23 = 0xaabb1234
            Instruction::new(Opcode::ADD, 22, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWR, 22, 0, 0x10000003, false, true),
            // Expected result: $22 = 0xaabbcc12
            Instruction::new(Opcode::ADD, 21, 0, 0xAABBCCDD, false, true),
            Instruction::new(Opcode::LWR, 21, 0, 0x10000000, false, true),
            // Expected result: $21 = 0x12345678

            // ========== Test SWL ==========

            // 29: 0x12345678
            // 28: 0x5678ccdd
            Instruction::new(Opcode::SW, 29, 0, 0x11000000, false, true),
            Instruction::new(Opcode::SWL, 28, 0, 0x11000001, false, true),
            // Expected result: $0x11000000 = 0x12345678
            Instruction::new(Opcode::SW, 29, 0, 0x12000000, false, true),
            Instruction::new(Opcode::SWL, 28, 0, 0x12000002, false, true),
            // Expected result: $0x12000000 = 0x125678cc
            Instruction::new(Opcode::SW, 29, 0, 0x13000000, false, true),
            Instruction::new(Opcode::SWL, 28, 0, 0x13000003, false, true),
            // Expected result: $0x13000000 = 0x5678ccdd
            Instruction::new(Opcode::SW, 29, 0, 0x14000000, false, true),
            Instruction::new(Opcode::SWL, 28, 0, 0x14000000, false, true),
            // Expected result: $0x14000000 = 0x12345656

            // ========== Test SWR ==========

            // 29: 0x12345678
            // 28: 0x5678ccdd
            Instruction::new(Opcode::SW, 29, 0, 0x15000000, false, true),
            Instruction::new(Opcode::SWR, 28, 0, 0x15000001, false, true),
            // Expected result: $0x15000000 = 0x78ccdd78
            Instruction::new(Opcode::SW, 29, 0, 0x16000000, false, true),
            Instruction::new(Opcode::SWR, 28, 0, 0x16000002, false, true),
            // Expected result: $0x16000000 = 0xccdd5678
            Instruction::new(Opcode::SW, 29, 0, 0x17000000, false, true),
            Instruction::new(Opcode::SWR, 28, 0, 0x17000003, false, true),
            // Expected result: $0x17000000 = 0xdd345678
            Instruction::new(Opcode::SW, 29, 0, 0x18000000, false, true),
            Instruction::new(Opcode::SWR, 28, 0, 0x18000000, false, true),
            // Expected result: $0x18000000 = 0x5678ccdd
        ];
        Program::new(instructions, 0, 0)
    }
}
