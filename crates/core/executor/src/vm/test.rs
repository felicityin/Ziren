use crate::{programs::tests::{
    fibonacci_program, max_memory_program, panic_program, secp256r1_add_program,
    secp256r1_double_program, simple_memory_program, simple_program, ssz_withdrawals_program,
    u256xu2048_mul_program,
}, vm::SimpleExecutor};
use zkm_stark::ZKMCoreOpts;

use crate::{Instruction, Opcode, Register, Program};

#[test]
fn test_add2() {
    // main:
    //     addi x29, x0, 5
    //     addi x30, x0, 37
    //     add RA, x30, x29
    let instructions = vec![
        Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
        Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
        Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
    ];

    let program = Program::new(instructions, 0, 0);

    let mut runtime = SimpleExecutor::new(program);
    runtime.run().unwrap();
    // assert_eq!(runtime.register(Register::RA as usize), 84);
}
