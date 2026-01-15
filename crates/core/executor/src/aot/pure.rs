use std::sync::Arc;

use crate::aot::common::*;
use crate::aot::{get_address_space, get_pc, set_pc, AotCompiler, AotError};
use crate::{Instruction, Opcode, Program};

impl AotCompiler {
    /// Create a new AOT instance for the given program.
    pub fn new(program: Program) -> Self {
        let program = Arc::new(program);
        Self { program }
    }

    pub fn create_pure_asm(&self) -> Result<String, AotError> {
        let mut asm_str = String::new();

        // header part
        asm_str += ".intel_syntax noprefix\n";
        asm_str += ".code64\n";
        asm_str += ".section .text\n";
        asm_str += ".global asm_run\n";

        // asm_run_internal part
        asm_str += "asm_run:\n";

        asm_str += &Self::push_external_registers();

        asm_str += &format!("   mov {REG_EXEC_STATE_PTR}, {REG_FIRST_ARG}\n");
        asm_str += &format!("   mov {REG_INSTRET_END}, {REG_SECOND_ARG}\n");

        let get_pc_ptr = format!("{:p}", get_pc as *const ());
        let get_address_space_ptr = format!("{:p}", get_address_space as *const ());

        asm_str += &Self::push_internal_registers();

        // Store the start of memory address space in r15
        asm_str += &format!("    mov r14, {get_address_space_ptr}\n");
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 1\n";
        asm_str += "    call r14\n";
        asm_str += "    mov r15, rax\n";
        // Store the start of register address space in high 64 bits of xmm0
        asm_str += "    mov rdi, rbx\n";
        asm_str += "    mov rsi, 0\n";
        asm_str += "    call r14\n";
        asm_str += "    pinsrq  xmm0, rax, 1\n";
        // Store the pointer to where `pc` is stored in the state in high 64 bits of xmm3
        asm_str += "    mov rdi, rbx\n";
        asm_str += &format!("   mov {REG_D}, {get_pc_ptr}\n");
        asm_str += &format!("   call {REG_D}\n");
        asm_str += "    pinsrq  xmm3, rax, 1\n"; // write `eax` to the third lane of xmm3

        asm_str += &Self::pop_internal_registers();

        asm_str += &Self::mips_regs_to_xmm();

        asm_str += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm_str += &format!("   pextrq {REG_A}, xmm3, 1\n"); // extract the upper 64 bits of the xmm3 register to REG_A
        asm_str += &format!("   movsxd {REG_A}, [{REG_C} + {REG_A}]\n");
        asm_str += &format!("   add {REG_A}, {REG_C}\n");
        asm_str += &format!("   jmp {REG_A}\n");

        for i in 0..(self.program.pc_base / 4) {
            asm_str += &format!("asm_execute_pc_{}:", i * 4);
            asm_str += "\n";
        }

        for (i, instruction) in self.program.instructions.iter().enumerate() {
            let pc = self.program.pc(i);
            asm_str += &format!("asm_execute_pc_{pc}:\n");

            asm_str += &(self.generate_instruction_asm(instruction, pc)?);

            asm_str += &format!("    dec {REG_INSTRET_END}\n");
            asm_str += &format!("    cmp {REG_INSTRET_END}, 0\n");
            asm_str += &format!("    je asm_run_end_{pc}\n");
        }

        let set_pc_ptr = format!("{:p}", set_pc as *const ());

        // asm_run_end part
        for i in 0..self.program.instructions.len() {
            let pc = self.program.pc(i);
            let next_pc = pc + 4;
            asm_str += &format!("asm_run_end_{pc}:\n");
            asm_str += &Self::xmm_to_mips_regs();
            asm_str += &format!("    mov {REG_FIRST_ARG}, rbx\n");
            asm_str += &format!("    mov {REG_SECOND_ARG}, {next_pc}\n");
            asm_str += &format!("    mov {REG_D}, {set_pc_ptr}\n");
            asm_str += &format!("    call {REG_D}\n");
            asm_str += &Self::pop_external_registers();
            asm_str += &format!("    xor {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
            asm_str += "    ret\n";
        }

        // map_pc_base part
        asm_str += ".section .rodata\n";
        asm_str += "map_pc_base:\n";

        for i in 0..(self.program.pc_base / 4) {
            asm_str += &format!("   .long asm_execute_pc_{} - map_pc_base\n", i * 4);
        }

        for i in 0..self.program.instructions.len() {
            let pc = self.program.pc(i);
            asm_str += &format!("   .long asm_execute_pc_{pc} - map_pc_base\n");
        }

        Ok(asm_str)
    }

    fn generate_instruction_asm(
        &self,
        instruction: &Instruction,
        pc: u32,
    ) -> Result<String, AotError> {
        if instruction.is_alu_instruction() {
            return self.generate_alu_asm(instruction, pc);
        }
        Ok(String::new())
    }

    fn generate_alu_asm(&self, instruction: &Instruction, _pc: u32) -> Result<String, AotError> {
        let mut asm_str = String::new();
        match instruction.opcode {
            Opcode::ADD | Opcode::SUB | Opcode::OR | Opcode::AND | Opcode::XOR | Opcode::MUL => {
                asm_str += &self.generate_base_alu_asm(instruction)?;
            }
            _ => return Err(AotError::NotSupported),
        }
        Ok(asm_str)
    }

    fn generate_base_alu_asm(&self, instruction: &Instruction) -> Result<String, AotError> {
        let mut asm_str = String::new();

        let asm_opcode = match instruction.opcode {
            Opcode::ADD => "add",
            Opcode::SUB => "sub",
            Opcode::AND => "and",
            Opcode::OR => "or",
            Opcode::XOR => "xor",
            Opcode::MUL => "imul",
            _ => return Err(AotError::NotSupported),
        };

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c;

        let str_reg_a = if MIPS_TO_X86_OVERRIDE_MAP[a as usize].is_some() {
            MIPS_TO_X86_OVERRIDE_MAP[a as usize].unwrap()
        } else {
            REG_A_W
        };

        if instruction.imm_c {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, str_reg_a, a != b);
            asm_str += &delta_str_b;
            asm_str += &format!("   {asm_opcode} {gpr_reg_b}, {c}\n");
            asm_str += &gpr_to_xmm(&gpr_reg_b, a);
        } else if a == c as u8 {
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr(c as u8, REG_C_W, true);
            asm_str += &delta_str_c;
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, str_reg_a, true);
            asm_str += &delta_str_b;
            asm_str += &format!("   {asm_opcode} {gpr_reg_b}, {gpr_reg_c}\n");
            asm_str += &gpr_to_xmm(&gpr_reg_b, a);
        } else {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, str_reg_a, true);
            asm_str += &delta_str_b; // data is now in gpr_reg_b
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr(c as u8, REG_C_W, false); // data is in gpr_reg_c now
            asm_str += &delta_str_c; // have to get a return value here, since it modifies further registers too
            asm_str += &format!("   {asm_opcode} {gpr_reg_b}, {gpr_reg_c}\n");
            asm_str += &gpr_to_xmm(&gpr_reg_b, a);
        }

        Ok(asm_str)
    }
}
