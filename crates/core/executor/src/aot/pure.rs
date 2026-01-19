use std::sync::Arc;

use crate::aot::common::*;
use crate::aot::{get_address_space, get_pc, set_pc, AotCompiler, AotError};
use crate::{Instruction, Opcode, Program, Register};

impl AotCompiler {
    /// Create a new AOT instance for the given program.
    pub fn new(program: Program) -> Self {
        let program = Arc::new(program);
        Self { program }
    }

    pub fn create_pure_asm(&self) -> Result<String, AotError> {
        let mut asm = String::new();

        // header part
        asm += ".intel_syntax noprefix\n";
        asm += ".code64\n";
        asm += ".section .text\n";
        asm += ".global asm_run\n";

        // asm_run_internal part
        asm += "asm_run:\n";

        asm += &Self::push_external_registers();

        asm += &format!("   mov {REG_EXEC_STATE_PTR}, {REG_FIRST_ARG}\n");
        asm += &format!("   mov {REG_INSTRET_END}, {REG_SECOND_ARG}\n");

        let get_pc_ptr = format!("{:p}", get_pc as *const ());
        let get_address_space_ptr = format!("{:p}", get_address_space as *const ());

        asm += &Self::push_internal_registers();

        // Store the start of memory address space in r15
        asm += &format!("    mov r14, {get_address_space_ptr}\n");
        asm += "    mov rdi, rbx\n";
        asm += "    mov rsi, 1\n";
        asm += "    call r14\n";
        asm += "    mov r15, rax\n";
        // Store the start of register address space in high 64 bits of xmm0
        asm += "    mov rdi, rbx\n";
        asm += "    mov rsi, 0\n";
        asm += "    call r14\n";
        asm += "    pinsrq  xmm0, rax, 1\n";
        // Store the pointer to where `pc` is stored in the state in high 64 bits of xmm3
        asm += "    mov rdi, rbx\n";
        asm += &format!("   mov {REG_D}, {get_pc_ptr}\n");
        asm += &format!("   call {REG_D}\n");
        asm += "    pinsrq  xmm3, rax, 1\n"; // write `eax` to the third lane of xmm3

        asm += &Self::pop_internal_registers();

        asm += &Self::mips_regs_to_xmm();

        asm += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm += &format!("   pextrq {REG_A}, xmm3, 1\n"); // extract the upper 64 bits of the xmm3 register to REG_A
        asm += &format!("   movsxd {REG_A}, [{REG_C} + {REG_A}]\n");
        asm += &format!("   add {REG_A}, {REG_C}\n");
        asm += &format!("   jmp {REG_A}\n");

        for i in 0..(self.program.pc_base / 4) {
            asm += &format!("asm_execute_pc_{}:", i * 4);
            asm += "\n";
        }

        for (i, instruction) in self.program.instructions.iter().enumerate() {
            let pc = self.program.pc(i);
            asm += &format!("asm_execute_pc_{pc}:\n");

            asm += &(Self::generate_instruction_asm(instruction, pc)?);

            asm += &format!("    dec {REG_INSTRET_END}\n");
            asm += &format!("    cmp {REG_INSTRET_END}, 0\n");
            asm += &format!("    je asm_run_end_{pc}\n");
        }

        let set_pc_ptr = format!("{:p}", set_pc as *const ());

        // asm_run_end part
        for i in 0..self.program.instructions.len() {
            let pc = self.program.pc(i);
            let next_pc = pc + 4;
            asm += &format!("asm_run_end_{pc}:\n");
            asm += &Self::xmm_to_mips_regs();
            asm += &format!("    mov {REG_FIRST_ARG}, rbx\n");
            asm += &format!("    mov {REG_SECOND_ARG}, {next_pc}\n");
            asm += &format!("    mov {REG_D}, {set_pc_ptr}\n");
            asm += &format!("    call {REG_D}\n");
            asm += &Self::pop_external_registers();
            asm += &format!("    xor {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
            asm += "    ret\n";
        }

        // map_pc_base part
        asm += ".section .rodata\n";
        asm += "map_pc_base:\n";

        for i in 0..(self.program.pc_base / 4) {
            asm += &format!("   .long asm_execute_pc_{} - map_pc_base\n", i * 4);
        }

        for i in 0..self.program.instructions.len() {
            let pc = self.program.pc(i);
            asm += &format!("   .long asm_execute_pc_{pc} - map_pc_base\n");
        }

        Ok(asm)
    }

    fn generate_instruction_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        if instruction.is_alu_instruction() {
            return Self::generate_alu_asm(instruction, pc);
        }
        Ok(String::new())
    }

    fn generate_alu_asm(instruction: &Instruction, _pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();
        match instruction.opcode {
            Opcode::ADD | Opcode::SUB | Opcode::OR | Opcode::AND | Opcode::XOR | Opcode::MUL => {
                asm += &Self::generate_base_alu_asm(instruction)?;
            }
            Opcode::NOR => {
                asm += &Self::generate_nor_asm(instruction)?;
            }
            Opcode::SLL | Opcode::SRL | Opcode::SRA | Opcode::ROR => {
                asm += &Self::generate_shift_asm(instruction)?;
            }
            Opcode::MULT | Opcode::MULTU => {
                asm += &Self::generate_mult_asm(instruction)?;
            }
            Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU => {
                asm += &Self::generate_div_mod_asm(instruction)?;
            }
            Opcode::SLT | Opcode::SLTU => {
                asm += &Self::generate_slt_asm(instruction)?;
            }
            Opcode::CLO | Opcode::CLZ => {
                asm += &Self::generate_cloz_asm(instruction)?;
            }
            _ => return Err(AotError::NotSupported),
        }
        Ok(asm)
    }

    fn generate_base_alu_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

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
            asm += &delta_str_b;
            asm += &format!("   {asm_opcode} {gpr_reg_b}, {c}\n");
            asm += &gpr_to_xmm(&gpr_reg_b, a);
        } else if a == c as u8 {
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr(c as u8, REG_C_W, true);
            asm += &delta_str_c;
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, str_reg_a, true);
            asm += &delta_str_b;
            asm += &format!("   {asm_opcode} {gpr_reg_b}, {gpr_reg_c}\n");
            asm += &gpr_to_xmm(&gpr_reg_b, a);
        } else {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, str_reg_a, true);
            asm += &delta_str_b; // data is now in gpr_reg_b
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr(c as u8, REG_C_W, false); // data is in gpr_reg_c now
            asm += &delta_str_c; // have to get a return value here, since it modifies further registers too
            asm += &format!("   {asm_opcode} {gpr_reg_b}, {gpr_reg_c}\n");
            asm += &gpr_to_xmm(&gpr_reg_b, a);
        }

        Ok(asm)
    }

    fn generate_nor_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c;

        let str_reg_a = if MIPS_TO_X86_OVERRIDE_MAP[a as usize].is_some() {
            MIPS_TO_X86_OVERRIDE_MAP[a as usize].unwrap()
        } else {
            REG_A_W
        };

        let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, str_reg_a, true);
        asm += &delta_str_b; // data is now in gpr_reg_b
        let (gpr_reg_c, delta_str_c) = xmm_to_gpr(c as u8, REG_C_W, false); // data is in gpr_reg_c now
        asm += &delta_str_c; // have to get a return value here, since it modifies further registers too
        asm += &format!("   or {gpr_reg_b}, {gpr_reg_c}\n");
        asm += &format!("   not {gpr_reg_b}\n");
        asm += &gpr_to_xmm(&gpr_reg_b, a);

        Ok(asm)
    }

    fn generate_shift_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c;

        let str_reg_a = if MIPS_TO_X86_OVERRIDE_MAP[a as usize].is_some() {
            MIPS_TO_X86_OVERRIDE_MAP[a as usize].unwrap()
        } else {
            REG_A_W
        };

        if instruction.imm_c {
            let asm_opcode = match instruction.opcode {
                Opcode::SLL => "shl",
                Opcode::SRL => "shr",
                Opcode::SRA => "sar",
                Opcode::ROR => "ror",
                _ => return Err(AotError::NotSupported),
            };

            let (reg_b, delta_str_b) = &xmm_to_gpr(b, str_reg_a, true);
            asm += delta_str_b;
            asm += &format!("   {asm_opcode} {reg_b}, {c}\n");
            asm += &gpr_to_xmm(reg_b, a);
        } else if instruction.opcode == Opcode::ROR {
            let (reg_b, delta_str_b) = &xmm_to_gpr(b, str_reg_a, true);
            asm += delta_str_b;

            let (_reg_c, delta_str_c) = &xmm_to_gpr(c as u8, "ecx", true);
            asm += delta_str_c;

            asm += &format!("   ror {reg_b}, cl\n");

            asm += &gpr_to_xmm(reg_b, a);
        } else {
            let asm_opcode = match instruction.opcode {
                Opcode::SLL => "shlx",
                Opcode::SRL => "shrx",
                Opcode::SRA => "sarx",
                _ => return Err(AotError::NotSupported),
            };

            let (reg_b, delta_str_b) = &xmm_to_gpr(b, REG_B_W, false);
            // after this force write, we set [a:4]_1 <- [b:4]_1
            asm += delta_str_b;

            let (reg_c, delta_str_c) = &xmm_to_gpr(c as u8, REG_C_W, false);
            asm += delta_str_c;

            asm += &format!("   {asm_opcode} {str_reg_a}, {reg_b}, {reg_c}\n");

            asm += &gpr_to_xmm(str_reg_a, a);
        }

        Ok(asm)
    }

    fn generate_mult_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let b = instruction.op_b as u8;
        let c = instruction.op_c as u8;

        // for implicit multiplication, we need to load the multiplicand into `eax`
        // result of hi bits are always stored in `edx`
        // can't use REG_C_W, because it is edx, and it gets overridden
        let (_, delta_str_b) = &xmm_to_gpr(b, "eax", true);
        let (gpr_reg_c, delta_str_c) = &xmm_to_gpr(c, REG_A_W, false);
        asm += delta_str_b;
        asm += delta_str_c;

        match instruction.opcode {
            Opcode::MULT => {
                asm += &format!("   imul {gpr_reg_c}\n");
            }
            Opcode::MULTU => {
                asm += &format!("   mul {gpr_reg_c}\n");
            }
            _ => return Err(AotError::NotSupported),
        }

        asm += &gpr_to_xmm("edx", Register::HI as u8);
        asm += &gpr_to_xmm("eax", Register::LO as u8);

        Ok(asm)
    }

    fn generate_div_mod_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c as u8;

        // Calculate the result. Inputs: eax, ecx. Outputs: eax, edx.
        // Note that for div we are tied to eax/edx because of idiv requirements

        let (_, delta_str_b) = &xmm_to_gpr(b, "eax", true);
        asm += delta_str_b;
        let (reg_c, delta_str_c) = &xmm_to_gpr(c, REG_A_W, false);
        asm += delta_str_c;
        asm += "   mov edx, 0\n";

        match instruction.opcode {
            Opcode::DIV => {
                // sign-extend EAX into EDX:EAX
                asm += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm += &format!("   idiv {reg_c}\n");

                asm += &gpr_to_xmm("edx", Register::HI as u8);
                asm += &gpr_to_xmm("eax", Register::LO as u8);
            }
            Opcode::DIVU => {
                asm += &format!("   div {reg_c}\n");

                asm += &gpr_to_xmm("edx", Register::HI as u8);
                asm += &gpr_to_xmm("eax", Register::LO as u8);
            }
            Opcode::MOD => {
                // sign-extend EAX into EDX:EAX
                asm += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm += &format!("   idiv {reg_c}\n");

                asm += &gpr_to_xmm("edx", a);
            }
            Opcode::MODU => {
                asm += &format!("   div {reg_c}\n");

                asm += &gpr_to_xmm("edx", a);
            }
            _ => return Err(AotError::NotSupported),
        }

        Ok(asm)
    }

    fn generate_slt_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c as u8;

        if instruction.imm_c {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, REG_B_W, false);
            asm += &delta_str_b;
            asm += &format!("   cmp {gpr_reg_b}, {c}\n");
            match instruction.opcode {
                Opcode::SLT => asm += "   setl al\n",
                Opcode::SLTU => asm += "   setb al\n",
                _ => return Err(AotError::NotSupported),
            }
            asm += "   movzx eax, al\n";
            asm += &gpr_to_xmm("eax", a);
        } else {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, REG_B_W, false);
            asm += &delta_str_b;
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr(c, REG_C_W, false);
            asm += &delta_str_c;
            asm += &format!("   cmp {gpr_reg_b}, {gpr_reg_c}\n");
            match instruction.opcode {
                Opcode::SLT => asm += "   setl al\n",
                Opcode::SLTU => asm += "   setb al\n",
                _ => return Err(AotError::NotSupported),
            }
            asm += "   movzx eax, al\n";
            asm += &gpr_to_xmm("eax", a);
        }

        Ok(asm)
    }

    fn generate_cloz_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;

        let str_reg_a = if MIPS_TO_X86_OVERRIDE_MAP[a as usize].is_some() {
            MIPS_TO_X86_OVERRIDE_MAP[a as usize].unwrap()
        } else {
            REG_A_W
        };

        let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, str_reg_a, a != b);
        asm += &delta_str_b;

        if instruction.opcode == Opcode::CLO {
            asm += &format!("   not {gpr_reg_b}\n");
        }

        asm += &format!("   lzcnt {gpr_reg_b}, {gpr_reg_b}\n");
        asm += &gpr_to_xmm(&gpr_reg_b, a);

        Ok(asm)
    }
}
