mod test;

use crate::aot::common::*;
use crate::aot::{get_address_space, get_pc, set_pc, AotCompiler, AotError};
use crate::syscalls::{SyscallCode, SyscallContext};
use crate::{Executor, Instruction, Opcode, Register};

impl AotCompiler {
    pub fn create_pure_asm(&self) -> Result<String, AotError> {
        let mut asm = String::new();

        // header part
        asm += ".intel_syntax noprefix\n";
        asm += ".code64\n";
        asm += ".section .text\n";
        asm += ".global asm_run\n";

        // asm_run_internal part
        asm += "asm_run:\n";

        asm += "    # push_external_registers\n";
        asm += &Self::push_external_registers();

        asm += "    # get params\n";
        asm += &format!("    mov {REG_EXECUTOR_PTR}, {REG_FIRST_ARG}\n");

        let get_pc_ptr = format!("{:p}", get_pc as *const ());
        let get_address_space_ptr = format!("{:p}", get_address_space as *const ());

        asm += "    # push_internal_registers\n";
        asm += &Self::push_internal_registers();

        // Store the start of memory address space in r15
        // asm += "    # Store the start of memory address space in r15\n";
        asm += &format!("    mov {REG_CALLER}, {get_address_space_ptr}\n");
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_SECOND_ARG}, 1\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += &format!("    mov {REG_MEMORY_PTR}, {REG_RETURN_VAL}\n");
        // Store the start of register address space in high 64 bits of xmm0
        asm += "    # Store the start of register address space in high 64 bits of xmm0\n";
        asm += &format!("    mov {REG_CALLER}, {get_address_space_ptr}\n");
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_SECOND_ARG}, 0\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += &format!("    pinsrq  xmm0, {REG_RETURN_VAL}, 1\n");
        // Store the pointer to where `pc` is stored in the state in high 64 bits of xmm1
        asm += "    # Store the pointer to where `pc` is stored in the state in high 64 bits of xmm1\n";
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_CALLER}, {get_pc_ptr}\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += &format!("    pinsrq  xmm1, {REG_RETURN_VAL}, 1\n"); // write `eax` to the third lane of xmm1
        asm += &format!("    mov {REG_NEXT_PC}, {REG_RETURN_VAL}\n");

        asm += "    # pop_internal_registers\n";
        asm += &Self::pop_internal_registers();

        asm += "    # load_xmm_regs\n";
        asm += &Self::load_xmm_regs();

        asm += "    # execute\n";
        asm += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm += &format!("   pextrq {REG_A}, xmm1, 1\n"); // extract the upper 64 bits of the xmm1 register to REG_A
        asm += &format!("   movsxd {REG_A}, [{REG_C} + {REG_A}]\n");
        asm += &format!("   add {REG_A}, {REG_C}\n");
        asm += &format!("   jmp {REG_A}\n");

        for i in 0..(self.program.pc_base / 4) {
            asm += &format!("asm_execute_pc_{}:", i * 4);
            asm += "\n";
        }

        let most_pc = self.program.pc_base + self.program.instructions.len() as u32 * 4;

        let mut i = 0;
        while i < self.program.instructions.len() {
            let pc = self.program.pc(i);
            let instruction = &self.program.instructions[i];
            asm += &format!("asm_execute_pc_{pc}:\n");

            // Check if we should suspend or not
            asm += &format!("    cmp {REG_NEXT_PC}, {most_pc}\n");
            asm += "    jae asm_run_end\n";
            asm += &format!("    mov {REG_NEXT_PC}, {}\n", pc + 4);
            i += 1;

            if instruction.is_branch_instruction() || instruction.is_jump_instruction() {
                // Processing the delay slot
                // Note that the processing order here differs from that in the executor
                // eg. The execution order of instructions in the executor is:
                //   jump       %x0        %x31       0
                //   sltu       %x2        %x1        1
                // But here:
                //   sltu       %x2        %x1        1
                //   jump       %x0        %x31       0
                let next_instruction = &self.program.instructions[i];
                let next_pc = self.program.pc(i);
                asm += &format!("asm_execute_pc_{next_pc}:\n");
                asm += &format!("    mov {REG_NEXT_PC}, {}\n", next_pc + 4);
                i += 1;
                asm += &(Self::generate_instruction_asm(next_instruction, next_pc)?);

                asm += &(Self::generate_instruction_asm(instruction, pc)?);
            } else {
                asm += &(Self::generate_instruction_asm(instruction, pc)?);
            }
        }

        let set_pc_ptr = format!("{:p}", set_pc as *const ());

        asm += "asm_run_end:\n";
        asm += "    # save_xmm_regs\n";
        asm += &Self::save_xmm_regs();
        asm += "    # call set_pc()\n";
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_SECOND_ARG}, {REG_NEXT_PC}\n");
        asm += &format!("    mov {REG_CALLER}, {set_pc_ptr}\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += "    # pop_external_registers\n";
        asm += &Self::pop_external_registers();
        asm += &format!("    xor {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += "    ret\n";

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

        std::fs::write("asm_pure_dump.s", &asm).expect("failed to write asm");

        Ok(asm)
    }

    fn generate_instruction_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        if instruction.is_alu_instruction() {
            return Self::generate_alu_asm(instruction, pc);
        } else if instruction.is_branch_instruction() {
            return Self::generate_branch_asm(instruction, pc);
        } else if instruction.is_jump_instruction() {
            return Self::generate_jump_asm(instruction, pc);
        } else if instruction.is_memory_load_instruction() {
            return Self::generate_memory_load_asm(instruction, pc);
        } else if instruction.is_memory_store_instruction() {
            return Self::generate_memory_store_asm(instruction, pc);
        } else if instruction.is_mov_cond_instruction() {
            return Self::generate_mov_cond_asm(instruction, pc);
        } else if instruction.is_misc_instruction() {
            return Self::generate_misc_asm(instruction, pc);
        } else if instruction.is_syscall_instruction() {
            return Self::generate_syscall_asm(instruction, pc);
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

        let extern_handler_ptr = format!("{:p}", execute_alu as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {_pc}\n");
        asm += &format!("   mov {REG_FOURTH_ARG}, {REG_CLK}\n");
        asm += &format!("   mov {REG_FIFTH_ARG}, {REG_GLOBAL_CLK}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        Ok(asm)
    }

    pub fn generate_base_alu_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let asm_opcode = match instruction.opcode {
            Opcode::ADD => "add",
            Opcode::SUB => "sub",
            Opcode::AND => "and",
            Opcode::OR => "or",
            Opcode::XOR => "xor",
            Opcode::MUL => "imul",
            _ => unreachable!(),
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

    pub fn generate_nor_asm(instruction: &Instruction) -> Result<String, AotError> {
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

    pub fn generate_shift_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c;

        let str_reg_a = if MIPS_TO_X86_OVERRIDE_MAP[a as usize].is_some() {
            MIPS_TO_X86_OVERRIDE_MAP[a as usize].unwrap()
        } else {
            REG_A_W
        };

        if instruction.imm_b {
            let asm_opcode = match instruction.opcode {
                Opcode::SLL => "shl",
                _ => unreachable!(),
            };

            asm += &format!("   mov {str_reg_a}, {}\n", instruction.op_b);
            asm += &format!("   {asm_opcode} {str_reg_a}, {c}\n");
            asm += &gpr_to_xmm(str_reg_a, a);
        } else if instruction.imm_c {
            let asm_opcode = match instruction.opcode {
                Opcode::SLL => "shl",
                Opcode::SRL => "shr",
                Opcode::SRA => "sar",
                Opcode::ROR => "ror",
                _ => unreachable!(),
            };

            let (reg_b, delta_str_b) = &xmm_to_gpr(b, str_reg_a, true);
            asm += delta_str_b;
            asm += &format!("   {asm_opcode} {reg_b}, {c}\n");
            asm += &gpr_to_xmm(reg_b, a);
        } else if instruction.opcode == Opcode::ROR {
            let (reg_b, delta_str_b) = &xmm_to_gpr(b, str_reg_a, true);
            asm += delta_str_b;

            let (_reg_c, delta_str_c) = &xmm_to_gpr(c as u8, REG_ROR_W, true);
            asm += delta_str_c;

            asm += &format!("   ror {reg_b}, {REG_ROR_8L}\n");

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

    pub fn generate_mult_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let b = instruction.op_b as u8;
        let c = instruction.op_c as u8;

        // for implicit multiplication, we need to load the multiplicand into `eax`
        // result of hi bits are always stored in `edx`
        // can't use REG_C_W, because it is edx, and it gets overridden
        let (_, delta_str_b) = &xmm_to_gpr(b, REG_LO, true);
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
            _ => unreachable!(),
        }

        asm += &gpr_to_xmm(REG_HI, Register::HI as u8);
        asm += &gpr_to_xmm(REG_LO, Register::LO as u8);

        Ok(asm)
    }

    pub fn generate_div_mod_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c as u8;

        // Calculate the result. Inputs: eax, ecx. Outputs: eax, edx.
        // Note that for div we are tied to eax/edx because of idiv requirements

        let (_, delta_str_b) = &xmm_to_gpr(b, REG_LO, true);
        asm += delta_str_b;
        let (reg_c, delta_str_c) = &xmm_to_gpr(c, REG_A_W, false);
        asm += delta_str_c;
        asm += &format!("   mov {REG_HI}, 0\n");

        match instruction.opcode {
            Opcode::DIV => {
                // sign-extend EAX into EDX:EAX
                asm += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm += &format!("   idiv {reg_c}\n");

                asm += &gpr_to_xmm(REG_HI, Register::HI as u8);
                asm += &gpr_to_xmm(REG_LO, Register::LO as u8);
            }
            Opcode::DIVU => {
                asm += &format!("   div {reg_c}\n");

                asm += &gpr_to_xmm(REG_HI, Register::HI as u8);
                asm += &gpr_to_xmm(REG_LO, Register::LO as u8);
            }
            Opcode::MOD => {
                // sign-extend EAX into EDX:EAX
                asm += "   cdq\n";
                // eax = eax / ecx, edx = eax % ecx
                asm += &format!("   idiv {reg_c}\n");

                asm += &gpr_to_xmm(REG_HI, a);
            }
            Opcode::MODU => {
                asm += &format!("   div {reg_c}\n");

                asm += &gpr_to_xmm(REG_HI, a);
            }
            _ => unreachable!(),
        }

        Ok(asm)
    }

    pub fn generate_slt_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let c = instruction.op_c as u8;

        if instruction.imm_c {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, REG_B_W, false);
            asm += &delta_str_b;
            asm += &format!("   cmp {gpr_reg_b}, {}\n", instruction.op_c);
            match instruction.opcode {
                Opcode::SLT => asm += &format!("   setl {REG_D_8L}\n"),
                Opcode::SLTU => asm += &format!("   setb {REG_D_8L}\n"),
                _ => unreachable!(),
            }
            asm += &format!("   movzx {REG_D_W}, {REG_D_8L}\n");
            asm += &gpr_to_xmm(REG_D_W, a);
        } else {
            let (gpr_reg_b, delta_str_b) = xmm_to_gpr(b, REG_B_W, false);
            asm += &delta_str_b;
            let (gpr_reg_c, delta_str_c) = xmm_to_gpr(c, REG_C_W, false);
            asm += &delta_str_c;
            asm += &format!("   cmp {gpr_reg_b}, {gpr_reg_c}\n");
            match instruction.opcode {
                Opcode::SLT => asm += &format!("   setl {REG_D_8L}\n"),
                Opcode::SLTU => asm += &format!("   setb {REG_D_8L}\n"),
                _ => unreachable!(),
            }
            asm += &format!("   movzx {REG_D_W}, {REG_D_8L}\n");
            asm += &gpr_to_xmm(REG_D_W, a);
        }

        Ok(asm)
    }

    pub fn generate_cloz_asm(instruction: &Instruction) -> Result<String, AotError> {
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

    pub fn generate_branch_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        let extern_handler_ptr = format!("{:p}", execute_branch as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_FOURTH_ARG}, {REG_CLK}\n");
        asm += &format!("   mov {REG_FIFTH_ARG}, {REG_GLOBAL_CLK}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let next_pc = pc + 4;
        let next_next_pc = next_pc.wrapping_add(instruction.op_c);

        let a = instruction.op_a;
        let b = instruction.op_b as u8;

        let (reg_a, delta_str_a) = &xmm_to_gpr(a, REG_A_W, false);
        asm += delta_str_a;

        if instruction.opcode == Opcode::BEQ || instruction.opcode == Opcode::BNE {
            let (reg_b, delta_str_b) = &xmm_to_gpr(b, REG_B_W, false);
            asm += delta_str_b;
            asm += &format!("   cmp {reg_a}, {reg_b}\n");
        } else {
            asm += &format!("   test {reg_a}, {reg_a}\n");
        }

        match instruction.opcode {
            Opcode::BEQ => {
                asm += &format!("   je asm_execute_pc_{next_next_pc}\n");
            }
            Opcode::BNE => {
                asm += &format!("   jne asm_execute_pc_{next_next_pc}\n");
            }
            Opcode::BLTZ => {
                asm += &format!("   js asm_execute_pc_{next_next_pc}\n");
            }
            Opcode::BLEZ => {
                asm += &format!("   jle asm_execute_pc_{next_next_pc}\n");
            }
            Opcode::BGTZ => {
                asm += &format!("   jg asm_execute_pc_{next_next_pc}\n");
            }
            Opcode::BGEZ => {
                asm += &format!("   jns asm_execute_pc_{next_next_pc}\n");
            }
            _ => unreachable!(),
        }

        Ok(asm)
    }

    pub fn generate_jump_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        let extern_handler_ptr = format!("{:p}", execute_jump as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_FOURTH_ARG}, {REG_CLK}\n");
        asm += &format!("   mov {REG_FIFTH_ARG}, {REG_GLOBAL_CLK}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let next_pc = pc + 4;
        let return_pc = next_pc + 4;

        let a = instruction.op_a;
        let b = instruction.op_b as u8;

        if a != 0 {
            asm += &format!("   mov {REG_A_W}, {return_pc}\n");
            asm += &gpr_to_xmm(REG_A_W, a);
        }

        match instruction.opcode {
            Opcode::Jump => {
                let (gpr_reg_b, delta_str_b) = &xmm_to_gpr(b, REG_B_W, false);
                asm += delta_str_b;

                let gpr_reg_b_64 = convert_x86_reg(gpr_reg_b, Width::W64).unwrap();
                asm += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
                asm += &format!("   movsxd {REG_A}, [{REG_C} + {gpr_reg_b_64}]\n");
                asm += &format!("   add {REG_A}, {REG_C}\n");
                asm += &format!("   jmp {REG_A}\n");
            }
            Opcode::Jumpi => {
                let target_pc = instruction.op_b;
                asm += &format!("   jmp asm_execute_pc_{target_pc}\n");
            }
            Opcode::JumpDirect => {
                let (_gpr_reg_b, delta_str_b) = &xmm_to_gpr(b, REG_B_W, false);
                asm += delta_str_b;

                let target_pc = next_pc + instruction.op_b;
                asm += &format!("   jmp asm_execute_pc_{target_pc}\n");
            }
            _ => unreachable!(),
        }

        Ok(asm)
    }

    pub fn generate_memory_load_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        let extern_handler_ptr = format!("{:p}", execute_memory_load as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_FOURTH_ARG}, {REG_CLK}\n");
        asm += &format!("   mov {REG_FIFTH_ARG}, {REG_GLOBAL_CLK}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let offset_ext = instruction.op_c;

        let (gpr_reg, delta_str) = xmm_to_gpr(b, REG_B_W, true);
        // gpr_reg = [b:4]_1
        asm += &delta_str;
        // REG_B_W = ptr = [b:4]_1 + offset_ext
        asm += &format!("   add {gpr_reg}, {offset_ext}\n");

        if instruction.opcode == Opcode::LWR || instruction.opcode == Opcode::LWL {
            asm += &format!("   mov {REG_D_W}, {gpr_reg}\n");
            asm += &format!("   and {gpr_reg}, 0xfffffffc\n");
        }

        if instruction.opcode == Opcode::LW || instruction.opcode == Opcode::LL {
            asm += &format!("   and {gpr_reg}, 0xfffffffc\n");
        }

        let gpr_reg_w64 =
            convert_x86_reg(&gpr_reg, Width::W64).ok_or(AotError::InvalidInstruction)?;
        assert_eq!(gpr_reg_w64, REG_B);

        // REG_B = REG_B + REG_AS2_PTR = <memory address in host memory>
        asm += &format!("   lea {gpr_reg_w64}, [{gpr_reg_w64} + {REG_MEMORY_PTR}]\n");

        let str_reg_a = if MIPS_TO_X86_OVERRIDE_MAP[a as usize].is_some() {
            MIPS_TO_X86_OVERRIDE_MAP[a as usize].unwrap()
        } else {
            REG_B_W
        };

        match instruction.opcode {
            Opcode::LB => {
                asm += &format!("   movsx {str_reg_a}, byte ptr [{gpr_reg_w64}]\n");
            }
            Opcode::LBU => {
                asm += &format!("   movzx {str_reg_a}, byte ptr [{gpr_reg_w64}]\n");
            }
            Opcode::LH => {
                asm += &format!("   movsx {str_reg_a}, word ptr [{gpr_reg_w64}]\n");
            }
            Opcode::LHU => {
                asm += &format!("   movzx {str_reg_a}, word ptr [{gpr_reg_w64}]\n");
            }
            Opcode::LW | Opcode::LL => {
                asm += &format!("   mov {str_reg_a}, [{gpr_reg_w64}]\n");
            }
            Opcode::LWR => {
                /*
                    vaddr = GPR[base] + sign_extend(offset)
                    aligned_addr = vaddr & ~3
                    aligned_word = MEM[aligned_addr, 4]
                    reg_old = GPR[rt]

                    offset = vaddr & 3
                    switch (offset) {
                        case 0: GPR[rt] = aligned_word# break;
                        case 1: GPR[rt] = (reg_old & 0xFF000000) | (aligned_word >> 8)# break;
                        case 2: GPR[rt] = (reg_old & 0xFFFF0000) | (aligned_word >> 16)# break;
                        case 3: GPR[rt] = (reg_old & 0xFFFFFF00) | (aligned_word >> 24)# break;
                    }
                */

                asm += &format!("   mov {str_reg_a}, [{gpr_reg_w64}]\n");

                let (gpr_reg_source, delta_str) = xmm_to_gpr(a, REG_C_W, false);
                asm += &delta_str;

                asm += &format!("   and {REG_D_W}, 3\n");
                asm += &format!("   cmp {REG_D_W}, 0\n");
                asm += &format!("   je .pc_{pc}_lwr_end\n");
                asm += &format!("   cmp {REG_D_W}, 1\n");
                asm += &format!("   je .pc_{pc}_lwr_case1\n");
                asm += &format!("   cmp {REG_D_W}, 2\n");
                asm += &format!("   je .pc_{pc}_lwr_case2\n");

                asm += &format!(".pc_{pc}_lwr_case3:\n");
                asm += &format!("   and {gpr_reg_source}, 0xFFFFFF00\n");
                asm += &format!("   shr {str_reg_a}, 24\n");
                asm += &format!("   or {str_reg_a}, {gpr_reg_source}\n");
                asm += &format!("   jmp .pc_{pc}_lwr_end\n");

                asm += &format!(".pc_{pc}_lwr_case2:\n");
                asm += &format!("   and {gpr_reg_source}, 0xFFFF0000\n");
                asm += &format!("   shr {str_reg_a}, 16\n");
                asm += &format!("   or {str_reg_a}, {gpr_reg_source}\n");
                asm += &format!("   jmp .pc_{pc}_lwr_end\n");

                asm += &format!(".pc_{pc}_lwr_case1:\n");
                asm += &format!("   and {gpr_reg_source}, 0xFF000000\n");
                asm += &format!("   shr {str_reg_a}, 8\n");
                asm += &format!("   or {str_reg_a}, {gpr_reg_source}\n");
                asm += &format!("   jmp .pc_{pc}_lwr_end\n");

                asm += &format!(".pc_{pc}_lwr_end:\n");
            }
            Opcode::LWL => {
                /*
                    vaddr = GPR[base] + sign_extend(offset)
                    aligned_addr = vaddr & ~3
                    aligned_word = MEM[aligned_addr, 4]
                    reg_old = GPR[rt]

                    offset = vaddr & 3
                    switch (offset) {
                        case 0: GPR[rt] = (reg_old & 0x00FFFFFF) | (aligned_word << 24)# break;
                        case 1: GPR[rt] = (reg_old & 0x0000FFFF) | (aligned_word << 16)# break;
                        case 2: GPR[rt] = (reg_old & 0x000000FF) | (aligned_word << 8)# break;
                        case 3: GPR[rt] = aligned_word# break;
                    }
                */
                asm += &format!("   mov {str_reg_a}, [{gpr_reg_w64}]\n");

                let (gpr_reg_source, delta_str) = xmm_to_gpr(a, REG_C_W, false);
                asm += &delta_str;

                asm += &format!("   and {REG_D_W}, 3\n");
                asm += &format!("   cmp {REG_D_W}, 0\n");
                asm += &format!("   je .pc_{pc}_lwl_case0\n");
                asm += &format!("   cmp {REG_D_W}, 1\n");
                asm += &format!("   je .pc_{pc}_lwl_case1\n");
                asm += &format!("   cmp {REG_D_W}, 2\n");
                asm += &format!("   je .pc_{pc}_lwl_case2\n");

                asm += &format!(".pc_{pc}_lwl_case3:\n");
                asm += &format!("   jmp .pc_{pc}_lwl_end\n");

                asm += &format!(".pc_{pc}_lwl_case2:\n");
                asm += &format!("   and {gpr_reg_source}, 0x000000FF\n");
                asm += &format!("   shl {str_reg_a}, 8\n");
                asm += &format!("   or {str_reg_a}, {gpr_reg_source}\n");
                asm += &format!("   jmp .pc_{pc}_lwl_end\n");

                asm += &format!(".pc_{pc}_lwl_case1:\n");
                asm += &format!("   and {gpr_reg_source}, 0x0000FFFF\n");
                asm += &format!("   shl {str_reg_a}, 16\n");
                asm += &format!("   or {str_reg_a}, {gpr_reg_source}\n");
                asm += &format!("   jmp .pc_{pc}_lwl_end\n");

                asm += &format!(".pc_{pc}_lwl_case0:\n");
                asm += &format!("   and {gpr_reg_source}, 0x00FFFFFF\n");
                asm += &format!("   shl {str_reg_a}, 24\n");
                asm += &format!("   or {str_reg_a}, {gpr_reg_source}\n");

                asm += &format!(".pc_{pc}_lwl_end:\n");
            }
            _ => unreachable!(),
        }

        asm += &gpr_to_mips_register(str_reg_a, a);

        Ok(asm)
    }

    pub fn generate_memory_store_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        let extern_handler_ptr = format!("{:p}", execute_memory_store as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_FOURTH_ARG}, {REG_CLK}\n");
        asm += &format!("   mov {REG_FIFTH_ARG}, {REG_GLOBAL_CLK}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let a = instruction.op_a;
        let b = instruction.op_b as u8;
        let offset_ext = instruction.op_c;

        let (gpr_reg, delta_str) = xmm_to_gpr(b, REG_B_W, true);
        // gpr_reg = [b:4]_1
        asm += &delta_str;
        // REG_B_W = ptr = [b:4]_1 + offset_ext
        asm += &format!("   add {gpr_reg}, {offset_ext}\n");

        if instruction.opcode == Opcode::SWR || instruction.opcode == Opcode::SWL {
            asm += &format!("   mov {REG_D_W}, {gpr_reg}\n");
            asm += &format!("   and {gpr_reg}, 0xfffffffc\n");
        }

        if instruction.opcode == Opcode::SW {
            asm += &format!("   and {gpr_reg}, 0xfffffffc\n");
        }

        let gpr_reg_w64 =
            convert_x86_reg(&gpr_reg, Width::W64).ok_or(AotError::InvalidInstruction)?;
        assert_eq!(gpr_reg_w64, REG_B);

        // REG_B = REG_B + REG_AS2_PTR = <memory address in host memory>
        asm += &format!("   lea {gpr_reg_w64}, [{gpr_reg_w64} + {REG_MEMORY_PTR}]\n");

        let (gpr_reg_source, delta_str) = xmm_to_gpr(a, REG_A_W, false);
        asm += &delta_str;

        match instruction.opcode {
            Opcode::SB => {
                let gpr_reg_source_w8l = convert_x86_reg(&gpr_reg_source, Width::W8L)
                    .ok_or(AotError::InvalidInstruction)?;
                asm += &format!("   mov byte ptr [{gpr_reg_w64}], {gpr_reg_source_w8l}\n");
            }
            Opcode::SH => {
                let gpr_reg_source_w16 = convert_x86_reg(&gpr_reg_source, Width::W16)
                    .ok_or(AotError::InvalidInstruction)?;
                asm += &format!("   mov word ptr [{gpr_reg_w64}], {gpr_reg_source_w16}\n");
            }
            Opcode::SW => {
                asm += &format!("   mov [{gpr_reg_w64}], {gpr_reg_source}\n");
            }
            Opcode::SC => {
                asm += &format!("   mov [{gpr_reg_w64}], {gpr_reg_source}\n");
                asm += &format!("   mov {gpr_reg_source}, 1\n");
                asm += &gpr_to_xmm(&gpr_reg_source, a);
            }
            Opcode::SWR => {
                /*
                    vaddr = GPR[base] + sign_extend(offset)
                    aligned_addr = vaddr & ~3
                    aligned_word = MEM[aligned_addr, 4]
                    reg_value = GPR[rt]

                    offset = vaddr & 3
                    switch (offset) {
                        case 0: MEM[aligned_addr] = reg_value# break;
                        case 1: MEM[aligned_addr] = (aligned_word & 0x000000FF) | (reg_value << 8)# break;
                        case 2: MEM[aligned_addr] = (aligned_word & 0x0000FFFF) | (reg_value << 16)# break;
                        case 3: MEM[aligned_addr] = (aligned_word & 0x00FFFFFF) | (reg_value << 24)# break;
                    }
                */

                asm += &format!("   and {REG_D_W}, 3\n");
                asm += &format!("   cmp {REG_D_W}, 0\n");
                asm += &format!("   je .pc_{pc}_swr_case0\n");
                asm += &format!("   cmp {REG_D_W}, 1\n");
                asm += &format!("   je .pc_{pc}_swr_case1\n");
                asm += &format!("   cmp {REG_D_W}, 2\n");
                asm += &format!("   je .pc_{pc}_swr_case2\n");

                asm += &format!(".pc_{pc}_swr_case3:\n");
                asm += &format!("   mov {REG_C_W}, [{gpr_reg_w64}]\n");
                asm += &format!("   and {REG_C_W}, 0x00FFFFFF\n");
                asm += &format!("   shl {gpr_reg_source}, 24\n");
                asm += &format!("   or {REG_C_W}, {gpr_reg_source}\n");
                asm += &format!("   mov [{gpr_reg_w64}], {REG_C_W}\n");
                asm += &format!("   jmp .pc_{pc}_swr_end\n");

                asm += &format!(".pc_{pc}_swr_case2:\n");
                asm += &format!("   mov {REG_C_W}, [{gpr_reg_w64}]\n");
                asm += &format!("   and {REG_C_W}, 0x0000FFFF\n");
                asm += &format!("   shl {gpr_reg_source}, 16\n");
                asm += &format!("   or {REG_C_W}, {gpr_reg_source}\n");
                asm += &format!("   mov [{gpr_reg_w64}], {REG_C_W}\n");
                asm += &format!("   jmp .pc_{pc}_swr_end\n");

                asm += &format!(".pc_{pc}_swr_case1:\n");
                asm += &format!("   mov {REG_C_W}, [{gpr_reg_w64}]\n");
                asm += &format!("   and {REG_C_W}, 0x000000FF\n");
                asm += &format!("   shl {gpr_reg_source}, 8\n");
                asm += &format!("   or {REG_C_W}, {gpr_reg_source}\n");
                asm += &format!("   mov [{gpr_reg_w64}], {REG_C_W}\n");
                asm += &format!("   jmp .pc_{pc}_swr_end\n");

                asm += &format!(".pc_{pc}_swr_case0:\n");
                asm += &format!("   mov [{gpr_reg_w64}], {gpr_reg_source}\n");

                asm += &format!(".pc_{pc}_swr_end:\n");
            }
            Opcode::SWL => {
                /*
                    vaddr = GPR[base] + sign_extend(offset)
                    aligned_addr = vaddr & ~3
                    aligned_word = MEM[aligned_addr, 4]
                    reg_value = GPR[rt]

                    offset = vaddr & 3
                    switch (offset) {
                        case 0: MEM[aligned_addr] = (aligned_word & 0xFFFFFF00) | (reg_value >> 24)# break;
                        case 1: MEM[aligned_addr] = (aligned_word & 0xFFFF0000) | (reg_value >> 16)# break;
                        case 2: MEM[aligned_addr] = (aligned_word & 0xFF000000) | (reg_value >> 8)# break;
                        case 3: MEM[aligned_addr] = reg_value# break;
                */

                asm += &format!("   and {REG_D_W}, 3\n");
                asm += &format!("   cmp {REG_D_W}, 0\n");
                asm += &format!("   je .pc_{pc}_swl_case0\n");
                asm += &format!("   cmp {REG_D_W}, 1\n");
                asm += &format!("   je .pc_{pc}_swl_case1\n");
                asm += &format!("   cmp {REG_D_W}, 2\n");
                asm += &format!("   je .pc_{pc}_swl_case2\n");

                asm += &format!(".pc_{pc}_swl_case3:\n");
                asm += &format!("   mov [{gpr_reg_w64}], {gpr_reg_source}\n");
                asm += &format!("   jmp .pc_{pc}_swl_end\n");

                asm += &format!(".pc_{pc}_swl_case2:\n");
                asm += &format!("   mov {REG_C_W}, [{gpr_reg_w64}]\n");
                asm += &format!("   and {REG_C_W}, 0xFF000000\n");
                asm += &format!("   shr {gpr_reg_source}, 8\n");
                asm += &format!("   or {REG_C_W}, {gpr_reg_source}\n");
                asm += &format!("   mov [{gpr_reg_w64}], {REG_C_W}\n");
                asm += &format!("   jmp .pc_{pc}_swl_end\n");

                asm += &format!(".pc_{pc}_swl_case1:\n");
                asm += &format!("   mov {REG_C_W}, [{gpr_reg_w64}]\n");
                asm += &format!("   and {REG_C_W}, 0xFFFF0000\n");
                asm += &format!("   shr {gpr_reg_source}, 16\n");
                asm += &format!("   or {REG_C_W}, {gpr_reg_source}\n");
                asm += &format!("   mov [{gpr_reg_w64}], {REG_C_W}\n");
                asm += &format!("   jmp .pc_{pc}_swl_end\n");

                asm += &format!(".pc_{pc}_swl_case0:\n");
                asm += &format!("   mov {REG_C_W}, [{gpr_reg_w64}]\n");
                asm += &format!("   and {REG_C_W}, 0xFFFFFF00\n");
                asm += &format!("   shr {gpr_reg_source}, 24\n");
                asm += &format!("   or {REG_C_W}, {gpr_reg_source}\n");
                asm += &format!("   mov [{gpr_reg_w64}], {REG_C_W}\n");

                asm += &format!(".pc_{pc}_swl_end:\n");
            }
            _ => unreachable!(),
        }

        Ok(asm)
    }

    fn generate_mov_cond_asm(instruction: &Instruction, _pc: u32) -> Result<String, AotError> {
        let extern_handler_ptr = format!("{:p}", execute_mov_cond as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);

        let mut asm = String::new();

        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers(); // pop the internal registers from the stack
        asm += &Self::pop_address_space_start();
        // read the memory from the memory location of the MIPS registers in `GuestMemory`
        // registers, to the appropriate XMM registers
        asm += &Self::load_xmm_regs();

        Ok(asm)
    }

    fn generate_misc_asm(instruction: &Instruction, _pc: u32) -> Result<String, AotError> {
        let extern_handler_ptr = match instruction.opcode {
            Opcode::MADDU => format!("{:p}", execute_maddu as *const ()),
            Opcode::MSUBU => format!("{:p}", execute_msubu as *const ()),
            Opcode::MADD => format!("{:p}", execute_madd as *const ()),
            Opcode::MSUB => format!("{:p}", execute_msub as *const ()),
            Opcode::WSBH => format!("{:p}", execute_wsbh as *const ()),
            Opcode::EXT => format!("{:p}", execute_ext as *const ()),
            Opcode::SEXT => format!("{:p}", execute_sext as *const ()),
            Opcode::INS => format!("{:p}", execute_ins as *const ()),
            Opcode::TEQ => format!("{:p}", execute_teq as *const ()),
            _ => unreachable!(),
        };
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);

        let mut asm = String::new();

        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers(); // pop the internal registers from the stack
        asm += &Self::pop_address_space_start();
        // read the memory from the memory location of the MIPS registers in `GuestMemory`
        // registers, to the appropriate XMM registers
        asm += &Self::load_xmm_regs();

        Ok(asm)
    }

    fn generate_syscall_asm(_instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let extern_handler_ptr = format!("{:p}", execute_syscall as *const ());

        let mut asm = String::new();

        asm += "   # syscall\n";
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();

        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   mov {REG_TMP}, {REG_RETURN_VAL}\n");

        asm += &Self::pop_internal_registers(); // pop the internal registers from the stack
        asm += &Self::pop_address_space_start();
        // read the memory from the memory location of the MIPS registers in `GuestMemory`
        // registers, to the appropriate XMM registers
        asm += &Self::load_xmm_regs();

        asm += &format!("   cmp {REG_TMP}, 0\n"); // Halt
        asm += &format!("   mov {REG_NEXT_PC}, 0\n");
        asm += "   je asm_run_end\n";
        asm += &format!("   cmp {REG_TMP}, 1\n"); // !EXIT_UNCONSTRAINED
        asm += &format!("   je end_syscall_{pc}\n");

        // EXIT_UNCONSTRAINED
        // Update the memory address space, register address space and xmm registers
        let get_address_space_ptr = format!("{:p}", get_address_space as *const ());
        asm += "    # push_internal_registers\n";
        asm += &Self::push_internal_registers();
        // Store the start of memory address space in r15
        // asm += "    # Store the start of memory address space in r15\n";
        asm += &format!("    mov {REG_CALLER}, {get_address_space_ptr}\n");
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_SECOND_ARG}, 1\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += &format!("    mov {REG_MEMORY_PTR}, {REG_RETURN_VAL}\n");
        // Store the start of register address space in high 64 bits of xmm0
        asm += "    # Store the start of register address space in high 64 bits of xmm0\n";
        asm += &format!("    mov {REG_CALLER}, {get_address_space_ptr}\n");
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_SECOND_ARG}, 0\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += &format!("    pinsrq  xmm0, {REG_RETURN_VAL}, 1\n");
        asm += "    # pop_internal_registers\n";
        asm += &Self::pop_internal_registers();
        asm += "    # load_xmm_regs\n";
        asm += &Self::load_xmm_regs();

        // Jump to the next instruction
        asm += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm += &format!("   movsxd {REG_A}, [{REG_C} + {REG_TMP}]\n");
        asm += &format!("   add {REG_A}, {REG_C}\n");
        asm += &format!("   jmp {REG_A}\n");

        asm += &format!("end_syscall_{pc}:\n");

        Ok(asm)
    }
}

extern "C" fn execute_mov_cond(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rs, rt) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let a = executor.state.read_register(rd);
    let c = executor.state.read_register(rt);
    let b = executor.state.read_register(rs);
    let mov = match instruction.opcode {
        Opcode::MEQ => c == 0,
        Opcode::MNE => c != 0,
        _ => {
            unreachable!()
        }
    };

    let a = if mov { b } else { a };
    executor.state.write_register(rd, a);
}

extern "C" fn execute_maddu(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = b as u64 * c as u64;
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = multiply.wrapping_add(addend);
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_msubu(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = b as u64 * c as u64;
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = addend.wrapping_sub(multiply);
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_madd(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = (b as i32 as i64) * (c as i32 as i64);
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = multiply.wrapping_add(addend as i64) as u64;
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_msub(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = (b as i32 as i64) * (c as i32 as i64);
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = (addend as i64).wrapping_sub(multiply) as u64;
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_wsbh(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());
    let b = executor.state.read_register(rt);
    let a = (((b >> 16) & 0xFF) << 24)
        | (((b >> 24) & 0xFF) << 16)
        | ((b & 0xFF) << 8)
        | ((b >> 8) & 0xFF);
    executor.state.write_register(rd, a);
}

extern "C" fn execute_ext(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let msbd = c >> 5;
    let lsb = c & 0x1f;
    let mask_msb = if msbd + lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msbd + lsb + 1)) - 1 };
    let a = (b & mask_msb) >> lsb;
    executor.state.write_register(rd, a);
}

extern "C" fn execute_sext(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let a = if c > 0 { (b & 0xffff) as i16 as i32 as u32 } else { (b & 0xff) as i8 as i32 as u32 };
    executor.state.write_register(rd, a);
}

extern "C" fn execute_ins(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let a = executor.state.read_register(rd);

    let msb = c >> 5;
    let lsb = c & 0x1f;
    let mask = if msb - lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msb - lsb + 1)) - 1 };
    let mask_field = mask << lsb;
    let a = (a & !mask_field) | ((b << lsb) & mask_field);

    executor.state.write_register(rd, a);
}

extern "C" fn execute_teq(executor: &mut Executor, instruction: &Instruction) {
    let (rs, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());

    let src2 = executor.state.read_register(rt);
    let src1 = executor.state.read_register(rs);

    if src1 == src2 {
        panic!("ExecutionError::ExceptionOrTrap()");
    }
}

extern "C" fn execute_syscall(executor: &mut Executor, pc: u32) -> u32 {
    executor.state.pc = pc;
    let syscall_id = executor.state.read_register(Register::V0 as u32);
    let c = executor.state.read_register(Register::A1 as u32);
    let b = executor.state.read_register(Register::A0 as u32);
    let syscall = SyscallCode::from_u32(syscall_id);
    log::trace!("pc: {:X} syscall {}, a0: {:X}, a1: {:X}", executor.state.pc, syscall_id, b, c);

    // `hint_slice` is allowed in unconstrained mode since it is used to write the hint.
    // Other syscalls are not allowed because they can lead to non-deterministic
    // behavior, especially since many syscalls modify memory in place,
    // which is not permitted in unconstrained mode. This will result in
    // non-zero memory lookups when generating a proof.

    if executor.unconstrained
        && (syscall != SyscallCode::EXIT_UNCONSTRAINED && syscall != SyscallCode::WRITE)
    {
        panic!("ExecutionError::InvalidSyscallUsage({syscall})");
    }

    // Update the syscall counts.
    let syscall_for_count = syscall.count_map();
    let syscall_count = executor.state.syscall_counts.entry(syscall_for_count).or_insert(0);
    *syscall_count += 1;

    let syscall_impl = executor.get_syscall(syscall).cloned();
    let mut precompile_rt = SyscallContext::new(executor);
    let (a, precompile_next_pc, precompile_cycles, returned_exit_code) = if let Some(syscall_impl) =
        syscall_impl
    {
        // Executing a syscall optionally returns a value to write to the t0
        // register. If it returns None, we just keep the
        // syscall_id in t0.
        let res = syscall_impl.execute(&mut precompile_rt, syscall, b, c).expect("syscall failed");
        let a = if let Some(r0) = res { r0 } else { syscall_id };

        // If the syscall is `HALT` and the exit code is non-zero, return an error.
        if syscall == SyscallCode::HALT && precompile_rt.exit_code != 0 {
            panic!("ExecutionError::HaltWithNonZeroExitCode({})", precompile_rt.exit_code);
        }

        (a, precompile_rt.next_pc, syscall_impl.num_extra_cycles(), precompile_rt.exit_code)
    } else {
        panic!("ExecutionError::UnsupportedSyscall({syscall_id}))");
    };

    if syscall == SyscallCode::HALT && returned_exit_code == 0 {
        executor.state.exited = true;
    }

    executor.state.write_register(Register::V0 as u32, a);
    executor.state.clk += precompile_cycles;
    executor.state.pc = precompile_next_pc;
    executor.state.next_pc = precompile_next_pc + 4;

    if executor.state.exited {
        0
    } else if syscall != SyscallCode::EXIT_UNCONSTRAINED {
        1
    } else {
        precompile_next_pc
    }
}

extern "C" fn execute_alu(executor: &mut Executor, instruction: &Instruction, pc: u32, clk: u32, global_clk: u64) {
    println!("{pc} {clk} {global_clk} {:?}", instruction);
    // executor.execute_alu(instruction).unwrap();
}

extern "C" fn execute_branch(_executor: &mut Executor, instruction: &Instruction, pc: u32, clk: u32, global_clk: u64) {
    println!("{pc} {clk} {global_clk} {:?}", instruction);
}

extern "C" fn execute_jump(_executor: &mut Executor, instruction: &Instruction, pc: u32, clk: u32, global_clk: u64) {
    println!("{pc} {clk} {global_clk} {:?}", instruction);
}

extern "C" fn execute_memory_store(executor: &mut Executor, instruction: &Instruction, pc: u32, clk: u32, global_clk: u64) {
    println!("{pc} {clk} {global_clk} {:?}", instruction);
    // executor.execute_store(instruction).unwrap();
}

extern "C" fn execute_memory_load(executor: &mut Executor, instruction: &Instruction, pc: u32, clk: u32, global_clk: u64) {
    println!("{pc} {clk} {global_clk} {:?}", instruction);
    // executor.execute_load(instruction).unwrap();
}
