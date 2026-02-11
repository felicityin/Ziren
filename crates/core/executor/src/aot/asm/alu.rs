use crate::aot::common::*;
use crate::aot::{AotCompiler, AotError};
use crate::events::MemoryAccessPosition;
use crate::{ExecutorMode, Instruction, Opcode, Register};

impl AotCompiler {
    pub fn generate_alu_asm(
        &self,
        instruction: &Instruction,
        _pc: u32,
    ) -> Result<String, AotError> {
        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            asm += &Self::get_access_register_meta_addr();
            if !instruction.imm_c {
                asm += &Self::set_access_register_meta(instruction.op_c, MemoryAccessPosition::C);
            }
            if !instruction.imm_b {
                asm += &Self::set_access_register_meta(instruction.op_b, MemoryAccessPosition::B);
            }
            if instruction.opcode.is_use_lo_hi_alu() {
                asm +=
                    &Self::set_access_register_meta(Register::LO as u32, MemoryAccessPosition::A);
                asm +=
                    &Self::set_access_register_meta(Register::HI as u32, MemoryAccessPosition::HI);
            } else {
                asm += &Self::set_access_register_meta(
                    instruction.op_a as u32,
                    MemoryAccessPosition::A,
                );
            }
        }

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
}
