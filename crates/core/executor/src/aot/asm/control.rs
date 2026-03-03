use crate::aot::common::*;
use crate::aot::{AotCompiler, AotError};
use crate::events::MemoryAccessPosition;
use crate::{ExecutorMode, Instruction, Opcode};

impl AotCompiler {
    pub fn generate_branch_asm(
        &self,
        instruction: &Instruction,
        pc: u32,
    ) -> Result<String, AotError> {
        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            // self.local_counts.event_counts[instruction.opcode as usize] += 1;
            asm += &Self::inc_event_counts(vec![(instruction.opcode, 1)]);

            if instruction.is_branch_cmp_instruction() {
                // self.local_counts.event_counts[Opcode::ADD as usize] += 1;
                // self.local_counts.event_counts[Opcode::SLT as usize] += 2;
                asm += &Self::inc_event_counts(vec![(Opcode::ADD, 1), (Opcode::SLT, 2)]);
            }

            asm += &Self::get_access_register_meta_addr();

            if !instruction.opcode.only_one_operand() {
                asm += &Self::set_access_register_meta_control(
                    instruction.op_b,
                    MemoryAccessPosition::B,
                );
            }

            asm += &Self::set_access_register_meta_control(
                instruction.op_a as u32,
                MemoryAccessPosition::A,
            );
        }

        let next_pc = pc + 4;
        let next_next_pc = next_pc.wrapping_add(instruction.op_c);

        let a = instruction.op_a;
        let b = instruction.op_b as u8;

        let (reg_a, delta_str_a) = &xmm_to_gpr(a, REG_A_W, false);
        asm += delta_str_a;

        if !instruction.opcode.only_one_operand() {
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

    pub fn generate_jump_asm(
        &self,
        instruction: &Instruction,
        pc: u32,
    ) -> Result<String, AotError> {
        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            // self.local_counts.event_counts[instruction.opcode as usize] += 1;
            asm += &Self::inc_event_counts(vec![(instruction.opcode, 1)]);

            if instruction.opcode == Opcode::JumpDirect {
                // self.local_counts.event_counts[Opcode::ADD as usize] += 1;
                asm += &Self::inc_event_counts(vec![(Opcode::ADD, 1)]);
            }

            asm += &Self::get_access_register_meta_addr();

            if instruction.opcode == Opcode::Jump {
                asm += &Self::set_access_register_meta_control(
                    instruction.op_b,
                    MemoryAccessPosition::B,
                );
            }

            asm += &Self::set_access_register_meta_control(
                instruction.op_a as u32,
                MemoryAccessPosition::A,
            );
        }

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
}
