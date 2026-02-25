use crate::aot::common::*;
use crate::aot::{AotCompiler, AotError};
use crate::events::MemoryAccessPosition;
use crate::{ExecutorMode, Instruction, Opcode};

impl AotCompiler {
    pub fn generate_memory_load_asm(
        &self,
        instruction: &Instruction,
        pc: u32,
    ) -> Result<String, AotError> {
        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            // self.local_counts.event_counts[instruction.opcode as usize] += 1;
            asm += &Self::inc_event_counts(vec![(instruction.opcode, 1)]);

            // self.local_counts.event_counts[Opcode::ADD as usize] += 2;
            asm += &Self::inc_event_counts(vec![(Opcode::ADD, 2)]);

            asm += &Self::get_access_register_meta_addr();
            asm += &Self::set_access_register_meta(instruction.op_b, MemoryAccessPosition::B);
            asm +=
                &Self::set_access_register_meta(instruction.op_a as u32, MemoryAccessPosition::A);
        }

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

        if self.executor_mode == ExecutorMode::Checkpoint {
            match instruction.opcode {
                Opcode::LW | Opcode::LL | Opcode::LWL | Opcode::LWR => {
                    asm += &Self::inc_local_memory_counter(pc, gpr_reg_w64);

                    // self.state.write_memory(addr, value);
                    // self.state.write_memory_access_meta(addr, shard, timestamp);
                    asm += &Self::set_access_memory_meta(gpr_reg_w64);
                }
                Opcode::LB | Opcode::LBU | Opcode::LH | Opcode::LHU => {
                    asm += &format!("   mov {REG_D_W}, {gpr_reg}\n");
                    asm += &format!("   and {REG_D_W}, 0xfffffffc\n");

                    asm += &Self::inc_local_memory_counter(pc, REG_D);

                    // self.state.write_memory(addr, value);
                    // self.state.write_memory_access_meta(addr, shard, timestamp);
                    asm += &Self::set_access_memory_meta(REG_D);
                }
                _ => unreachable!(),
            }
        }

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

    pub fn generate_memory_store_asm(
        &self,
        instruction: &Instruction,
        pc: u32,
    ) -> Result<String, AotError> {
        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            // self.local_counts.event_counts[instruction.opcode as usize] += 1;
            asm += &Self::inc_event_counts(vec![(instruction.opcode, 1)]);

            asm += &Self::get_access_register_meta_addr();
            asm += &Self::set_access_register_meta(instruction.op_b, MemoryAccessPosition::B);
            asm +=
                &Self::set_access_register_meta(instruction.op_a as u32, MemoryAccessPosition::A);
        }

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

        if instruction.opcode == Opcode::SW || instruction.opcode == Opcode::SC {
            asm += &format!("   and {gpr_reg}, 0xfffffffc\n");
        }

        let gpr_reg_w64 =
            convert_x86_reg(&gpr_reg, Width::W64).ok_or(AotError::InvalidInstruction)?;
        assert_eq!(gpr_reg_w64, REG_B);

        if self.executor_mode == ExecutorMode::Checkpoint {
            match instruction.opcode {
                Opcode::SW | Opcode::SC | Opcode::SWL | Opcode::SWR => {
                    asm += &Self::inc_local_memory_counter(pc, gpr_reg_w64);

                    // self.state.write_memory(addr, value);
                    // self.state.write_memory_access_meta(addr, shard, timestamp);
                    asm += &Self::set_access_memory_meta(gpr_reg_w64);
                }
                Opcode::SB | Opcode::SH => {
                    asm += &format!("   mov {REG_D_W}, {gpr_reg}\n");
                    asm += &format!("   and {REG_D_W}, 0xfffffffc\n");

                    asm += &Self::inc_local_memory_counter(pc, REG_D);

                    // self.state.write_memory(addr, value);
                    // self.state.write_memory_access_meta(addr, shard, timestamp);
                    asm += &Self::set_access_memory_meta(REG_D);
                }
                _ => unreachable!(),
            }
        }

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
}
