mod alu;
mod control;
mod memory;
mod misc;
mod syscall;

use crate::aot::common::*;
use crate::{
    aot::{get_access_clk_space, get_access_shard_space, get_address_space, AotCompiler, AotError},
    events::MemoryAccessPosition,
    ExecutorMode, Instruction, DEFAULT_CLK_INC,
};

impl AotCompiler {
    pub fn generate_instruction_asm(
        &self,
        instruction: &Instruction,
        pc: u32,
    ) -> Result<String, AotError> {
        if instruction.is_alu_instruction() {
            return self.generate_alu_asm(instruction, pc);
        } else if instruction.is_branch_instruction() {
            return self.generate_branch_asm(instruction, pc);
        } else if instruction.is_jump_instruction() {
            return self.generate_jump_asm(instruction, pc);
        } else if instruction.is_memory_load_instruction() {
            return self.generate_memory_load_asm(instruction, pc);
        } else if instruction.is_memory_store_instruction() {
            return self.generate_memory_store_asm(instruction, pc);
        } else if instruction.is_mov_cond_instruction() {
            return self.generate_mov_cond_asm(instruction, pc);
        } else if instruction.is_misc_instruction() {
            return self.generate_misc_asm(instruction, pc);
        } else if instruction.is_syscall_instruction() {
            return self.generate_syscall_asm(instruction, pc);
        }
        Ok(String::new())
    }

    pub fn get_address_space_start(&self) -> String {
        let get_address_space_ptr = format!("{:p}", get_address_space as *const ());
        let get_access_shard_space_ptr = format!("{:p}", get_access_shard_space as *const ());
        let get_access_clk_space_ptr = format!("{:p}", get_access_clk_space as *const ());

        let mut asm = String::new();

        // Store the start of memory address space in r15
        asm += "    # Store the start of memory address space in r15\n";
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
        asm += &format!("    pinsrq  xmm{REG_ADDR_SPACE}, {REG_RETURN_VAL}, 1\n"); // write `eax` to the third lane of xmm0

        if self.executor_mode == ExecutorMode::Checkpoint {
            // Store the address of the shard for accessing the register in high 64 bits of xmm1
            asm += "    # Store the address of the shard for accessing the register in high 64 bits of xmm1\n";
            asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
            asm += &format!("    mov {REG_SECOND_ARG}, 0\n");
            asm += &format!("    mov {REG_CALLER}, {get_access_shard_space_ptr}\n");
            asm += &format!("    call {REG_CALLER}\n");
            asm += &format!("    pinsrq  xmm{ACCESS_REG_SHARD}, {REG_RETURN_VAL}, 1\n"); // write `eax` to the third lane of xmm1

            // Store the address of the clk for accessing the register in high 64 bits of xmm2
            asm += "    # Store the address of the clk for accessing the register in high 64 bits of xmm2\n";
            asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
            asm += &format!("    mov {REG_SECOND_ARG}, 0\n");
            asm += &format!("    mov {REG_CALLER}, {get_access_clk_space_ptr}\n");
            asm += &format!("    call {REG_CALLER}\n");
            asm += &format!("    pinsrq  xmm{ACCESS_REG_CLK}, {REG_RETURN_VAL}, 1\n"); // write `eax` to the third lane of xmm2

            // Store the address of the shard for accessing the memory in high 64 bits of xmm3
            asm += "    # Store the address of the shard for accessing the memory in high 64 bits of xmm3\n";
            asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
            asm += &format!("    mov {REG_SECOND_ARG}, 1\n");
            asm += &format!("    mov {REG_CALLER}, {get_access_shard_space_ptr}\n");
            asm += &format!("    call {REG_CALLER}\n");
            asm += &format!("    pinsrq  xmm{ACCESS_MEM_SHARD}, {REG_RETURN_VAL}, 1\n"); // write `eax` to the third lane of xmm3

            // Store the address of the clk for accessing the memory in high 64 bits of xmm4
            asm += "    # Store the address of the clk for accessing the memory in high 64 bits of xmm4\n";
            asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
            asm += &format!("    mov {REG_SECOND_ARG}, 1\n");
            asm += &format!("    mov {REG_CALLER}, {get_access_clk_space_ptr}\n");
            asm += &format!("    call {REG_CALLER}\n");
            asm += &format!("    pinsrq  xmm{ACCESS_MEM_CLK}, {REG_RETURN_VAL}, 1\n");
            // write `eax` to the third lane of xmm4
        }
        asm
    }

    pub fn get_access_register_meta_addr() -> String {
        let mut asm = String::new();
        asm += &format!("   pextrq {REG_A}, xmm{ACCESS_REG_SHARD}, 1\n");
        asm += &format!("   pextrq {REG_B}, xmm{ACCESS_REG_CLK}, 1\n");
        asm
    }

    pub fn set_access_register_meta(addr: u32, pos: MemoryAccessPosition) -> String {
        let addr = addr << 2;
        let mut asm = String::new();

        // unsafe { self.access_shard.write(MIPS_REGISTER_SPACE, ptr, shard) };
        asm += &format!("   mov dword ptr [{REG_A} + {addr}], {REG_SHARD_W}\n");

        // unsafe { self.access_clk.write(MIPS_REGISTER_SPACE, ptr, clk) };
        asm += &format!("   lea {REG_C}, [{REG_CLK} - {}]\n", DEFAULT_CLK_INC - pos as u32);
        asm += &format!("   mov dword ptr [{REG_B} + {addr}], {REG_C_W}\n");

        asm
    }

    pub fn set_access_register_meta_control(addr: u32, pos: MemoryAccessPosition) -> String {
        let addr = addr << 2;
        let mut asm = String::new();

        // unsafe { self.access_shard.write(MIPS_REGISTER_SPACE, ptr, shard) };
        asm += &format!("   mov dword ptr [{REG_A} + {addr}], {REG_SHARD_W}\n");

        // unsafe { self.access_clk.write(MIPS_REGISTER_SPACE, ptr, clk) };
        asm += &format!(
            "   lea {REG_C}, [{REG_CLK} - {}]\n",
            DEFAULT_CLK_INC + DEFAULT_CLK_INC - pos as u32
        );
        asm += &format!("   mov dword ptr [{REG_B} + {addr}], {REG_C_W}\n");

        asm
    }

    // unsafe { self.access_shard.write(MIPS_MEMORY_SPACE, ptr, shard) };
    pub fn set_access_memory_shard(addr: &str) -> String {
        let mut asm = String::new();

        asm += &format!("   pextrq {REG_A}, xmm{ACCESS_MEM_SHARD}, 1\n");
        asm += &format!("   mov dword ptr [{REG_A} + {addr}], {REG_SHARD_W}\n");

        asm
    }

    // unsafe { self.access_clk.write(MIPS_MEMORY_SPACE, ptr, clk) };
    pub fn set_access_memory_clk(addr: &str) -> String {
        let mut asm = String::new();

        asm += &format!("   lea {REG_C}, [{REG_CLK} - {DEFAULT_CLK_INC}]\n");
        asm += &format!("   pextrq {REG_A}, xmm{ACCESS_MEM_CLK}, 1\n");
        asm += &format!("   mov [{REG_A} + {addr}], {REG_C}\n");

        asm
    }
}
