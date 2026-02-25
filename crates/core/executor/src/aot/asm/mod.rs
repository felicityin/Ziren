mod alu;
mod control;
mod memory;
mod misc;
mod syscall;

use std::mem::offset_of;

use crate::aot::{common::*, get_accesssed};
use crate::{
    aot::{get_access_clk_space, get_access_shard_space, get_address_space, AotCompiler, AotError},
    events::MemoryAccessPosition,
    ExecutorMode, Instruction, DEFAULT_CLK_INC,
};
use crate::{Executor, LocalCounts, Opcode};

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
        let get_accessed_ptr = format!("{:p}", get_accesssed as *const ());

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
            asm += &format!("    pinsrq  xmm{ACCESS_MEM_CLK}, {REG_RETURN_VAL}, 1\n"); // write `eax` to the third lane of xmm4

            // Store the address of whether the register has been accessed in the high 64 bits of xmm5
            asm += "    # Store the address of whether the register has been accessed in the high 64 bits of xmm5\n";
            asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
            asm += &format!("    mov {REG_SECOND_ARG}, 0\n");
            asm += &format!("    mov {REG_CALLER}, {get_accessed_ptr}\n");
            asm += &format!("    call {REG_CALLER}\n");
            asm += &format!("    pinsrq  xmm{REG_ACCESSED}, {REG_RETURN_VAL}, 1\n"); // write `eax` to the third lane of xmm5

            // Store the address of whether the memory has been accessed in the high 64 bits of xmm6
            asm += "    # Store the address of whether the memory has been accessed in the high 64 bits of xmm6\n";
            asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
            asm += &format!("    mov {REG_SECOND_ARG}, 1\n");
            asm += &format!("    mov {REG_CALLER}, {get_accessed_ptr}\n");
            asm += &format!("    call {REG_CALLER}\n");
            asm += &format!("    pinsrq  xmm{MEM_ACCESSED}, {REG_RETURN_VAL}, 1\n");
            // write `eax` to the third lane of xmm6
        }
        asm
    }

    pub fn get_access_register_meta_addr() -> String {
        let mut asm = String::new();
        asm += &format!("   pextrq {REG_A}, xmm{ACCESS_REG_SHARD}, 1\n");
        asm += &format!("   pextrq {REG_B}, xmm{ACCESS_REG_CLK}, 1\n");
        asm += &format!("   pextrq {REG_D}, xmm{REG_ACCESSED}, 1\n");
        asm
    }

    /// self.state.write_register_access_meta(addr, shard, timestamp);
    /// self.state.set_register_accessed(addr);
    pub fn set_access_register_meta(addr: u32, pos: MemoryAccessPosition) -> String {
        let addr = addr << 2;
        let mut asm = String::new();

        // unsafe { self.access_shard.write(MIPS_REGISTER_SPACE, ptr, shard) };
        asm += &format!("   mov dword ptr [{REG_A} + {addr}], {REG_SHARD_W}\n");

        // unsafe { self.access_clk.write(MIPS_REGISTER_SPACE, ptr, clk) };
        asm += &format!("   lea {REG_C}, [{REG_CLK} - {}]\n", DEFAULT_CLK_INC - pos as u32);
        asm += &format!("   mov dword ptr [{REG_B} + {addr}], {REG_C_W}\n");

        // self.state.set_register_accessed(addr);
        asm += &format!("   mov dword ptr [{REG_D} + {addr}], 1\n");

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

        // self.state.set_register_accessed(addr);
        asm += &format!("   mov dword ptr [{REG_D} + {addr}], 1\n");

        asm
    }

    pub fn set_access_memory_meta(addr: &str) -> String {
        let mut asm = String::new();

        // unsafe { self.access_shard.write(MIPS_MEMORY_SPACE, ptr, shard) };
        asm += &format!("   pextrq {REG_A}, xmm{ACCESS_MEM_SHARD}, 1\n");
        asm += &format!("   mov dword ptr [{REG_A} + {addr}], {REG_SHARD_W}\n");

        // unsafe { self.access_clk.write(MIPS_MEMORY_SPACE, ptr, clk) };
        asm += &format!("   lea {REG_C}, [{REG_CLK} - {DEFAULT_CLK_INC}]\n");
        asm += &format!("   pextrq {REG_A}, xmm{ACCESS_MEM_CLK}, 1\n");
        asm += &format!("   mov [{REG_A} + {addr}], {REG_C}\n");

        // self.set_memory_accessed(addr);
        asm += &format!("   pextrq {REG_A}, xmm{MEM_ACCESSED}, 1\n");
        asm += &format!("   mov dword ptr [{REG_A} + {addr}], 1\n");

        asm
    }

    pub fn inc_event_counts(opcods: Vec<(Opcode, u8)>) -> String {
        let mut asm = String::new();

        for (opcode, count) in opcods {
            asm += &format!(
                "   lea {REG_A}, [{REG_EXECUTOR_PTR} + {} + {}]\n",
                offset_of!(Executor, local_counts) + offset_of!(LocalCounts, event_counts),
                opcode as usize * 8
            );
            asm += &format!("   add qword ptr [{REG_A}], {count}\n");
        }

        asm
    }

    /// We update the local memory counter in two cases:
    ///  1. This is the first time the address is touched, this corresponds to the
    ///     condition record.shard != shard.
    ///  2. The address is being accessed in a syscall. In this case, we need to send it.
    ///
    /// if !self.unconstrained && (prev_shard != shard || self.in_syscall) {
    ///    self.local_counts.local_mem += 1;
    /// }
    pub fn inc_local_memory_counter(pc: u32, addr: &str) -> String {
        let mut asm = String::new();

        // If self.unconstrained == true, skip the local memory counter increment.
        asm += &format!("   lea {REG_A}, [{REG_EXECUTOR_PTR} + {}]\n", offset_of!(Executor, unconstrained));
        asm += &format!("   movzx {REG_A_W}, byte ptr [{REG_A}]\n");
        asm += &format!("   test {REG_A_8L}, {REG_A_8L}\n");
        asm += &format!("   jnz .{pc}_skip_local_mem_inc\n");

        // If self.in_syscall == true, increase the local memory counter.
        asm += &format!("   lea {REG_A}, [{REG_EXECUTOR_PTR} + {}]\n", offset_of!(Executor, in_syscall));
        asm += &format!("   mov {REG_A_8L}, byte ptr [{REG_A}]\n");
        asm += &format!("   test {REG_A_8L}, {REG_A_8L}\n");
        asm += &format!("   jnz .{pc}_local_mem_inc\n");

        // If prev_shard == shard, skip the local memory counter increment.
        asm += &format!("   pextrq {REG_A}, xmm{ACCESS_MEM_SHARD}, 1\n");
        asm += &format!("   mov {REG_A_W}, dword ptr [{REG_A} + {addr}]\n");
        asm += &format!("   cmp {REG_A_W}, {REG_SHARD_W}\n");
        asm += &format!("   je .{pc}_skip_local_mem_inc\n");

        // self.local_counts.local_mem += 1;
        asm += &format!(".{pc}_local_mem_inc:\n");
        asm += &format!("   lea {REG_A}, [{REG_EXECUTOR_PTR} + {}]\n", offset_of!(Executor, local_counts) + offset_of!(LocalCounts, local_mem));
        asm += &format!("   add qword ptr [{REG_A}], 1\n");

        asm += &format!(".{pc}_skip_local_mem_inc:\n");

        asm
    }
}
