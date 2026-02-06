mod test;

use std::mem::offset_of;

use crate::aot::common::*;
use crate::aot::{get_address_space, get_pc, set_pc, AotCompiler, AotError};
use crate::events::{MemoryAccessMeta, MemoryAccessPosition};
use crate::memory::Entry;
use crate::syscalls::{SyscallCode, SyscallContext};
use crate::{ExecutionState, Executor, Instruction, MipsAirId, Opcode, Register, estimate_mips_event_counts, estimate_mips_lde_size, pad_mips_event_counts};

#[inline]
fn sync_reg_to_global_clk() -> String {
    let global_clk_offset = offset_of!(Executor, state)
            + offset_of!(ExecutionState, global_clk);
    format!(
        "    mov QWORD PTR [{REG_EXECUTOR_PTR} + {global_clk_offset}], {REG_GLOBAL_CLK}\n"
    )
}

#[inline]
fn sync_global_clk_to_reg() -> String {
    let global_clk_offset = offset_of!(Executor, state)
            + offset_of!(ExecutionState, global_clk);
    format!(
        "    mov {REG_GLOBAL_CLK}, [{REG_EXECUTOR_PTR} + {global_clk_offset}]\n"
    )
}

impl AotCompiler {
    pub fn create_metered_asm(&self) -> Result<String, AotError> {
        let mut asm = String::new();

        let clk_offset = offset_of!(Executor, state)
            + offset_of!(ExecutionState, clk);

        let sync_reg_to_clk = || {
            format!(
                "    mov QWORD PTR [{REG_EXECUTOR_PTR} + {clk_offset}], {REG_CLK}\n"
            )
        };
        let sync_clk_to_reg = || {
            format!(
                "    mov {REG_CLK}, [{REG_EXECUTOR_PTR} + {clk_offset}]\n"
            )
        };

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

        asm += "    # pop_internal_registers\n";
        asm += &Self::pop_internal_registers();

        asm += "    # load_xmm_regs\n";
        asm += &Self::load_xmm_regs();

        asm += "    # state.clk = 0\n";
        asm += &format!("   mov {REG_CLK}, 0\n");
        asm += "    # state.global_clk = 0\n";
        asm += &format!("   mov {REG_GLOBAL_CLK}, 0\n");

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
        let most_clk = self.shard_size - self.max_syscall_cycles;

        let mut i = 0;
        while i < self.program.instructions.len() {
            let pc = self.program.pc(i);
            let instruction = &self.program.instructions[i];
            asm += &format!("asm_execute_pc_{pc}:\n");

            // Check if we should suspend or not
            asm += &format!("    cmp {REG_NEXT_PC}, {most_pc}\n");
            asm += "    je asm_run_end\n";
            asm += &format!("    cmp {REG_CLK}, {most_clk}\n");
            asm += "    je asm_run_end\n";
            asm += &format!("    mov {REG_NEXT_PC}, {}\n", pc + 4);
            asm += &format!("    add {REG_CLK}, 5\n");
            asm += &format!("    inc {REG_GLOBAL_CLK}\n");
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
                asm += &format!("    cmp {REG_NEXT_PC}, {most_pc}\n");
                asm += "    je asm_run_end\n";
                asm += &format!("    add {REG_CLK}, 5\n");
                asm += &format!("    cmp {REG_CLK}, {most_clk}\n");
                asm += "    je asm_run_end\n";
                asm += &format!("    inc {REG_GLOBAL_CLK}\n");
                i += 1;
                asm += &(Self::generate_metered_instruction_asm(next_instruction, next_pc)?);

                asm += &(Self::generate_metered_instruction_asm(instruction, pc)?);
            } else {
                asm += &(Self::generate_metered_instruction_asm(instruction, pc)?);
            }
        }

        let set_pc_ptr = format!("{:p}", set_pc as *const ());

        asm += "asm_run_end:\n";
        asm += "    # save_xmm_regs\n";
        asm += &Self::save_xmm_regs();
        asm += "    # sync_reg_to_clk\n";
        asm += &sync_reg_to_clk();
        asm += "    # sync_reg_to_global_clk\n";
        asm += &sync_reg_to_global_clk();
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

        std::fs::write("asm_metered_dump.s", &asm).expect("failed to write asm");

        Ok(asm)
    }

    fn generate_metered_instruction_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        if instruction.is_alu_instruction() {
            return Self::generate_metered_alu_asm(instruction, pc);
        } else if instruction.is_branch_instruction() {
            return Self::generate_metered_branch_asm(instruction, pc);
        } else if instruction.is_jump_instruction() {
            return Self::generate_metered_jump_asm(instruction, pc);
        } else if instruction.is_memory_load_instruction() {
            return Self::generate_metered_memory_load_asm(instruction, pc);
        } else if instruction.is_memory_store_instruction() {
            return Self::generate_metered_memory_store_asm(instruction, pc);
        } else if instruction.is_mov_cond_instruction() {
            return Self::generate_metered_mov_cond_asm(instruction, pc);
        } else if instruction.is_misc_instruction() {
            return Self::generate_metered_misc_asm(instruction, pc);
        } else if instruction.is_syscall_instruction() {
            return Self::generate_metered_syscall_asm(instruction, pc);
        }

        Ok(String::new())
    }

    fn generate_metered_alu_asm(instruction: &Instruction, _pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();
        // match instruction.opcode {
        //     Opcode::ADD | Opcode::SUB | Opcode::OR | Opcode::AND | Opcode::XOR | Opcode::MUL => {
        //         asm += &Self::generate_metered_base_alu_asm(instruction)?;
        //     }
        //     Opcode::NOR => {
        //         asm += &Self::generate_metered_nor_asm(instruction)?;
        //     }
        //     Opcode::SLL | Opcode::SRL | Opcode::SRA | Opcode::ROR => {
        //         asm += &Self::generate_metered_shift_asm(instruction)?;
        //     }
        //     Opcode::MULT | Opcode::MULTU => {
        //         asm += &Self::generate_metered_mult_asm(instruction)?;
        //     }
        //     Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU => {
        //         asm += &Self::generate_metered_div_mod_asm(instruction)?;
        //     }
        //     Opcode::SLT | Opcode::SLTU => {
        //         asm += &Self::generate_metered_slt_asm(instruction)?;
        //     }
        //     Opcode::CLO | Opcode::CLZ => {
        //         asm += &Self::generate_metered_cloz_asm(instruction)?;
        //     }
        //     _ => return Err(AotError::NotSupported),
        // }

        let extern_handler_ptr = format!("{:p}", execute_alu as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {_pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        // asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true
        Ok(asm)
    }

    fn generate_metered_base_alu_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        asm += &Self::generate_base_alu_asm(instruction)?;

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_nor_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        asm += &Self::generate_nor_asm(instruction)?;

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_shift_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        asm += &Self::generate_shift_asm(instruction)?;

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_mult_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        asm += &Self::generate_mult_asm(instruction)?;

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_div_mod_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        asm += &Self::generate_div_mod_asm(instruction)?;

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_slt_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        asm += &Self::generate_slt_asm(instruction)?;

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_cloz_asm(instruction: &Instruction) -> Result<String, AotError> {
        let mut asm = String::new();

        asm += &Self::generate_cloz_asm(instruction)?;

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_branch_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        let extern_handler_ptr = format!("{:p}", execute_branch as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        asm += &Self::generate_branch_asm(instruction, pc)?;

        Ok(asm)
    }

    fn generate_metered_jump_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        let extern_handler_ptr = format!("{:p}", execute_jump as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        asm += &Self::generate_jump_asm(instruction, pc)?;

        Ok(asm)
    }

    fn generate_metered_memory_load_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        // asm += &Self::generate_memory_load_asm(instruction, pc)?;

        let extern_handler_ptr = format!("{:p}", execute_memory_load as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_memory_store_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let mut asm = String::new();

        // asm += &Self::generate_memory_store_asm(instruction, pc)?;
        let extern_handler_ptr = format!("{:p}", execute_memory_store as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        let extern_handler_ptr = format!("{:p}", inc_shard_if_need as *const ());
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n"); // inc_shard_if_need() return true

        Ok(asm)
    }

    fn generate_metered_mov_cond_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let extern_handler_ptr = format!("{:p}", execute_mov_cond as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);

        let mut asm = String::new();

        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers(); // pop the internal registers from the stack
        asm += &Self::pop_address_space_start();
        // read the memory from the memory location of the MIPS registers in `GuestMemory`
        // registers, to the appropriate XMM registers
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n");

        Ok(asm)
    }

    fn generate_metered_misc_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
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
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   test {REG_RETURN_VAL}, {REG_RETURN_VAL}\n");
        asm += &Self::pop_internal_registers(); // pop the internal registers from the stack
        asm += &Self::pop_address_space_start();
        // read the memory from the memory location of the MIPS registers in `GuestMemory`
        // registers, to the appropriate XMM registers
        asm += &Self::load_xmm_regs();

        asm += &format!("    jnz asm_run_end\n");

        Ok(asm)
    }

    fn generate_metered_syscall_asm(instruction: &Instruction, pc: u32) -> Result<String, AotError> {
        let extern_handler_ptr = format!("{:p}", execute_syscall as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);

        let mut asm = String::new();

        asm += "   # syscall\n";
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();

        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   mov {REG_TMP}, {REG_RETURN_VAL}\n");

        asm += &Self::pop_internal_registers();
        asm += &Self::pop_address_space_start();
        asm += &Self::load_xmm_regs();

        asm += &format!("   cmp {REG_TMP}, 0\n"); // Halt
        asm += "   je asm_run_end\n";
        asm += &format!("   cmp {REG_TMP}, 1\n"); // !EXIT_UNCONSTRAINED
        asm += &format!("   je end_syscall_{pc}\n");
        asm += &format!("   cmp {REG_TMP}, 2\n"); // Generate one shard
        asm += "   je asm_run_end\n";

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

extern "C" fn execute_mov_cond(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
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

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_maddu(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
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

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_msubu(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
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

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_madd(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
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

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_msub(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
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

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_wsbh(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
    let (rd, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());
    let b = executor.state.read_register(rt);
    let a = (((b >> 16) & 0xFF) << 24)
        | (((b >> 24) & 0xFF) << 16)
        | ((b & 0xFF) << 8)
        | ((b >> 8) & 0xFF);
    executor.state.write_register(rd, a);

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_ext(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let msbd = c >> 5;
    let lsb = c & 0x1f;
    let mask_msb = if msbd + lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msbd + lsb + 1)) - 1 };
    let a = (b & mask_msb) >> lsb;
    executor.state.write_register(rd, a);

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_sext(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let a = if c > 0 { (b & 0xffff) as i16 as i32 as u32 } else { (b & 0xff) as i8 as i32 as u32 };
    executor.state.write_register(rd, a);

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_ins(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let a = executor.state.read_register(rd);

    let msb = c >> 5;
    let lsb = c & 0x1f;
    let mask = if msb - lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msb - lsb + 1)) - 1 };
    let mask_field = mask << lsb;
    let a = (a & !mask_field) | ((b << lsb) & mask_field);

    executor.state.write_register(rd, a);

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_teq(executor: &mut Executor, instruction: &Instruction, pc: u32) -> bool {
    println!("{pc} {:?}", instruction);
    let (rs, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());

    let src2 = executor.state.read_register(rt);
    let src1 = executor.state.read_register(rs);

    if src1 == src2 {
        panic!("ExecutionError::ExceptionOrTrap()");
    }

    inc_shard_if_need(executor, instruction)
}

extern "C" fn execute_syscall(executor: &mut Executor, instruction: &Instruction, pc: u32) -> u32 {
    println!("{pc} {:?}", instruction);
    executor.state.pc = pc;
    let syscall_id = executor.state.read_register(Register::V0 as u32);
    let c = executor.state.read_register(Register::A1 as u32);
    let b = executor.state.read_register(Register::A0 as u32);
    let syscall = SyscallCode::from_u32(syscall_id);
    log::trace!("pc: {} syscall {}, a0: {}, a1: {}", executor.state.pc, syscall, b, c);
    println!("pc: {} syscall {}, a0: {}, a1: {}", executor.state.pc, syscall, b, c);

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

    let inc_shard = inc_shard_if_need(executor, instruction);

    if executor.state.exited {
        0
    } else if syscall != SyscallCode::EXIT_UNCONSTRAINED {
        1
    } else if inc_shard {
        2
    } else {
        precompile_next_pc
    }
}

#[inline]
fn record_access_timestamp(executor: &mut Executor, instruction: &Instruction) {
    if instruction.is_alu_instruction() {
        alu_rr_timestamp(executor, instruction);
        alu_rw_timestamp(executor, instruction);
    } else if instruction.is_memory_load_instruction() {
        let (rt_reg, rs_reg) =
            (instruction.op_a.into(), (instruction.op_b as u8).into());
        register_timestamp(executor, rs_reg, MemoryAccessPosition::B);
        register_timestamp(executor, rt_reg, MemoryAccessPosition::A);
    } else if instruction.is_memory_store_instruction() {
        let (rt_reg, rs_reg) =
            (instruction.op_a.into(), (instruction.op_b as u8).into());
        register_timestamp(executor, rs_reg, MemoryAccessPosition::B);
        if instruction.opcode != Opcode::SC {
            register_timestamp(executor,rt_reg, MemoryAccessPosition::A)
        }
    } else if instruction.is_branch_instruction() {
        let (src1, src2) =
            (instruction.op_a.into(), (instruction.op_b as u8).into());
        if !instruction.opcode.only_one_operand() {
            register_timestamp(executor,src2, MemoryAccessPosition::B)
        };
        register_timestamp(executor, src1, MemoryAccessPosition::A);
    } else if instruction.is_jump_instruction() {
        // if instruction.opcode == Opcode::Jump {
        //     self.execute_jump(instruction)
        // } else if instruction.opcode == Opcode::Jumpi {
        //     self.execute_jumpi(instruction)
        // } else {
        //     self.execute_jump_direct(instruction)
        // };
    } else if instruction.is_mov_cond_instruction() {

    } else if instruction.is_misc_instruction() {

    } else if instruction.opcode == Opcode::SYSCALL {
        
    }
}

#[inline]
extern "C" fn inc_shard_if_need(executor: &mut Executor, instruction: &Instruction) -> bool {
    // record_access_timestamp(executor, instruction);

    // Increment the clock.
    // executor.state.global_clk += 1;
    // executor.state.clk += 5;
    // println!("-----self.state.global_clk: {}, clk: {}", executor.state.global_clk, executor.state.clk);

    // If the cycle limit is exceeded, return an error.
    if let Some(max_cycles) = executor.max_cycles {
        if executor.state.global_clk >= max_cycles {
            panic!("Err(ExecutionError::ExceededCycleLimit(max_cycles))");
        }
    }

    // If there's not enough cycles left for another instruction, move to the next shard.
    let cpu_exit = executor.max_syscall_cycles + executor.state.clk >= executor.shard_size;

    // Every N cycles, check if there exists at least one shape that fits.
    //
    // If we're close to not fitting, early stop the shard to ensure we don't OOM.
    let mut shape_match_found = true;
    if executor.state.global_clk.is_multiple_of(executor.shape_check_frequency) {
        // Estimate the number of events in the trace.
        let event_counts = estimate_mips_event_counts(
            (executor.state.clk / 5) as u64,
            executor.local_counts.local_mem as u64,
            executor.local_counts.syscalls_sent as u64,
            *executor.local_counts.event_counts,
        );

        // Check if the LDE size is too large.
        if executor.lde_size_check {
            let padded_event_counts =
                pad_mips_event_counts(event_counts, executor.shape_check_frequency);
            let padded_lde_size = estimate_mips_lde_size(padded_event_counts, &executor.costs);
            if padded_lde_size > executor.lde_size_threshold {
                tracing::warn!(
                    "stopping shard early due to lde size: {} Gib",
                    (padded_lde_size as f64) / (1 << 9) as f64,
                );
                shape_match_found = false;
            }
        } else if let Some(maximal_shapes) = &executor.maximal_shapes {
            // Check if we're too "close" to a maximal shape.

            let distance = |threshold: usize, count: usize| {
                if count != 0 {
                    threshold - count
                } else {
                    usize::MAX
                }
            };

            shape_match_found = false;

            for shape in maximal_shapes.iter() {
                let cpu_threshold = shape[MipsAirId::Cpu];
                if executor.state.clk > ((1 << cpu_threshold) << 2) {
                    continue;
                }

                let mut l_infinity = usize::MAX;
                let mut shape_too_small = false;
                for air in MipsAirId::core() {
                    if air == MipsAirId::Cpu {
                        continue;
                    }

                    let threshold = 1 << shape[air];
                    let count = event_counts[air] as usize;
                    if count > threshold {
                        shape_too_small = true;
                        break;
                    }

                    if distance(threshold, count) < l_infinity {
                        l_infinity = distance(threshold, count);
                    }
                }

                if shape_too_small {
                    continue;
                }

                if l_infinity >= 32 * (executor.shape_check_frequency as usize) {
                    shape_match_found = true;
                    break;
                }
            }

            if !shape_match_found {
                executor.record.counts = Some(event_counts);
                tracing::debug!(
                    "stopping shard early due to no shapes fitting: \
                    clk: {},
                    clk_usage: {}",
                    (executor.state.clk / 5).next_power_of_two().ilog2(),
                    ((executor.state.clk / 5) as f64).log2(),
                );
            }
        }
    }

    if cpu_exit || !shape_match_found {
        println!("============executor.state.clk: {}", executor.state.clk);
        executor.state.records_clk.push(executor.state.clk);
        executor.state.current_shard += 1;
        executor.state.clk = 0;
        return true;
    }
    false
}

#[inline]
fn alu_rr_timestamp(executor: &mut Executor, instruction: &Instruction) {
    if !instruction.imm_c {
        let (rs1, rs2) = (
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        register_timestamp(executor, rs2, MemoryAccessPosition::C);
        register_timestamp(executor,  rs1, MemoryAccessPosition::B);
    } else if !instruction.imm_b && instruction.imm_c {
        let rs1 = (instruction.op_b as u8).into();
        register_timestamp(executor,  rs1, MemoryAccessPosition::B);
    }
}

#[inline]
fn alu_rw_timestamp(executor: &mut Executor, instruction: &Instruction) {
    if instruction.opcode.is_use_lo_hi_alu() {
        register_timestamp(executor, Register::LO, MemoryAccessPosition::A);
        register_timestamp(executor, Register::HI, MemoryAccessPosition::HI);
    } else {
        let rd = instruction.op_a.into();
        register_timestamp(executor, rd, MemoryAccessPosition::A);
    };
}

#[inline]
fn register_timestamp(executor: &mut Executor, register: Register, position: MemoryAccessPosition) {
    let addr = register as u32;
    let entry = executor.state.access_meta.registers.entry(addr);

    // If it's the first time accessing this address, initialize previous values.
    let record: &mut MemoryAccessMeta = match entry {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(MemoryAccessMeta::default()),
    };

    record.shard = executor.state.current_shard;
    record.timestamp = executor.state.clk + position as u32;
}

extern "C" fn execute_alu(executor: &mut Executor, instruction: &Instruction, pc: u32) {
    println!("{pc} {:?}", instruction);
    executor.execute_alu(instruction).unwrap();
}

extern "C" fn execute_branch(_executor: &mut Executor, instruction: &Instruction, pc: u32) {
    println!("{pc} {:?}", instruction);
}

extern "C" fn execute_jump(_executor: &mut Executor, instruction: &Instruction, pc: u32) {
    println!("{pc} {:?}", instruction);
}

extern "C" fn execute_memory_store(executor: &mut Executor, instruction: &Instruction, pc: u32) {
    println!("{pc} {:?}", instruction);
    executor.execute_store(instruction).unwrap();
}

extern "C" fn execute_memory_load(executor: &mut Executor, instruction: &Instruction, pc: u32) {
    println!("{pc} {:?}", instruction);
    executor.execute_load(instruction).unwrap();
}
