use crate::aot::common::*;
use crate::aot::{AotCompiler, AotError};
use crate::events::MemoryAccessPosition;
use crate::syscalls::{SyscallCode, SyscallContext};
use crate::{Executor, ExecutorMode, Instruction, Register, DEFAULT_CLK_INC};

impl AotCompiler {
    pub fn generate_syscall_asm(
        &self,
        instruction: &Instruction,
        pc: u32,
        is_delay_slot: bool,
    ) -> Result<String, AotError> {
        let extern_handler_ptr = format!("{:p}", execute_syscall as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);

        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            asm += &Self::get_access_register_meta_addr();
            asm += &Self::set_access_register_meta(
                Register::A1 as u32,
                MemoryAccessPosition::C,
                is_delay_slot,
            );
            asm += &Self::set_access_register_meta(
                Register::A0 as u32,
                MemoryAccessPosition::B,
                is_delay_slot,
            );
            asm += &Self::set_access_register_meta(
                Register::V0 as u32,
                MemoryAccessPosition::A,
                is_delay_slot,
            );
        }

        asm += &Self::sync_reg_to_pc();
        asm += &Self::sync_reg_to_clk();
        asm += &Self::sync_reg_to_global_clk();

        asm += "   # syscall\n";
        asm += &Self::before_call();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_THIRD_ARG}, {pc}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &format!("   pinsrq  xmm{TMP}, {REG_RETURN_VAL}, 1\n");
        asm += &Self::after_call();

        asm += &Self::sync_clk_to_reg();

        asm += &format!("   pextrq {REG_D}, xmm{TMP}, 1\n");
        asm += &format!("   cmp {REG_D}, 1\n"); // !EXIT_UNCONSTRAINED
        asm += &format!("   je end_syscall_{pc}\n");
        asm += &format!("   cmp {REG_D}, 0\n"); // Halt
        asm += "   je asm_end\n";

        // EXIT_UNCONSTRAINED
        // Update the memory address space, register address space and xmm registers
        asm += "    # push_internal_registers\n";
        asm += &Self::push_internal_registers();

        asm += &self.get_address_space_start();

        asm += "    # pop_internal_registers\n";
        asm += &Self::pop_internal_registers();
        asm += "    # load_xmm_regs\n";
        asm += &Self::load_xmm_regs();

        // Jump to the next instruction
        asm += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm += &format!("   movsxd {REG_A}, [{REG_C} + {REG_D}]\n");
        asm += &format!("   add {REG_A}, {REG_C}\n");
        asm += &format!("   jmp {REG_A}\n");

        asm += &format!("end_syscall_{pc}:\n");

        Ok(asm)
    }
}

extern "C" fn execute_syscall(executor: &mut Executor, _instruction: &Instruction, pc: u32) -> u32 {
    executor.state.pc = pc;
    let syscall_id = executor.state.read_register(Register::V0 as u32);
    let c = executor.state.read_register(Register::A1 as u32);
    let b = executor.state.read_register(Register::A0 as u32);
    let syscall = SyscallCode::from_u32(syscall_id);
    log::trace!("pc: {} syscall {}, a0: {}, a1: {}", executor.state.pc, syscall, b, c);
    // println!(
    //     "aot 0 pc: {}, clk: {}, global_clk: {}, {}",
    //     executor.state.pc, executor.state.clk, executor.state.global_clk, syscall
    // );

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

    // println!(
    //     "aot 1 pc: {}, clk: {}, global_clk: {}, {}, next_next_pc: {}",
    //     precompile_next_pc,
    //     executor.state.clk,
    //     executor.state.global_clk,
    //     syscall,
    //     precompile_next_pc + 4,
    // );

    if executor.state.exited {
        executor.state.clk += DEFAULT_CLK_INC;
        executor.state.global_clk += 1;
        executor.state.pc = 0;
        0
    } else if syscall != SyscallCode::EXIT_UNCONSTRAINED
        && syscall != SyscallCode::ENTER_UNCONSTRAINED
    {
        1
    } else {
        precompile_next_pc
    }
}
