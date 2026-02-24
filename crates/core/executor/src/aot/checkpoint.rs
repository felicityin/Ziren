use std::mem::offset_of;

use crate::aot::common::*;
use crate::aot::{get_pc, AotCompiler, AotError};
use crate::{
    DEFAULT_CLK_INC, ExecutionState, Executor, MipsAirId, estimate_mips_event_counts, estimate_mips_lde_size, pad_mips_event_counts
};

impl AotCompiler {
    pub fn create_metered_asm(&self) -> Result<String, AotError> {
        let mut asm = String::new();

        // pc
        let pc_offset = offset_of!(Executor, state) + offset_of!(ExecutionState, pc);
        let sync_reg_to_pc =
            || format!("    mov DWORD PTR [{REG_EXECUTOR_PTR} + {pc_offset}], {REG_NEXT_PC_W}\n");

        // global clk
        let global_clk_offset =
            offset_of!(Executor, state) + offset_of!(ExecutionState, global_clk);
        let sync_reg_to_global_clk = || {
            format!(
                "    mov QWORD PTR [{REG_EXECUTOR_PTR} + {global_clk_offset}], {REG_GLOBAL_CLK}\n"
            )
        };
        let sync_global_clk_to_reg =
            || format!("    mov {REG_GLOBAL_CLK}, [{REG_EXECUTOR_PTR} + {global_clk_offset}]\n");

        // clk
        let clk_offset = offset_of!(Executor, state) + offset_of!(ExecutionState, clk);
        let sync_reg_to_clk =
            || format!("    mov DWORD PTR [{REG_EXECUTOR_PTR} + {clk_offset}], {REG_CLK_W}\n");

        // shard
        let shard_offset = offset_of!(Executor, state) + offset_of!(ExecutionState, current_shard);
        let sync_shard_to_reg =
            || format!("    mov {REG_SHARD}, [{REG_EXECUTOR_PTR} + {shard_offset}]\n");

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

        asm += "    # push_internal_registers\n";
        asm += &Self::push_internal_registers();

        asm += &self.get_address_space_start();

        // Store the pointer to where `pc` is stored in the state to the register
        asm += "    # Store the pointer to where `pc` is stored to the register\n";
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_CALLER}, {get_pc_ptr}\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += &format!("    mov {REG_NEXT_PC}, {REG_RETURN_VAL}\n");

        asm += "    # pop_internal_registers\n";
        asm += &Self::pop_internal_registers();

        asm += "    # load_xmm_regs\n";
        asm += &Self::load_xmm_regs();

        asm += "    # state.clk = 0\n";
        asm += &format!("   mov {REG_CLK}, 0\n");

        asm += &sync_global_clk_to_reg();
        asm += &sync_shard_to_reg();

        asm += "    # execute\n";
        asm += &format!("   lea {REG_C}, [rip + map_pc_base]\n");
        asm += &format!("   movsxd {REG_A}, [{REG_C} + {REG_NEXT_PC}]\n");
        asm += &format!("   add {REG_A}, {REG_C}\n");
        asm += &format!("   jmp {REG_A}\n");

        for i in 0..(self.program.pc_base / 4) {
            asm += &format!("asm_execute_pc_{}:", i * 4);
            asm += "\n";
        }

        let inc_shard_if_need_ptr = format!("{:p}", inc_shard_if_need as *const ());
        let most_pc = self.program.pc_base + self.program.instructions.len() as u32 * 4;
        let most_clk = self.shard_size - self.max_syscall_cycles;
        let shape_check_frequency = self.shape_check_frequency;

        let mut i = 0;
        while i < self.program.instructions.len() {
            let pc = self.program.pc(i);
            let instruction = &self.program.instructions[i];
            asm += &format!("asm_execute_pc_{pc}:\n");

            // Check if we should suspend or not
            asm += &format!("    cmp {REG_NEXT_PC}, {most_pc}\n");
            asm += "    jae asm_run_end\n";
            asm += &format!("    cmp {REG_CLK}, {most_clk}\n");
            asm += "    jae asm_run_end\n";

            // Check global_clk % shape_check_frequency
            asm += "    # global_clk % shape_check_frequency\n";
            asm += &format!("    mov {REG_HI}, 0\n");
            asm += &format!("    mov {REG_LO_64}, {REG_GLOBAL_CLK}\n");
            asm += &format!("    mov {REG_D}, {shape_check_frequency}\n");
            asm += &format!("    div {REG_D}\n");
            asm += &format!("    test {REG_HI_64}, {REG_HI_64}\n");
            asm += &format!("    jnz .{pc}_inc_pc_clk\n");

            // global_clk % shape_check_frequency == 0
            // Call inc_shard_if_need()
            asm += &sync_reg_to_pc();
            asm += &sync_reg_to_clk();
            asm += &sync_reg_to_global_clk();
            asm += "    # call inc_shard_if_need()\n";
            asm += &Self::before_call();
            asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
            asm += &format!("    mov {REG_CALLER}, {inc_shard_if_need_ptr}\n");
            asm += &format!("    call {REG_CALLER}\n");
            asm += "    test al, al\n";
            asm += &Self::after_call();
            asm += "    jnz asm_run_end\n";

            asm += &format!(".{pc}_inc_pc_clk:\n");
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
                asm += &format!("    add {REG_CLK}, 5\n");
                asm += &format!("    inc {REG_GLOBAL_CLK}\n");
                i += 1;
                asm += &(self.generate_instruction_asm(next_instruction, next_pc)?);

                asm += &(self.generate_instruction_asm(instruction, pc)?);
            } else {
                asm += &(self.generate_instruction_asm(instruction, pc)?);
            }
        }

        asm += "asm_run_end:\n";
        asm += "    # save_xmm_regs\n";
        asm += &Self::save_xmm_regs();
        asm += &sync_reg_to_pc();
        asm += &sync_reg_to_clk();
        asm += &sync_reg_to_global_clk();
        asm += "    # call inc_shard_if_need()\n";
        asm += &format!("    mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("    mov {REG_CALLER}, {inc_shard_if_need_ptr}\n");
        asm += &format!("    call {REG_CALLER}\n");
        asm += "    # pop_external_registers\n";
        asm += &Self::pop_external_registers();
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
}

#[inline]
extern "C" fn inc_shard_if_need(executor: &mut Executor) -> bool {
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
            (executor.state.clk / DEFAULT_CLK_INC) as u64,
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
                    (executor.state.clk / DEFAULT_CLK_INC).next_power_of_two().ilog2(),
                    ((executor.state.clk / DEFAULT_CLK_INC) as f64).log2(),
                );
            }
        }
    }

    if cpu_exit || !shape_match_found {
        println!("------Shard {} ended with clk {} and global_clk {}", executor.state.current_shard, executor.state.clk, executor.state.global_clk);
        executor.state.records_clk.push(executor.state.clk);
        executor.state.current_shard += 1;
        executor.state.clk = 0;
        return true;
    }
    false
}

// Run all tests: `RUST_TEST_THREADS=1 cargo test test_aot_metered`
#[cfg(test)]
mod tests {
    use zkm_stark::ZKMCoreOpts;

    use crate::Executor;
    use crate::{
        programs::tests::{
            fibonacci_program, max_memory_program, secp256r1_add_program, secp256r1_double_program,
            simple_memory_program, simple_program, ssz_withdrawals_program, u256xu2048_mul_program,
            unaligned_memory_program,
        },
        Instruction, Opcode, Program, Register,
    };

    #[test]
    fn test_aot_metered_add() {
        // add
        simple_op_code_test(Opcode::ADD, 37 + 5, 37, 5);
        // addi
        simple_op_code_i_test(Opcode::ADD, 37 + 5 + 42, 37, 5, 42);
        // addi negative
        simple_op_code_i_test(Opcode::ADD, 5 - 1 + 4, 5, 0xFFFF_FFFF, 4);

        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 100, false, true),
            Instruction::new(Opcode::ADD, Register::RA as u8, 0, 200, false, true),
            Instruction::new(
                Opcode::ADD,
                Register::RA as u8,
                29,
                Register::RA as u32,
                false,
                false,
            ),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::RA), 300);

        assert_eq!(runtime.state.clk, 15);
        assert_eq!(runtime.state.global_clk, 3);
        assert_eq!(runtime.state.current_shard, 1);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 12);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    #[test]
    fn test_aot_metered_sub() {
        // sub
        simple_op_code_test(Opcode::SUB, 37 - 5, 37, 5);
        // subi
        simple_op_code_i_test(Opcode::SUB, 37 - 5 - 2, 37, 5, 2);
        // subi negative
        simple_op_code_i_test(Opcode::SUB, 5 + 1 - 4, 5, 0xFFFF_FFFF, 4);
    }

    #[test]
    fn test_aot_metered_and() {
        // and
        simple_op_code_test(Opcode::AND, 37 & 5, 37, 5);
        // andi
        simple_op_code_i_test(Opcode::AND, 37 & 5 & 42, 37, 5, 42);
    }

    #[test]
    fn test_aot_metered_or() {
        // or
        simple_op_code_test(Opcode::OR, 37 | 5, 37, 5);
        // ori
        simple_op_code_i_test(Opcode::OR, 37 | 5 | 42, 37, 5, 42);
    }

    #[test]
    fn test_aot_metered_xor() {
        // xor
        simple_op_code_test(Opcode::XOR, 37 ^ 5, 37, 5);
        // xori
        simple_op_code_i_test(Opcode::XOR, 37 ^ 5 ^ 42, 37, 5, 42);
    }

    #[test]
    fn test_aot_metered_mul() {
        simple_op_code_test(Opcode::MUL, 0x00001200, 0x00007e00, 0xb6db6db7);
        simple_op_code_test(Opcode::MUL, 0x00001240, 0x00007fc0, 0xb6db6db7);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x00000000, 0x00000000);
        simple_op_code_test(Opcode::MUL, 0x00000001, 0x00000001, 0x00000001);
        simple_op_code_test(Opcode::MUL, 0x00000015, 0x00000003, 0x00000007);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x00000000, 0xffff8000);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x80000000, 0x00000000);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x80000000, 0xffff8000);
        simple_op_code_test(Opcode::MUL, 0x0000ff7f, 0xaaaaaaab, 0x0002fe7d);
        simple_op_code_test(Opcode::MUL, 0x0000ff7f, 0x0002fe7d, 0xaaaaaaab);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0xff000000, 0xff000000);
        simple_op_code_test(Opcode::MUL, 0x00000001, 0xffffffff, 0xffffffff);
        simple_op_code_test(Opcode::MUL, 0xffffffff, 0xffffffff, 0x00000001);
        simple_op_code_test(Opcode::MUL, 0xffffffff, 0x00000001, 0xffffffff);
    }

    #[test]
    fn test_aot_metered_shift() {
        // sllv
        simple_op_code_test(Opcode::SLL, 1 << 2, 1, 2);
        // srlv
        simple_op_code_test(Opcode::SRL, 8 >> 1, 8, 1);
        // srav
        simple_op_code_test(Opcode::SRA, 37 >> 4, 37, 4);
        // rotrv
        let c = (((0x12345678 as u64) + ((0x12345678 as u64) << 32)) >> 4) as u32;
        simple_op_code_test(Opcode::ROR, c, 0x12345678, 4);

        // sll
        simple_op_code_i_test(Opcode::SLL, 1 << 2 << 3, 1, 2, 3);
        // srl
        simple_op_code_i_test(Opcode::SRL, 8 >> 1 >> 1, 8, 1, 1);
        // sra
        simple_op_code_i_test(Opcode::SRA, 37 >> 4 >> 1, 37, 4, 1);
        // rotr
        let c = ((c as u64) + ((c as u64) << 32)) >> 4;
        simple_op_code_i_test(Opcode::ROR, c as u32, 0x12345678, 4, 4);

        // sll
        let instructions =
            vec![Instruction::new(Opcode::SLL, Register::RA as u8, 40, 16, true, true)];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::RA), 40 << 16);
    }

    #[test]
    fn test_aot_metered_shifts() {
        simple_op_code_test(Opcode::SLL, 0x00000001, 0x00000001, 0);
        simple_op_code_test(Opcode::SLL, 0x00000002, 0x00000001, 1);
        simple_op_code_test(Opcode::SLL, 0x00000080, 0x00000001, 7);
        simple_op_code_test(Opcode::SLL, 0x00004000, 0x00000001, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0x00000001, 31);
        simple_op_code_test(Opcode::SLL, 0xffffffff, 0xffffffff, 0);
        simple_op_code_test(Opcode::SLL, 0xfffffffe, 0xffffffff, 1);
        simple_op_code_test(Opcode::SLL, 0xffffff80, 0xffffffff, 7);
        simple_op_code_test(Opcode::SLL, 0xffffc000, 0xffffffff, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0xffffffff, 31);
        simple_op_code_test(Opcode::SLL, 0x21212121, 0x21212121, 0);
        simple_op_code_test(Opcode::SLL, 0x42424242, 0x21212121, 1);
        simple_op_code_test(Opcode::SLL, 0x90909080, 0x21212121, 7);
        simple_op_code_test(Opcode::SLL, 0x48484000, 0x21212121, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0x21212121, 31);
        simple_op_code_test(Opcode::SLL, 0x21212121, 0x21212121, 0xffffffe0);
        simple_op_code_test(Opcode::SLL, 0x42424242, 0x21212121, 0xffffffe1);
        simple_op_code_test(Opcode::SLL, 0x90909080, 0x21212121, 0xffffffe7);
        simple_op_code_test(Opcode::SLL, 0x48484000, 0x21212121, 0xffffffee);
        simple_op_code_test(Opcode::SLL, 0x00000000, 0x21212120, 0xffffffff);

        simple_op_code_test(Opcode::SRL, 0xffff8000, 0xffff8000, 0);
        simple_op_code_test(Opcode::SRL, 0x7fffc000, 0xffff8000, 1);
        simple_op_code_test(Opcode::SRL, 0x01ffff00, 0xffff8000, 7);
        simple_op_code_test(Opcode::SRL, 0x0003fffe, 0xffff8000, 14);
        simple_op_code_test(Opcode::SRL, 0x0001ffff, 0xffff8001, 15);
        simple_op_code_test(Opcode::SRL, 0xffffffff, 0xffffffff, 0);
        simple_op_code_test(Opcode::SRL, 0x7fffffff, 0xffffffff, 1);
        simple_op_code_test(Opcode::SRL, 0x01ffffff, 0xffffffff, 7);
        simple_op_code_test(Opcode::SRL, 0x0003ffff, 0xffffffff, 14);
        simple_op_code_test(Opcode::SRL, 0x00000001, 0xffffffff, 31);
        simple_op_code_test(Opcode::SRL, 0x21212121, 0x21212121, 0);
        simple_op_code_test(Opcode::SRL, 0x10909090, 0x21212121, 1);
        simple_op_code_test(Opcode::SRL, 0x00424242, 0x21212121, 7);
        simple_op_code_test(Opcode::SRL, 0x00008484, 0x21212121, 14);
        simple_op_code_test(Opcode::SRL, 0x00000000, 0x21212121, 31);
        simple_op_code_test(Opcode::SRL, 0x21212121, 0x21212121, 0xffffffe0);
        simple_op_code_test(Opcode::SRL, 0x10909090, 0x21212121, 0xffffffe1);
        simple_op_code_test(Opcode::SRL, 0x00424242, 0x21212121, 0xffffffe7);
        simple_op_code_test(Opcode::SRL, 0x00008484, 0x21212121, 0xffffffee);
        simple_op_code_test(Opcode::SRL, 0x00000000, 0x21212121, 0xffffffff);

        simple_op_code_test(Opcode::SRA, 0x00000000, 0x00000000, 0);
        simple_op_code_test(Opcode::SRA, 0xc0000000, 0x80000000, 1);
        simple_op_code_test(Opcode::SRA, 0xff000000, 0x80000000, 7);
        simple_op_code_test(Opcode::SRA, 0xfffe0000, 0x80000000, 14);
        simple_op_code_test(Opcode::SRA, 0xffffffff, 0x80000001, 31);
        simple_op_code_test(Opcode::SRA, 0x7fffffff, 0x7fffffff, 0);
        simple_op_code_test(Opcode::SRA, 0x3fffffff, 0x7fffffff, 1);
        simple_op_code_test(Opcode::SRA, 0x00ffffff, 0x7fffffff, 7);
        simple_op_code_test(Opcode::SRA, 0x0001ffff, 0x7fffffff, 14);
        simple_op_code_test(Opcode::SRA, 0x00000000, 0x7fffffff, 31);
        simple_op_code_test(Opcode::SRA, 0x81818181, 0x81818181, 0);
        simple_op_code_test(Opcode::SRA, 0xc0c0c0c0, 0x81818181, 1);
        simple_op_code_test(Opcode::SRA, 0xff030303, 0x81818181, 7);
        simple_op_code_test(Opcode::SRA, 0xfffe0606, 0x81818181, 14);
        simple_op_code_test(Opcode::SRA, 0xffffffff, 0x81818181, 31);
    }

    #[test]
    fn test_aot_metered_mult() {
        let mult = |b: u32, c: u32| -> (u32, u32) {
            let out = (((b as i32) as i64) * ((c as i32) as i64)) as u64;
            (out as u32, (out >> 32) as u32) // lo,hi
        };
        let multu = |b: u32, c: u32| -> (u32, u32) {
            let out = b as u64 * c as u64;
            (out as u32, (out >> 32) as u32) //lo,hi
        };

        let tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in tests {
            let (lo, hi) = mult(b, c);
            lo_hi_op_code_test(Opcode::MULT, hi, lo, b, c);

            let (lo, hi) = multu(b, c);
            lo_hi_op_code_test(Opcode::MULTU, hi, lo, b, c);
        }
    }

    #[test]
    fn test_aot_metered_div() {
        let div = |b: u32, c: u32| -> (u32, u32) {
            (
                ((b as i32) / (c as i32)) as u32, // lo
                ((b as i32) % (c as i32)) as u32, // hi
            )
        };
        let divu = |b: u32, c: u32| -> (u32, u32) {
            (b / c, b % c) // lo,hi
        };

        let tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in tests {
            let (lo, hi) = div(b, c);
            lo_hi_op_code_test(Opcode::DIV, hi, lo, b, c);

            let (lo, hi) = divu(b, c);
            lo_hi_op_code_test(Opcode::DIVU, hi, lo, b, c);
        }
    }

    #[test]
    fn test_aot_metered_mod() {
        let modu = |b: u32, c: u32| -> u32 { b % c };
        let modu_tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in modu_tests {
            let expected = modu(b, c);
            simple_op_code_test(Opcode::MODU, expected, b, c);
        }

        let mod_signed = |b: u32, c: u32| -> u32 { ((b as i32) % (c as i32)) as u32 };
        let mod_tests = vec![
            (10, 3),
            (100, 7),
            (1234, 56),
            (0xffff, 0xff),
            (u32::MAX - 1, u32::MAX - 2),
            (0xffff_ffff, 3),
            (0xffff_fffe, 7),
        ];
        for (b, c) in mod_tests {
            let expected = mod_signed(b, c);
            simple_op_code_test(Opcode::MOD, expected, b, c);
        }
    }

    #[test]
    fn test_aot_metered_slt() {
        // slt
        simple_op_code_test(Opcode::SLT, 1, 5, 10);
        simple_op_code_test(Opcode::SLT, 0, 10, 5);
        simple_op_code_test(Opcode::SLT, 0, 10, 10);
        // slti
        op_code_one_i_test(Opcode::SLT, 1, 5, 10);
        op_code_one_i_test(Opcode::SLT, 0, 10, 5);
        op_code_one_i_test(Opcode::SLT, 0, 10, 10);
        // sltu
        simple_op_code_test(Opcode::SLTU, 1, 5, 10);
        simple_op_code_test(Opcode::SLTU, 0, 10, 5);
        simple_op_code_test(Opcode::SLTU, 0, 10, 10);
        // sltiu
        op_code_one_i_test(Opcode::SLTU, 1, 5, 10);
        op_code_one_i_test(Opcode::SLTU, 0, 10, 5);
        op_code_one_i_test(Opcode::SLTU, 0, 10, 10);
    }

    #[test]
    fn test_aot_metered_nor() {
        let nor = |b: u32, c: u32| -> u32 { !(b | c) };
        let mod_tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in mod_tests {
            let expected = nor(b, c);
            simple_op_code_test(Opcode::NOR, expected, b, c);
        }
    }

    #[test]
    fn test_aot_metered_cloz() {
        let clz = |b: u32| -> u32 { b.leading_zeros() };
        let clo = |b: u32| -> u32 { b.leading_ones() };
        let cloz_tests = vec![10, 100, 1234, 0xffff, u32::MAX - 1];
        for b in cloz_tests {
            let expected = clz(b);
            op_code_one_test(Opcode::CLZ, expected, b);
            let expected = clo(b);
            op_code_one_test(Opcode::CLO, expected, b);
        }
    }

    #[test]
    fn test_aot_metered_beq_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 1, false, true),
            Instruction::new(Opcode::BEQ, 29, 30, 8, false, false),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 33, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 24);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(30);
        assert_eq!(shard, 1);
        assert_eq!(clk, 12);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 0);
        assert_eq!(clk, 0);
        let (shard, clk) = runtime.state.read_register_access_meta(33);
        assert_eq!(shard, 1);
        assert_eq!(clk, 23);
    }

    #[test]
    fn test_aot_metered_beq_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 2, false, true),
            Instruction::new(Opcode::BEQ, 29, 30, 100, false, false),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 16);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(30);
        assert_eq!(shard, 1);
        assert_eq!(clk, 12);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
    }

    #[test]
    fn test_aot_metered_bne_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BNE, Register::A0 as u8, 1, 8, true, true),
            Instruction::new(Opcode::SW, 31, 0, 0x10000000, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 12);

        let (shard, clk) = runtime.state.read_register_access_meta(4);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_memory_access_meta(0x10000000);
        assert_eq!(shard, 1);
        assert_eq!(clk, 5);
    }

    #[test]
    fn test_aot_metered_bne_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BNE, Register::A0 as u8, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 8);

        let (shard, clk) = runtime.state.read_register_access_meta(4);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
    }

    #[test]
    fn test_aot_metered_bltz_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xFFFF_FFFF, false, true),
            Instruction::new(Opcode::BLTZ, 29, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 16);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
    }

    #[test]
    fn test_aot_metered_bltz_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BLTZ, Register::A0 as u8, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 8);

        let (shard, clk) = runtime.state.read_register_access_meta(4);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
    }

    #[test]
    fn test_aot_metered_blez_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BLEZ, Register::A0 as u8, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 12);

        let (shard, clk) = runtime.state.read_register_access_meta(4);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    #[test]
    fn test_aot_metered_blez_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::BLEZ, 29, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 12);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    #[test]
    fn test_aot_metered_bgtz_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::BGTZ, 29, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 16);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
    }

    #[test]
    fn test_aot_metered_bgtz_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BGTZ, Register::A0 as u8, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 8);

        let (shard, clk) = runtime.state.read_register_access_meta(4);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
    }

    #[test]
    fn test_aot_metered_bgez_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BGEZ, Register::A0 as u8, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 33, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 16);

        let (shard, clk) = runtime.state.read_register_access_meta(4);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(33);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
    }

    #[test]
    fn test_aot_metered_bgez_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xFFFF_FFFF, false, true),
            Instruction::new(Opcode::BGEZ, 29, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 12);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    #[test]
    fn test_aot_metered_j() {
        //   j 8
        //
        // The j instruction performs an unconditional jump to a specified address.

        let instructions = vec![
            Instruction::new(Opcode::Jumpi, 0, 8, 0, false, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 12);

        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);

        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    #[test]
    fn test_aot_metered_jr() {
        //   addi x11, x11, 12
        //   jr x11
        //
        // The jr instruction jumps to an address stored in a register.

        let instructions = vec![
            Instruction::new(Opcode::ADD, 11, 11, 12, false, true),
            Instruction::new(Opcode::Jump, 0, 11, 0, false, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 16);

        let (shard, clk) = runtime.state.read_register_access_meta(11);
        assert_eq!(shard, 1);
        assert_eq!(clk, 7);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
    }

    #[test]
    fn test_aot_metered_jal() {
        //   addi x11, x11, 8
        //   jal x11
        //
        // The jal instruction jumps to an address and stores the return address in $ra.

        let instructions = vec![
            Instruction::new(Opcode::Jumpi, 13, 8, 0, false, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 12);
        assert_eq!(runtime.state.read_register(13), 8);

        let (shard, clk) = runtime.state.read_register_access_meta(13);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    #[test]
    fn test_aot_metered_jalr() {
        //   addi x11, x11, 12
        //   jalr x11
        //
        // Similar to jal, but jumps to an address stored in a register.

        let instructions = vec![
            Instruction::new(Opcode::ADD, 11, 11, 12, false, true),
            Instruction::new(Opcode::Jump, 13, 11, 0, false, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 32, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 16);
        assert_eq!(runtime.state.read_register(13), 12);

        let (shard, clk) = runtime.state.read_register_access_meta(11);
        assert_eq!(shard, 1);
        assert_eq!(clk, 7);
        let (shard, clk) = runtime.state.read_register_access_meta(13);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
    }

    #[test]
    fn test_aot_metered_bal() {
        let instructions = vec![
            Instruction::new(Opcode::JumpDirect, 31, 4, 0, false, true),
            Instruction::new(Opcode::ADD, 1, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 2, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.pc, 12);
        assert_eq!(runtime.state.read_register(31), 8);

        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 3);
    }

    #[test]
    fn test_aot_metered_simple_memory_program_run() {
        let program = simple_memory_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();

        // Assert SW & LW case
        assert_eq!(runtime.register(28.into()), 0x12348765);

        // Assert LBU cases
        assert_eq!(runtime.register(27.into()), 0x65);
        assert_eq!(runtime.register(26.into()), 0x87);
        assert_eq!(runtime.register(25.into()), 0x34);
        assert_eq!(runtime.register(24.into()), 0x12);

        // Assert LB cases
        assert_eq!(runtime.register(23.into()), 0x65);
        assert_eq!(runtime.register(22.into()), 0xffffff87);

        // Assert LHU cases
        assert_eq!(runtime.register(21.into()), 0x8765);
        assert_eq!(runtime.register(20.into()), 0x1234);

        // Assert LH cases
        assert_eq!(runtime.register(19.into()), 0xffff8765);
        assert_eq!(runtime.register(18.into()), 0x1234);

        // Assert SB cases
        assert_eq!(runtime.register(16.into()), 0x12348725);
        assert_eq!(runtime.register(15.into()), 0x12342525);
        assert_eq!(runtime.register(14.into()), 0x12252525);
        assert_eq!(runtime.register(13.into()), 0x25252525);

        // Assert SH cases
        assert_eq!(runtime.register(10.into()), 0x12346525);
        assert_eq!(runtime.register(11.into()), 0x65256525);
    }

    #[test]
    fn test_aot_metered_sc() {
        let instructions = vec![
            // Save the value 0x12348765 into address 0x43627530
            Instruction::new(Opcode::ADD, 29, 0, 0x12348765, false, true),
            Instruction::new(Opcode::SC, 29, 0, 0x43627530, false, true),
            Instruction::new(Opcode::LW, 28, 0, 0x43627530, false, true),
        ];

        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();

        assert_eq!(runtime.register(28.into()), 0x12348765);
        assert_eq!(runtime.register(29.into()), 1);
    }

    #[test]
    fn test_aot_metered_swl() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xaabbccdd, false, true),
            Instruction::new(Opcode::SW, 29, 0, 0x10000000, false, true),
            Instruction::new(Opcode::ADD, 28, 0, 0x12345678, false, true),
            Instruction::new(Opcode::SWL, 28, 0, 0x10000001, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.word(0x10000000), 0xaabb1234);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
        let (shard, clk) = runtime.state.read_register_access_meta(28);
        assert_eq!(shard, 1);
        assert_eq!(clk, 18);
        let (shard, clk) = runtime.state.read_memory_access_meta(0x10000000);
        assert_eq!(shard, 1);
        assert_eq!(clk, 15);
    }

    #[test]
    fn test_aot_metered_unaligned_memory_program_run() {
        let program = unaligned_memory_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();

        assert_eq!(runtime.word(0x10000000), 0x12345678);
        assert_eq!(runtime.register(28.into()), 0x5678ccdd);
        assert_eq!(runtime.register(27.into()), 0x345678dd);
        assert_eq!(runtime.register(26.into()), 0x12345678);
        assert_eq!(runtime.register(25.into()), 0x78bbccdd);

        assert_eq!(runtime.register(24.into()), 0xaa123456);
        assert_eq!(runtime.register(23.into()), 0xaabb1234);
        assert_eq!(runtime.register(22.into()), 0xaabbcc12);
        assert_eq!(runtime.register(21.into()), 0x12345678);

        assert_eq!(runtime.word(0x11000000), 0x12345678);
        assert_eq!(runtime.word(0x12000000), 0x125678cc);
        assert_eq!(runtime.word(0x13000000), 0x5678ccdd);
        assert_eq!(runtime.word(0x14000000), 0x12345656);

        assert_eq!(runtime.word(0x15000000), 0x78ccdd78);
        assert_eq!(runtime.word(0x16000000), 0xccdd5678);
        assert_eq!(runtime.word(0x17000000), 0xdd345678);
        assert_eq!(runtime.word(0x18000000), 0x5678ccdd);
    }

    #[test]
    fn test_aot_metered_mov_cond() {
        simple_op_code_test(Opcode::MEQ, 10, 10, 0);
        simple_op_code_test(Opcode::MNE, 10, 10, 1);
    }

    #[test]
    fn test_aot_metered_maddu() {
        let maddu = |hi_val: u32, lo_val: u32, b: u32, c: u32| -> (u32, u32) {
            let multiply = b as u64 * c as u64;
            let addend = ((hi_val as u64) << 32) + lo_val as u64;
            let out = multiply.wrapping_add(addend);
            let out_lo = out as u32;
            let out_hi = (out >> 32) as u32;
            (out_lo, out_hi)
        };
        let (expected_lo, expected_hi) = maddu(100, 200, 300, 400);
        m_lo_hi_op_code_test(Opcode::MADDU, expected_hi, expected_lo, 100, 200, 300, 400);
    }

    #[test]
    fn test_aot_metered_msubu() {
        let msubu = |hi_val: u32, lo_val: u32, b: u32, c: u32| -> (u32, u32) {
            let multiply = b as u64 * c as u64;
            let addend = ((hi_val as u64) << 32) + lo_val as u64;
            let out = addend.wrapping_sub(multiply);
            let out_lo = out as u32;
            let out_hi = (out >> 32) as u32;
            (out_lo, out_hi)
        };
        let (expected_lo, expected_hi) = msubu(100, 200, 300, 400);
        m_lo_hi_op_code_test(Opcode::MSUBU, expected_hi, expected_lo, 100, 200, 300, 400);
    }

    #[test]
    fn test_aot_metered_madd() {
        let madd = |hi_val: u32, lo_val: u32, b: u32, c: u32| -> (u32, u32) {
            let multiply = (b as i32 as i64) * (c as i32 as i64);
            let addend = ((hi_val as u64) << 32) + lo_val as u64;
            let out = multiply.wrapping_add(addend as i64) as u64;
            let out_lo = out as u32;
            let out_hi = (out >> 32) as u32;
            (out_lo, out_hi)
        };
        let (expected_lo, expected_hi) = madd(100, 200, 300, 400);
        m_lo_hi_op_code_test(Opcode::MADDU, expected_hi, expected_lo, 100, 200, 300, 400);
    }

    #[test]
    fn test_aot_metered_msub() {
        let msub = |hi_val: u32, lo_val: u32, b: u32, c: u32| -> (u32, u32) {
            let multiply = (b as i32 as i64) * (c as i32 as i64);
            let addend = ((hi_val as u64) << 32) + lo_val as u64;
            let out = (addend as i64).wrapping_sub(multiply) as u64;
            let out_lo = out as u32;
            let out_hi = (out >> 32) as u32;
            (out_lo, out_hi)
        };
        let (expected_lo, expected_hi) = msub(100, 200, 300, 400);
        m_lo_hi_op_code_test(Opcode::MSUBU, expected_hi, expected_lo, 100, 200, 300, 400);
    }

    #[test]
    fn test_aot_metered_wsbh() {
        let wsbh = |b: u32| -> u32 {
            (((b >> 16) & 0xFF) << 24)
                | (((b >> 24) & 0xFF) << 16)
                | ((b & 0xFF) << 8)
                | ((b >> 8) & 0xFF)
        };
        let expected = wsbh(200);
        op_code_one_test(Opcode::WSBH, expected, 200);
    }

    #[test]
    fn test_aot_metered_ext() {
        let ext = |b: u32, c: u32| -> u32 {
            let msbd = c >> 5;
            let lsb = c & 0x1f;
            let mask_msb =
                if msbd + lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msbd + lsb + 1)) - 1 };
            (b & mask_msb) >> lsb
        };
        let expected = ext(100, 200);
        op_code_one_i_test(Opcode::EXT, expected, 100, 200);
    }

    #[test]
    fn test_aot_metered_sext() {
        let sext = |b: u32, c: u32| -> u32 {
            if c > 0 {
                (b & 0xffff) as i16 as i32 as u32
            } else {
                (b & 0xff) as i8 as i32 as u32
            }
        };
        let expected = sext(100, 200);
        op_code_one_i_test(Opcode::SEXT, expected, 100, 200);
    }

    #[test]
    fn test_aot_metered_ins() {
        let ins = |a: u32, b: u32, c: u32| -> u32 {
            let msb = c >> 5;
            let lsb = c & 0x1f;
            let mask = if msb - lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msb - lsb + 1)) - 1 };
            let mask_field = mask << lsb;
            let a = (a & !mask_field) | ((b << lsb) & mask_field);
            a
        };
        let expected = ins(100, 200, 0b00000_00000_00000);

        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 100, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 200, false, true),
            Instruction::new(Opcode::INS, 29, 30, 0b00000_00000_00000, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(29.into()), expected);
    }

    #[test]
    fn test_aot_metered_hello_run() {
        let program = Program::from(test_artifacts::HELLO_WORLD_ELF).unwrap();
        let mut runtime = Executor::new(program.clone(), ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.clk, 17950);
        assert_eq!(runtime.state.global_clk, 3590);
        assert_eq!(runtime.state.current_shard, 1);

        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.shard_size = 10000;
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.state.clk, 7995);
        assert_eq!(runtime.state.global_clk, 3590);
        assert_eq!(runtime.state.current_shard, 2);

        let (shard, clk) = runtime.state.read_register_access_meta(3);
        assert_eq!(shard, 2);
        assert_eq!(clk, 7952);
        let (shard, clk) = runtime.state.read_register_access_meta(13);
        assert_eq!(shard, 2);
        assert_eq!(clk, 6948);
        let (shard, clk) = runtime.state.read_register_access_meta(23);
        assert_eq!(shard, 2);
        assert_eq!(clk, 6918);
        let (shard, clk) = runtime.state.read_register_access_meta(33);
        assert_eq!(shard, 2);
        assert_eq!(clk, 7932);
        let (shard, clk) = runtime.state.read_memory_access_meta(256164);
        assert_eq!(shard, 1);
        assert_eq!(clk, 2995);
        let (shard, clk) = runtime.state.read_memory_access_meta(256288);
        assert_eq!(shard, 2);
        assert_eq!(clk, 7885);
    }

    #[test]
    fn test_aot_metered_sha2_run() {
        let program = Program::from(test_artifacts::SHA2_ELF).unwrap();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    #[test]
    fn test_aot_metered_simple_program_run() {
        let program = simple_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    #[test]
    fn test_aot_metered_fibo_run() {
        let program = fibonacci_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.shard_size = 10000;
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();

        assert_eq!(runtime.state.clk, 7815);
        assert_eq!(runtime.state.global_clk, 3554);
        assert_eq!(runtime.state.current_shard, 2);
    }

    #[test]
    fn test_aot_metered_max_memory_program_run() {
        let program = max_memory_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    #[test]
    fn test_aot_metered_u256xu2048_mul() {
        let program = u256xu2048_mul_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    #[test]
    fn test_aot_metered_ssz_withdrawals_program_run() {
        let program = ssz_withdrawals_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    #[test]
    fn test_aot_metered_secp256r1_add_program_run() {
        let program = secp256r1_add_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    #[test]
    fn test_aot_metered_secp256r1_double_program_run() {
        let program = secp256r1_double_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    #[test]
    fn test_aot_metered_unconstrained_run() {
        let program = Program::from(test_artifacts::UNCONSTRAINED_ELF).unwrap();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
    }

    // Since it panics within the assembly code, it will cause a fatal runtime error.
    // #[test]
    // #[should_panic]
    // fn test_aot_metered_panic() {
    //     let program = panic_program();
    //     let mut runtime = Executor::new(program, ZKMCoreOpts::default());
    //     runtime.aot_run().unwrap();
    // }

    fn simple_op_code_test(opcode: Opcode, expected: u32, a: u32, b: u32) {
        // addi x29, x0, a
        // addi x30, x0, b
        // <opcode> RA, x29, x30
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, a, false, true),
            Instruction::new(Opcode::ADD, 30, 0, b, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, 30, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);
        assert_eq!(runtime.state.pc, 12);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 12);
        let (shard, clk) = runtime.state.read_register_access_meta(30);
        assert_eq!(shard, 1);
        assert_eq!(clk, 11);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    fn simple_op_code_i_test(opcode: Opcode, expected: u32, a: u32, b: u32, c: u32) {
        // addi x29, x0, a
        // <opcode i> x30, x29, b
        // <opcode i> RA, x30, c
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, a, false, true),
            Instruction::new(opcode, 30, 29, b, false, true),
            Instruction::new(opcode, Register::RA as u8, 30, c, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);
        assert_eq!(runtime.state.pc, 12);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 7);
        let (shard, clk) = runtime.state.read_register_access_meta(30);
        assert_eq!(shard, 1);
        assert_eq!(clk, 12);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
    }

    fn lo_hi_op_code_test(opcode: Opcode, expected_hi: u32, expected_lo: u32, b: u32, c: u32) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, b, false, true),
            Instruction::new(Opcode::ADD, 30, 0, c, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, 30, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::LO), expected_lo);
        assert_eq!(runtime.register(Register::HI), expected_hi);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 12);
        let (shard, clk) = runtime.state.read_register_access_meta(30);
        assert_eq!(shard, 1);
        assert_eq!(clk, 11);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 13);
        let (shard, clk) = runtime.state.read_register_access_meta(33);
        assert_eq!(shard, 1);
        assert_eq!(clk, 14);
    }

    fn m_lo_hi_op_code_test(
        opcode: Opcode,
        expected_hi: u32,
        expected_lo: u32,
        hi: u32,
        lo: u32,
        b: u32,
        c: u32,
    ) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, Register::LO as u8, 0, lo, false, true),
            Instruction::new(Opcode::ADD, Register::HI as u8, 0, hi, false, true),
            Instruction::new(Opcode::ADD, 29, 0, b, false, true),
            Instruction::new(Opcode::ADD, 30, 0, c, false, true),
            Instruction::new(opcode, Register::LO as u8, 29, 30, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::LO), expected_lo);
        assert_eq!(runtime.register(Register::HI), expected_hi);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 22);
        let (shard, clk) = runtime.state.read_register_access_meta(30);
        assert_eq!(shard, 1);
        assert_eq!(clk, 21);
        let (shard, clk) = runtime.state.read_register_access_meta(32);
        assert_eq!(shard, 1);
        assert_eq!(clk, 23);
        let (shard, clk) = runtime.state.read_register_access_meta(33);
        assert_eq!(shard, 1);
        assert_eq!(clk, 24);
    }

    fn op_code_one_i_test(opcode: Opcode, expected: u32, b: u32, c: u32) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, b, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, c, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 7);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
    }

    fn op_code_one_test(opcode: Opcode, expected: u32, c: u32) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, c, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, c, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.aot_compile_metered_lib();
        runtime.aot_metered_run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);

        let (shard, clk) = runtime.state.read_register_access_meta(29);
        assert_eq!(shard, 1);
        assert_eq!(clk, 7);
        let (shard, clk) = runtime.state.read_register_access_meta(31);
        assert_eq!(shard, 1);
        assert_eq!(clk, 8);
    }
}
