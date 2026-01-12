use std::{
    str::FromStr, sync::Arc
};

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zkm_stark::{StarkVerifyingKey, ZKMCoreOpts, koala_bear_poseidon2::KoalaBearPoseidon2};

use crate::{
    DEFAULT_PC_INC, ExecutionError, ExecutionReport, Instruction, LocalCounts, MIPS_COSTS, MaximalShapes, MipsAirId, NUM_REGISTERS, Opcode, Program, Register, ZKMReduceProof, context::ZKMContext, dependencies::{
        emit_branch_dependencies, emit_cloclz_dependencies, emit_divrem_dependencies,
        emit_jump_dependencies, emit_memory_dependencies, emit_misc_dependencies,
    }, estimate_mips_event_counts, estimate_mips_lde_size, events::{
        AluEvent, BranchEvent, CompAluEvent, CpuEvent, JumpEvent, MemInstrEvent,
        MemoryAccessPosition, MemoryInitializeFinalizeEvent, MemoryLocalEvent, MemoryReadRecord,
        MemoryRecord, MemoryRecordEnum, MemoryWriteRecord, MiscEvent, MovCondEvent, SyscallEvent,
    }, hook::{HookEnv, HookRegistry}, memory::{Entry, Memory}, pad_mips_event_counts, record::{ExecutionRecord, MemoryAccessRecord}, sign_extend, state::{ExecutionState, ForkState}, subproof::SubproofVerifier, syscalls::{Syscall, SyscallCode, SyscallContext, default_syscall_map}, vm::{memory::{AddressMap, GuestMemory, config::{MemoryConfig, MIPS_REGISTER_SPACE}}, state::{VmCheckpointState, VmSimpleState}}
};

/// An executor for the MIPS zkVM.
///
/// The executor is responsible for executing a user program and tracing important events which
/// occur during execution (i.e., memory reads, alu operations, etc).
pub struct CheckpointGenerator<'a, MEM = GuestMemory>  {
    /// The program.
    pub program: Arc<Program>,

    /// A buffer for stdout and stderr IO.
    pub io_buf: HashMap<u32, String>,

    /// The state of the execution.
    pub state: VmCheckpointState<MEM>,

    /// Whether the runtime is in constrained mode or not.
    ///
    /// In unconstrained mode, any events, clock, register, or memory changes are reset after
    /// leaving the unconstrained block. The only thing preserved is written to the input
    /// stream.
    pub unconstrained: bool,

    /// Whether we should write to the report.
    pub print_report: bool,

    /// The maximum size of each shard.
    pub shard_size: u32,

    /// The maximum number of shards to execute at once.
    pub shard_batch_size: u32,

    /// The maximum number of cpu cycles to use for execution.
    pub max_cycles: Option<u64>,

    /// The maximum number of cycles for a syscall.
    pub max_syscall_cycles: u32,

    /// The mapping between syscall codes and their implementations.
    pub syscall_map: HashMap<SyscallCode, Arc<dyn Syscall>>,

    /// The costs of the program.
    pub costs: HashMap<MipsAirId, u64>,

    /// The maximal shapes for the program.
    pub maximal_shapes: Option<MaximalShapes>,

    /// The frequency to check the stopping condition.
    pub shape_check_frequency: u64,

    /// Statistics for event counts.
    pub local_counts: LocalCounts,

    /// Early exit if the estimate LDE size is too big.
    pub lde_size_check: bool,

    /// The maximum LDE size to allow.
    pub lde_size_threshold: u64,

    /// Verifier used to sanity check `verify_zkm_proof` during runtime.
    pub subproof_verifier: Option<&'a dyn SubproofVerifier>,
}

impl<'a> CheckpointGenerator<'a>  {
    /// Create a new [`VmExecutor`] from a program and options.
    #[must_use]
    pub fn new(program: Program, opts: ZKMCoreOpts, context: ZKMContext<'a>) -> Self {
        // Create a shared reference to the program.
        let program = Arc::new(program);

        let memory = GuestMemory::new(&program.image);
        let state = VmCheckpointState::new(program.pc_start, program.next_pc, memory);

        // Determine the maximum number of cycles for any syscall.
        let syscall_map = default_syscall_map();
        let max_syscall_cycles =
            syscall_map.values().map(|syscall| syscall.num_extra_cycles()).max().unwrap_or(0);

        let costs: HashMap<String, usize> = serde_json::from_str(MIPS_COSTS).unwrap();
        let costs: HashMap<MipsAirId, u64> =
            costs.into_iter().map(|(k, v)| (MipsAirId::from_str(&k).unwrap(), v as u64)).collect();

        Self {
            program,
            io_buf: HashMap::new(),
            state,
            unconstrained: false,
            print_report: false,
            shard_size: (opts.shard_size as u32) * 4,
            shard_batch_size: opts.shard_batch_size as u32,
            max_cycles: context.max_cycles,
            max_syscall_cycles,
            syscall_map,
            costs,
            maximal_shapes: None,
            shape_check_frequency: opts.shape_check_frequency,
            local_counts: LocalCounts::default(),
            lde_size_check: false,
            lde_size_threshold: 0,
            subproof_verifier: context.subproof_verifier,
        }
    }

    pub fn run(&mut self) -> Result<(VmCheckpointState, bool), ExecutionError> {
        // Clone self.state without proof_stream in it so it's faster.
        let proof_stream = std::mem::take(&mut self.state.proof_stream);
        let mut checkpoint = tracing::debug_span!("clone").in_scope(|| self.state.clone());
        self.state.proof_stream = proof_stream;

        let done = tracing::debug_span!("execute").in_scope(|| self.execute())?;

        checkpoint.max_clks = std::mem::take(&mut self.state.max_clks);
        Ok((checkpoint, done))
    }

    /// Executes one cycle of the program, returning whether the program has finished.
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn execute(&mut self) -> Result<bool, ExecutionError> {
        let mut done = false;
        let mut num_shards_executed = 0;
    
        // Loop until we've executed `self.shard_batch_size` shards if `self.shard_batch_size` is set.
        loop {
            if self.execute_cycle()? {
                done = true;
                break;
            }

            // We restrict the execution of branch/jump and its delay slot to be in the same shard.
            if !self.unconstrained && !self.state.next_is_delayslot {
                if self.inc_shard_if_need() {
                    num_shards_executed += 1;
                    if num_shards_executed >= self.shard_batch_size {
                        break;
                    }
                }
            }
        }

        Ok(done)
    }

    /// Fetch the instruction at the current program counter.
    #[inline]
    fn fetch(&self) -> Instruction {
        self.program.fetch(self.state.pc)
    }

    /// Executes one cycle of the program, returning whether the program has finished.
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn execute_cycle(&mut self) -> Result<bool, ExecutionError> {
        // Fetch the instruction at the current program counter.
        let instruction = self.fetch();

        // Log the current state of the runtime.
        #[cfg(debug_assertions)]
        self.log(&instruction);

        // Execute the instruction.
        self.execute_operation(&instruction)?;

        // Increment the clock.
        self.state.global_clk += 1;

        // If the cycle limit is exceeded, return an error.
        if let Some(max_cycles) = self.max_cycles {
            if self.state.global_clk >= max_cycles {
                return Err(ExecutionError::ExceededCycleLimit(max_cycles));
            }
        }

        let done = self.state.pc == 0
            || self.state.exited
            || self.state.pc.wrapping_sub(self.program.pc_base)
                >= (self.program.instructions.len() * 4) as u32;
        if done {
            if self.unconstrained {
                tracing::error!("program ended in unconstrained mode at clk {}", self.state.global_clk);
                return Err(ExecutionError::EndInUnconstrained());
            }
        }

        Ok(done)
    }

    fn inc_shard_if_need(&mut self) -> bool {
        // If there's not enough cycles left for another instruction, move to the next shard.
        let cpu_exit = self.max_syscall_cycles + self.state.clk >= self.shard_size;

        // Every N cycles, check if there exists at least one shape that fits.
        //
        // If we're close to not fitting, early stop the shard to ensure we don't OOM.
        let mut shape_match_found = true;
        if self.state.global_clk.is_multiple_of(self.shape_check_frequency) {
            // Estimate the number of events in the trace.
            let event_counts = estimate_mips_event_counts(
                (self.state.clk / 5) as u64,
                self.local_counts.local_mem as u64,
                self.local_counts.syscalls_sent as u64,
                *self.local_counts.event_counts,
            );

            // Check if the LDE size is too large.
            if self.lde_size_check {
                let padded_event_counts =
                    pad_mips_event_counts(event_counts, self.shape_check_frequency);
                let padded_lde_size = estimate_mips_lde_size(padded_event_counts, &self.costs);
                if padded_lde_size > self.lde_size_threshold {
                    tracing::warn!(
                        "stopping shard early due to lde size: {} Gib",
                        (padded_lde_size as f64) / (1 << 9) as f64,
                    );
                    shape_match_found = false;
                }
            } else if let Some(maximal_shapes) = &self.maximal_shapes {
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
                    if self.state.clk > ((1 << cpu_threshold) << 2) {
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

                    if l_infinity >= 32 * (self.shape_check_frequency as usize) {
                        shape_match_found = true;
                        break;
                    }
                }

                if !shape_match_found {
                    tracing::debug!(
                        "stopping shard early due to no shapes fitting: \
                        clk: {},
                        clk_usage: {}",
                        (self.state.clk / 5).next_power_of_two().ilog2(),
                        ((self.state.clk / 5) as f64).log2(),
                    );
                }
            }
        }

        if cpu_exit || !shape_match_found {
            self.state.max_clks.push(self.state.clk);
            self.state.current_shard += 1;
            self.state.clk = 0;
            return true;
        }
        false
    }

    /// Execute the given instruction over the current state of the runtime.
    #[allow(clippy::too_many_lines)]
    fn execute_operation(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        // let mut pc = self.state.pc;
        // let mut exit_code = 0u32; // use in halt code

        // let mut next_pc = self.state.next_pc;
        // let mut next_next_pc = self.state.next_pc + 4;

        if instruction.is_alu_instruction() {
            self.execute_alu(instruction)?;
        // } else if instruction.is_memory_load_instruction() {
        //    self.execute_load(instruction)?;
        // } else if instruction.is_memory_store_instruction() {
        //    self.execute_store(instruction)?;
        // } else if instruction.is_branch_instruction() {
        //    self.execute_branch(instruction, next_pc, next_next_pc);
        // } else if instruction.is_jump_instruction() {
        //     // Jump instructions.
        //     if instruction.opcode == Opcode::Jump {
        //         self.execute_jump(instruction)
        //     } else if instruction.opcode == Opcode::Jumpi {
        //         self.execute_jumpi(instruction)
        //     } else {
        //         self.execute_jump_direct(instruction)
        //     };
        //     // self.state.next_is_delayslot = true;
        // } else if instruction.is_mov_cond_instruction() {
        //     self.execute_condmov(instruction);
        // } else if instruction.is_misc_instruction() {
        //     if instruction.opcode == Opcode::WSBH {
        //         self.execute_wsbh(instruction);
        //     } else if instruction.opcode == Opcode::EXT {
        //         self.execute_ext(instruction);
        //     } else if instruction.opcode == Opcode::MADDU {
        //         self.execute_maddu(instruction);
        //     } else if instruction.opcode == Opcode::INS {
        //         self.execute_ins(instruction);
        //     } else if instruction.opcode == Opcode::SEXT {
        //         self.execute_sext(instruction);
        //     } else if instruction.opcode == Opcode::TEQ {
        //         self.execute_teq(instruction)?;
        //     } else if instruction.opcode == Opcode::MSUBU {
        //         self.execute_msubu(instruction);
        //     } else if instruction.opcode == Opcode::MADD {
        //         self.execute_madd(instruction);
        //     } else if instruction.opcode == Opcode::MSUB {
        //         self.execute_msub(instruction);
        //     }
        // } else if instruction.opcode == Opcode::SYSCALL {
        //     self.execute_syscall()?;
        } else if instruction.opcode == Opcode::UNIMPL {
            tracing::error!("{:X}: {:X}", self.state.pc, instruction.op_c);
            return Err(ExecutionError::UnsupportedInstruction(instruction.op_c));
        } else {
            unreachable!()
        }

        // if next_next_pc == 0 {
        //     tracing::error!("Null pointer reference {:X}: {:X}", self.state.pc, instruction.op_c);
        //     return Err(ExecutionError::NullPointerReference());
        // }

        // // Update the program counter.
        // self.state.pc = next_pc;
        // self.state.next_pc = next_next_pc;
        Ok(())
    }

    fn execute_alu(
        &mut self,
        instruction: &Instruction,
    ) -> Result<(), ExecutionError> {
        let (rd, b, c) = self.alu_rr(instruction);
        if matches!(instruction.opcode, Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU)
            && c == 0
        {
            return Err(ExecutionError::ExceptionOrTrap());
        }

        let (a, hi) = match instruction.opcode {
            Opcode::ADD => (b.overflowing_add(c).0, 0),
            Opcode::SUB => (b.overflowing_sub(c).0, 0),

            Opcode::SLL => (b << (c & 0x1f), 0),
            Opcode::SRL => (b >> (c & 0x1F), 0),
            Opcode::SRA => {
                // same as SRA
                let sin = b as i32;
                let sout = sin >> (c & 0x1f);
                (sout as u32, 0)
            }
            Opcode::ROR => {
                let sin = (b as u64) + ((b as u64) << 32);
                let sout = sin >> (c & 0x1f);
                (sout as u32, 0)
            }
            Opcode::MUL => (b.overflowing_mul(c).0, 0),
            Opcode::SLTU => {
                if b < c {
                    (1, 0)
                } else {
                    (0, 0)
                }
            }
            Opcode::SLT => {
                if (b as i32) < (c as i32) {
                    (1, 0)
                } else {
                    (0, 0)
                }
            }

            Opcode::MULT => {
                let out = (((b as i32) as i64) * ((c as i32) as i64)) as u64;
                (out as u32, (out >> 32) as u32) // lo,hi
            }
            Opcode::MULTU => {
                let out = b as u64 * c as u64;
                (out as u32, (out >> 32) as u32) //lo,hi
            }
            Opcode::DIV => (
                ((b as i32) / (c as i32)) as u32, // lo
                ((b as i32) % (c as i32)) as u32, // hi
            ),
            Opcode::DIVU => (b / c, b % c), //lo,hi
            Opcode::MOD => (((b as i32) % (c as i32)) as u32, 0),
            Opcode::MODU => (b % c, 0), //lo,hi
            Opcode::AND => (b & c, 0),
            Opcode::OR => (b | c, 0),
            Opcode::XOR => (b ^ c, 0),
            Opcode::NOR => (!(b | c), 0),
            Opcode::CLZ => (b.leading_zeros(), 0),
            Opcode::CLO => (b.leading_ones(), 0),
            _ => {
                unreachable!()
            }
        };

        self.alu_rw(instruction, rd, hi, a);

        self.state.pc = self.state.next_pc;
        self.state.next_pc += DEFAULT_PC_INC;
        Ok(())
    }

    /// Fetch the destination register and input operand values for an ALU instruction.
    fn alu_rr(&mut self, instruction: &Instruction) -> (Register, u32, u32) {
        if !instruction.imm_c {
            let (rd, rs1, rs2) = (
                instruction.op_a.into(),
                (instruction.op_b as u8).into(),
                (instruction.op_c as u8).into(),
            );
            let c = self.rr_cpu(rs2, MemoryAccessPosition::C);
            let b = self.rr_cpu(rs1, MemoryAccessPosition::B);
            (rd, b, c)
        } else if !instruction.imm_b && instruction.imm_c {
            let (rd, rs1, imm) =
                (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
            let (rd, b, c) = (rd, self.rr_cpu(rs1, MemoryAccessPosition::B), imm);
            (rd, b, c)
        } else {
            debug_assert!(instruction.imm_b && instruction.imm_c);
            let (rd, b, c) = (instruction.op_a.into(), instruction.op_b, instruction.op_c);
            (rd, b, c)
        }
    }

    /// Read a register.
    #[inline]
    pub fn rr_cpu(&mut self, register: Register, position: MemoryAccessPosition) -> u32 {
        let rs = self.state.vm_read::<u8, 4>(MIPS_REGISTER_SPACE, register as u32);
        u32::from_le_bytes(rs)
    }

    /// Get the current timestamp for a given memory access position.
    #[must_use]
    #[inline]
    pub const fn timestamp(&self, position: &MemoryAccessPosition) -> u32 {
        self.state.clk + *position as u32
    }

    /// Get the current shard.
    #[must_use]
    #[inline]
    pub fn shard(&self) -> u32 {
        self.state.current_shard
    }

    /// Set the destination register with the result and emit an ALU event.
    fn alu_rw(
        &mut self,
        op: &Instruction,
        rd: Register,
        hi: u32,
        a: u32,
    ) -> Option<u32> {
        let hi = if op.opcode.is_use_lo_hi_alu() {
            self.rw_cpu(Register::LO, a);
            self.rw_cpu(Register::HI, hi);
            Some(hi)
        } else {
            self.rw_cpu(rd, a);
            None
        };

        hi
    }

    /// Write to a register.
    pub fn rw_cpu(&mut self, register: Register, value: u32) {
        // Register %x0 should always be 0.
        // We always write 0 to %x0.
        let value = if register == Register::ZERO { 0 } else { value };

        let rd = value.to_le_bytes();
        self.state.vm_write::<u8, 4>(MIPS_REGISTER_SPACE, register as u32, &rd);
    }

    pub fn register(&self, offset: usize) -> u32 {
        let bytes = unsafe { self.state.memory.read::<u8, 4>(MIPS_REGISTER_SPACE, offset as u32) };
        u32::from_le_bytes(bytes)
    }

    #[inline]
    #[cfg(debug_assertions)]
    fn log(&mut self, _: &Instruction) {
        if !self.unconstrained && self.state.global_clk.is_multiple_of(10_000_000) {
            tracing::info!("clk = {} pc = 0x{:x?}", self.state.global_clk, self.state.pc);
        }
    }
}
