use std::{
    fs::File, io::{BufWriter, Write}, str::FromStr, sync::Arc
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
    }, hook::{HookEnv, HookRegistry}, memory::{Entry, Memory}, pad_mips_event_counts, record::{ExecutionRecord, MemoryAccessRecord}, sign_extend, state::{ExecutionState, ForkState}, subproof::SubproofVerifier, syscalls::{Syscall, SyscallCode, SyscallContext, default_syscall_map}, vm::{memory::{AddressMap, GuestMemory, config::{MemoryConfig, MIPS_REGISTER_AS}}, state::{VmCheckpointState, VmSimpleState}}
};
use crate::NoOpSubproofVerifier;

/// An executor for the MIPS zkVM.
///
/// The executor is responsible for executing a user program and tracing important events which
/// occur during execution (i.e., memory reads, alu operations, etc).
pub struct RecordGenerator<'a, MEM = GuestMemory>  {
    /// The program.
    pub program: Arc<Program>,

    /// A buffer for stdout and stderr IO.
    pub io_buf: HashMap<u32, String>,

    /// The state of the execution.
    pub state: VmCheckpointState<MEM>,

    /// The current trace of the execution that is being collected.
    pub record: ExecutionRecord,

    /// The collected records, split by cpu cycles.
    pub records: Vec<ExecutionRecord>,

    /// Local memory access events.
    pub local_memory_access: HashMap<u32, MemoryLocalEvent>,

    /// The memory accesses for the current cycle.
    pub memory_accesses: MemoryAccessRecord,

    /// Whether the runtime is in constrained mode or not.
    ///
    /// In unconstrained mode, any events, clock, register, or memory changes are reset after
    /// leaving the unconstrained block. The only thing preserved is written to the input
    /// stream.
    pub unconstrained: bool,

    /// Whether we should write to the report.
    pub print_report: bool,

    /// The maximum number of shards to execute at once.
    pub shard_batch_size: u32,

    /// The mapping between syscall codes and their implementations.
    pub syscall_map: HashMap<SyscallCode, Arc<dyn Syscall>>,

    /// Statistics for event counts.
    pub local_counts: LocalCounts,

    /// Verifier used to sanity check `verify_zkm_proof` during runtime.
    pub subproof_verifier: Option<&'a dyn SubproofVerifier>,

    /// Report of the program execution.
    pub report: ExecutionReport,

    /// A buffer for writing trace events to a file.
    pub trace_buf: Option<BufWriter<File>>,
}

impl<'a> RecordGenerator<'a>  {
    /// Create a new [`VmExecutor`] from a program and options.
    #[must_use]
    pub fn new(program: Program, opts: ZKMCoreOpts, state: VmCheckpointState) -> Self {
        // Create a shared reference to the program.
        let program = Arc::new(program);

        // Create a default record with the program.
        let record = ExecutionRecord::new(program.clone());

        // Determine the maximum number of cycles for any syscall.
        let syscall_map = default_syscall_map();

        // If `TRACE_FILE`` is set, initialize the trace buffer.
        let trace_buf = if let Ok(trace_file) = std::env::var("TRACE_FILE") {
            let file = File::create(trace_file).unwrap();
            Some(BufWriter::new(file))
        } else {
            None
        };

        Self {
            program,
            io_buf: HashMap::new(),
            state,
            record,
            records: vec![],
            local_memory_access: HashMap::new(),
            memory_accesses: MemoryAccessRecord::default(),
            unconstrained: false,
            print_report: true,
            report: ExecutionReport::default(),
            shard_batch_size: opts.shard_batch_size as u32,
            syscall_map,
            local_counts: LocalCounts::default(),
            // We already passed the deferred proof verifier when creating checkpoints, so the proofs were
            // already verified. So here we use a noop verifier to not print any warnings.
            subproof_verifier: Some(&NoOpSubproofVerifier),
            trace_buf,
        }
    }

    pub fn run(&mut self) -> Result<(Vec<ExecutionRecord>, ExecutionReport), ExecutionError> {
        // Execute from the checkpoint.
        let _done = self.execute()?;
        Ok((std::mem::take(&mut self.records), std::mem::take(&mut self.report)))
    }

    /// Executes one cycle of the program, returning whether the program has finished.
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn execute(&mut self) -> Result<bool, ExecutionError> {
        // Get the program.
        let program = self.program.clone();

        // Get the current shard.
        let start_shard = self.state.current_shard;

        // Loop until we've executed `self.shard_batch_size` shards.
        let mut done = false;
        let mut num_shards_executed = 0;
        loop {
            if self.execute_cycle()? {
                done = true;
                break;
            }

            if self.inc_shard_if_need() {
                self.bump_record();
                num_shards_executed += 1;
                if num_shards_executed >= self.shard_batch_size {
                    break;
                }
            }
        }

        // Get the final public values.
        let public_values = self.record.public_values;

        if done {
            self.postprocess();

            // Push the remaining execution record with memory initialize & finalize events.
            self.bump_record();
            log::debug!("last step {}", self.state.global_clk);
        }

        // Push the remaining execution record, if there are any CPU events.
        if !self.record.cpu_events.is_empty() {
            self.bump_record();
        }

        // Set the global public values for all shards.
        let mut last_next_pc = 0;
        let mut last_exit_code = 0;
        for (i, record) in self.records.iter_mut().enumerate() {
            record.program = program.clone();
            record.public_values = public_values;
            record.public_values.committed_value_digest = public_values.committed_value_digest;
            record.public_values.deferred_proofs_digest = public_values.deferred_proofs_digest;
            record.public_values.execution_shard = start_shard + i as u32;
            if record.cpu_events.is_empty() {
                record.public_values.start_pc = last_next_pc;
                record.public_values.next_pc = last_next_pc;
                record.public_values.exit_code = last_exit_code;
            } else {
                record.public_values.start_pc = record.cpu_events[0].pc;
                record.public_values.next_pc = record.cpu_events.last().unwrap().next_pc;
                record.public_values.exit_code = record.cpu_events.last().unwrap().exit_code;
                last_next_pc = record.public_values.next_pc;
                last_exit_code = record.public_values.exit_code;
            }
        }

        Ok(done)
    }

    fn postprocess(&mut self) {
        // Flush remaining stdout/stderr
        for (fd, buf) in &self.io_buf {
            if !buf.is_empty() {
                match fd {
                    1 => {
                        println!("stdout: {buf}");
                    }
                    2 => {
                        println!("stderr: {buf}");
                    }
                    _ => {}
                }
            }
        }

        // Flush trace buf
        if let Some(ref mut buf) = self.trace_buf {
            buf.flush().unwrap();
        }

        // Ensure that all proofs and input bytes were read, otherwise warn the user.
        if self.state.proof_stream_ptr != self.state.proof_stream.len() {
            tracing::warn!(
                "Not all proofs were read. Proving will fail during recursion. Did you pass too
        many proofs in or forget to call verify_zkm_proof?"
            );
        }
        if self.state.input_stream_ptr != self.state.input_stream.len() {
            tracing::warn!("Not all input bytes were read. Read bytes: {} / {}", self.state.input_stream_ptr, self.state.input_stream.len());
        }

        // SECTION: Set up all MemoryInitializeFinalizeEvents needed for memory argument.
        let memory_finalize_events = &mut self.record.global_memory_finalize_events;

        // We handle the addr = 0 case separately, as we constrain it to be 0 in the first row
        // of the memory finalize table so it must be first in the array of events.
        let addr_0_record = self.state.memory.get(0);

        let addr_0_final_record = match addr_0_record {
            Some(record) => record,
            None => &MemoryRecord { value: 0, shard: 0, timestamp: 1 },
        };
        memory_finalize_events
            .push(MemoryInitializeFinalizeEvent::finalize_from_record(0, addr_0_final_record));

        let memory_initialize_events = &mut self.record.global_memory_initialize_events;
        let addr_0_initialize_event = MemoryInitializeFinalizeEvent::initialize(0, 0);
        memory_initialize_events.push(addr_0_initialize_event);

        // Count the number of touched memory addresses manually, since `PagedMemory` doesn't
        // already know its length.
        self.report.touched_memory_addresses = 0;
        for addr in 1..NUM_REGISTERS as u32 {
            let record = self.state.memory.registers.get(addr);
            if let Some(record) = record {
                if self.print_report {
                    self.report.touched_memory_addresses += 1;
                }
                // Program memory is initialized in the MemoryProgram chip and doesn't require
                // any events, so we only send init events for other memory
                // addresses.
                if !self.record.program.image.contains_key(&addr) {
                    let initial_value =
                        self.state.uninitialized_memory.registers.get(addr).unwrap_or(&0);
                    memory_initialize_events
                        .push(MemoryInitializeFinalizeEvent::initialize(addr, *initial_value));
                }

                memory_finalize_events
                    .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, record));
            }
        }
        for addr in self.state.memory.page_table.keys() {
            self.report.touched_memory_addresses += 1;
            if addr == 0 {
                // Handled above.
                continue;
            }

            // Program memory is initialized in the MemoryProgram chip and doesn't require any
            // events, so we only send init events for other memory addresses.
            if !self.record.program.image.contains_key(&addr) {
                let initial_value = self.state.uninitialized_memory.get(addr).unwrap_or(&0);
                memory_initialize_events
                    .push(MemoryInitializeFinalizeEvent::initialize(addr, *initial_value));
            }

            let record = *self.state.memory.get(addr).unwrap();
            memory_finalize_events
                .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &record));
        }
    }

    /// Bump the record.
    pub fn bump_record(&mut self) {
        self.local_counts = LocalCounts::default();
        // Copy all of the existing local memory accesses to the record's local_memory_access vec.
        for (_, event) in self.local_memory_access.drain() {
            self.record.cpu_local_memory_access.push(event);
        }

        let removed_record =
            std::mem::replace(&mut self.record, ExecutionRecord::new(self.program.clone()));
        let public_values = removed_record.public_values;
        self.record.public_values = public_values;
        self.records.push(removed_record);
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

        let done = self.state.pc == 0
            || self.state.exited
            || self.state.pc.wrapping_sub(self.program.pc_base)
                >= (self.program.instructions.len() * 4) as u32;
        if done {
            self.state.max_clks.push(self.state.clk);

            if self.unconstrained {
                tracing::error!("program ended in unconstrained mode at clk {}", self.state.global_clk);
                return Err(ExecutionError::EndInUnconstrained());
            }
        }

        Ok(done)
    }

    fn inc_shard_if_need(&mut self) -> bool {
        if !self.state.max_clks.is_empty() && self.state.clk >= self.state.max_clks[self.state.max_clks_index as usize] {
            self.state.current_shard += 1;
            self.state.clk = 0;
            self.state.max_clks_index += 1;
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
        let record = self.rr_traced(register, self.shard(), self.timestamp(&position), None);
        if !self.unconstrained {
            match position {
                MemoryAccessPosition::A => self.memory_accesses.a = Some(record.into()),
                MemoryAccessPosition::B => self.memory_accesses.b = Some(record.into()),
                MemoryAccessPosition::C => self.memory_accesses.c = Some(record.into()),
                _ => unreachable!(),
            }
        }
        record.value
    }

    /// Read a register and create an access record.
    ///
    /// Assumes that self.mode IS [`ExecutorMode::Trace`].
    pub fn rr_traced(
        &mut self,
        register: Register,
        shard: u32,
        timestamp: u32,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        // Get the memory record entry.
        let addr = register as u32;
        // let entry = self.state.memory.registers.entry(addr);
        let rs = self.state.vm_read::<u8, 4>(MIPS_REGISTER_AS, addr);
        let value = u32::from_le_bytes(rs);

        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        // if self.unconstrained {
        //     let record = match entry {
        //         Entry::Occupied(ref entry) => Some(entry.get()),
        //         Entry::Vacant(_) => None,
        //     };
        //     self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        // }

        // If it's the first time accessing this address, initialize previous values.
        // let record: &mut MemoryRecord = match entry {
        //     Entry::Occupied(entry) => entry.into_mut(),
        //     Entry::Vacant(entry) => {
        //         // If addr has a specific value to be initialized with, use that, otherwise 0.
        //         let value = self.state.uninitialized_memory.registers.get(addr).unwrap_or(&0);
        //         entry.insert(MemoryRecord { value: *value, shard: 0, timestamp: 0 })
        //     }
        // };

        // let prev_record = *record;
        // record.shard = shard;
        // record.timestamp = timestamp;

        let prev_record = MemoryRecord { value, shard: 0, timestamp: 0 };
        let record = MemoryRecord { value, shard, timestamp };

        if !self.unconstrained {
            let local_memory_access = if let Some(local_memory_access) = local_memory_access {
                local_memory_access
            } else {
                &mut self.local_memory_access
            };
            local_memory_access
                .entry(addr)
                .and_modify(|e| {
                    e.final_mem_access = record;
                })
                .or_insert(MemoryLocalEvent {
                    addr,
                    initial_mem_access: prev_record,
                    final_mem_access: record,
                });
        }

        // Construct the memory read record.
        MemoryReadRecord::new(
            record.value,
            record.shard,
            record.timestamp,
            prev_record.shard,
            prev_record.timestamp,
        )
    }

    /// Get the current shard.
    #[must_use]
    #[inline]
    pub fn shard(&self) -> u32 {
        self.state.current_shard
    }

    /// Get the current timestamp for a given memory access position.
    #[must_use]
    #[inline]
    pub const fn timestamp(&self, position: &MemoryAccessPosition) -> u32 {
        self.state.clk + *position as u32
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
        self.state.vm_write::<u8, 4>(MIPS_REGISTER_AS, register as u32, &rd);
    }

    pub fn register(&self, offset: usize) -> u32 {
        let bytes = unsafe { self.state.memory.read::<u8, 4>(MIPS_REGISTER_AS, offset as u32) };
        u32::from_le_bytes(bytes)
    }

    #[inline]
    #[cfg(debug_assertions)]
    fn log(&mut self, _: &Instruction) {
        // Write the current program counter to the trace buffer for the cycle tracer.
        if let Some(ref mut buf) = self.trace_buf {
            if !self.unconstrained {
                buf.write_all(&u32::to_be_bytes(self.state.pc)).unwrap();
            }
        }

        if !self.unconstrained && self.state.global_clk.is_multiple_of(10_000_000) {
            tracing::info!("clk = {} pc = 0x{:x?}", self.state.global_clk, self.state.pc);
        }
    }
}
