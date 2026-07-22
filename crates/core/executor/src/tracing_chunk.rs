//! `generate_records`'s typed-event construction.
//!
//! `TracingVM` wraps a `CoreVM<Oracle>` (already-decided shard: one `TracingVM` per
//! `crate::splicing::SplicedChunk`) and mirrors `Executor::emit_events` and its per-opcode-family
//! `emit_*` helpers, writing into the wrapped `CoreVM`'s own `record: ExecutionRecord` instead of
//! `Executor`'s. `Executor`'s own event-construction code is untouched -- this is a separate copy
//! for the `generate_records` pipeline.
//!
//! Not named `tracing.rs` to avoid shadowing the `tracing` crate, used pervasively elsewhere in
//! this crate via `tracing::debug_span!`/`tracing::info!`.

use std::{collections::VecDeque, sync::Arc};

use crate::{
    dependencies::{
        emit_branch_dependencies, emit_cloclz_dependencies, emit_divrem_dependencies,
        emit_jump_dependencies, emit_memory_dependencies, emit_misc_dependencies,
    },
    events::{
        AluEvent, BranchEvent, CompAluEvent, CpuEvent, JumpEvent, MemInstrEvent,
        MemoryInitializeFinalizeEvent, MemoryRecord, MemoryRecordEnum, MemoryWriteRecord,
        MiscEvent, MovCondEvent,
    },
    executor::LocalCounts,
    memory::{Memory, PagedMemory},
    record::{ExecutionRecord, MemoryAccessRecord},
    register::NUM_REGISTERS,
    splicing::SplicedChunk,
    vm::{CoreVM, MemValue, Oracle, StepOutcome},
    ExecutionError, Opcode, Program,
};

/// Emit the program's global memory initialize/finalize events into `record`, for every address
/// ever touched. Must only be called once, after the final shard's `TracingVM` finishes -- reads
/// `MinimalRunner`'s still-live final registers/memory (`MinimalRunner::registers`/`memory`/
/// `uninitialized_memory`), since that's the only place the full, final state of every touched
/// address exists (an `Oracle`-sourced `CoreVM` only ever sees a sequential replay, never a full
/// memory map). Mirrors `Executor::postprocess`'s memory-events section.
pub fn emit_globals(
    registers: &[MemValue; NUM_REGISTERS],
    registers_touched: &[bool; NUM_REGISTERS],
    memory: &PagedMemory<MemValue>,
    uninitialized_memory: &Memory<u32>,
    program: &Program,
    record: &mut ExecutionRecord,
) {
    let addr_0_final_record: MemoryRecord = registers[0].into();
    record
        .global_memory_finalize_events
        .push(MemoryInitializeFinalizeEvent::finalize_from_record(0, &addr_0_final_record));
    record.global_memory_initialize_events.push(MemoryInitializeFinalizeEvent::initialize(0, 0));

    for addr in 1..NUM_REGISTERS as u32 {
        if registers_touched[addr as usize] {
            if !program.image.contains_key(&addr) {
                let initial_value = uninitialized_memory.registers.get(addr).copied().unwrap_or(0);
                record
                    .global_memory_initialize_events
                    .push(MemoryInitializeFinalizeEvent::initialize(addr, initial_value));
            }
            let reg_record: MemoryRecord = registers[addr as usize].into();
            record
                .global_memory_finalize_events
                .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &reg_record));
        }
    }

    for addr in memory.keys() {
        if addr == 0 {
            continue;
        }
        if !program.image.contains_key(&addr) {
            let initial_value = uninitialized_memory.get(addr).copied().unwrap_or(0);
            record
                .global_memory_initialize_events
                .push(MemoryInitializeFinalizeEvent::initialize(addr, initial_value));
        }
        let mem_record: MemoryRecord = (*memory.get(addr).unwrap()).into();
        record
            .global_memory_finalize_events
            .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &mem_record));
    }
}

/// Replays one already-decided shard (a [`SplicedChunk`]) and emits its typed events.
pub struct TracingVM {
    core: CoreVM<Oracle>,
}

/// The result of tracing one [`SplicedChunk`].
pub struct TracedShard {
    pub record: ExecutionRecord,
    pub done: bool,
    /// Carried into the next `TracingVM`: `HINT_LEN`/`HINT_READ` replay against this stream and
    /// must stay positioned exactly where the previous shard left off.
    pub input_stream: VecDeque<Vec<u8>>,
}

impl TracingVM {
    #[must_use]
    pub fn new(program: Arc<Program>, chunk: SplicedChunk, input_stream: VecDeque<Vec<u8>>) -> Self {
        let mut core = CoreVM::new(program, Oracle::new(chunk.oracle));
        core.pc = chunk.pc_start;
        core.next_pc = chunk.next_pc_start;
        core.clk = chunk.initial_timestamp;
        core.initial_timestamp = chunk.initial_timestamp;
        core.global_clk = chunk.global_clk_start;
        // `CoreVM::new()` only seeded registers from the program's initial image, correct for
        // the first-ever shard but not any later one -- `TracingVM` doesn't run continuously
        // across shards the way `SplicingVM` does, so it needs the real mid-execution snapshot
        // `SplicingVM` captured at this shard's cut point.
        core.registers = chunk.registers_start;
        core.current_shard = chunk.shard;
        core.record.public_values.shard = chunk.shard;
        core.input_stream = input_stream;
        core.is_tracing = true;
        Self { core }
    }

    /// Replay every instruction in this shard, emitting typed events, until the shard's oracle is
    /// exhausted (or the program halts).
    ///
    /// # Errors
    /// Returns an error if replaying the chunk's instructions fails.
    pub fn trace(mut self) -> Result<TracedShard, ExecutionError> {
        loop {
            let outcome = self.core.step()?;
            self.emit_events(&outcome);
            if outcome.done || self.core.mem.remaining() == 0 {
                let done = outcome.done;
                // Mirrors `Executor::bump_record`: close out this shard's local-memory-access
                // chain into the record before handing it off.
                for (_, event) in self.core.local_memory_access.drain() {
                    self.core.record.cpu_local_memory_access.push(event);
                }
                return Ok(TracedShard {
                    record: self.core.record,
                    done,
                    input_stream: self.core.input_stream,
                });
            }
        }
    }

    /// Split off the syscalls/precompiles this shard doesn't already own (mirrors
    /// `crate::state::ExecutionState` bookkeeping); callers use `TracedShard` directly.
    #[must_use]
    pub fn local_counts(&self) -> &LocalCounts {
        &self.core.local_counts
    }

    // ---- ported verbatim from Executor::emit_events and its per-opcode-family helpers ----

    fn emit_events(&mut self, outcome: &StepOutcome) {
        let StepOutcome {
            instruction,
            clk,
            pc,
            next_pc,
            next_next_pc,
            a,
            b,
            c,
            hi_or_prev_a,
            memory_accesses,
            exit_code,
            syscall_code,
            num_extra_cycles,
            ..
        } = *outcome;

        self.core.record.first_instruction_pc.get_or_insert(pc);
        self.core.record.first_instruction_clk.get_or_insert(clk);
        self.core.record.last_next_pc = next_pc;
        self.core.record.last_exit_code = exit_code;
        self.core.record.last_timestamp = clk + 5 + u64::from(num_extra_cycles);

        // Opcodes whose chip has been migrated off of `CpuChip` no longer need a `CpuEvent`:
        // their own chip does its own program lookup, state chaining, and register access.
        let migrated_off_cpu_chip = matches!(
            instruction.opcode,
            Opcode::ADD
                | Opcode::SUB
                | Opcode::SLL
                | Opcode::XOR
                | Opcode::OR
                | Opcode::AND
                | Opcode::NOR
                | Opcode::SRL
                | Opcode::SRA
                | Opcode::ROR
                | Opcode::SLT
                | Opcode::SLTU
                | Opcode::CLZ
                | Opcode::CLO
                | Opcode::MUL
                | Opcode::MULT
                | Opcode::MULTU
                | Opcode::DIV
                | Opcode::DIVU
                | Opcode::MOD
                | Opcode::MODU
                | Opcode::BEQ
                | Opcode::BNE
                | Opcode::BLTZ
                | Opcode::BGEZ
                | Opcode::BLEZ
                | Opcode::BGTZ
                | Opcode::Jump
                | Opcode::Jumpi
                | Opcode::JumpDirect
                | Opcode::LB
                | Opcode::LBU
                | Opcode::LH
                | Opcode::LHU
                | Opcode::LW
                | Opcode::LWL
                | Opcode::LWR
                | Opcode::LL
                | Opcode::SB
                | Opcode::SH
                | Opcode::SW
                | Opcode::SWL
                | Opcode::SWR
                | Opcode::SC
                | Opcode::MEQ
                | Opcode::MNE
                | Opcode::WSBH
                | Opcode::SEXT
                | Opcode::EXT
                | Opcode::INS
                | Opcode::MADDU
                | Opcode::MSUBU
                | Opcode::MADD
                | Opcode::MSUB
                | Opcode::TEQ
                | Opcode::SYSCALL
        );
        if !migrated_off_cpu_chip {
            self.emit_cpu(
                clk,
                pc,
                next_pc,
                next_next_pc,
                a,
                b,
                c,
                hi_or_prev_a,
                memory_accesses,
                exit_code,
                num_extra_cycles,
            );
        }

        if instruction.is_alu_instruction() {
            self.emit_alu_event(clk, pc, next_pc, instruction.opcode, instruction.imm_b, hi_or_prev_a, a, b, c, memory_accesses);
        } else if instruction.is_memory_load_instruction() || instruction.is_memory_store_instruction() {
            self.emit_mem_instr_event(clk, pc, next_pc, instruction.opcode, a, b, c, hi_or_prev_a.unwrap_or(0), memory_accesses);
        } else if instruction.is_branch_instruction() {
            self.emit_branch_event(clk, pc, instruction.opcode, a, b, c, next_pc, next_next_pc, memory_accesses);
        } else if instruction.is_jump_instruction() {
            self.emit_jump_event(clk, pc, instruction.opcode, a, b, c, next_pc, next_next_pc, memory_accesses);
        } else if instruction.is_misc_instruction() {
            self.emit_misc_event(clk, pc, next_pc, instruction.opcode, a, b, c, hi_or_prev_a.unwrap_or(0), memory_accesses);
        } else if instruction.is_syscall_instruction() {
            self.emit_syscall_event(clk, pc, memory_accesses, syscall_code, b, c, next_pc);
        } else {
            unreachable!()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_cpu(
        &mut self,
        clk: u64,
        pc: u32,
        next_pc: u32,
        next_next_pc: u32,
        a: u32,
        b: u32,
        c: u32,
        hi_or_prev_a: Option<u32>,
        record: MemoryAccessRecord,
        exit_code: u32,
        num_extra_cycles: u32,
    ) {
        self.core.record.cpu_events.push(CpuEvent {
            clk,
            pc,
            next_pc,
            next_next_pc,
            a,
            a_record: record.a,
            b,
            b_record: record.b,
            c,
            c_record: record.c,
            hi: hi_or_prev_a,
            hi_record: record.hi,
            memory_record: record.memory,
            exit_code,
            num_extra_cycles,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_alu_event(
        &mut self,
        clk: u64,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        imm_b: bool,
        hi_or_prev_a: Option<u32>,
        a: u32,
        b: u32,
        c: u32,
        record: MemoryAccessRecord,
    ) {
        let event = AluEvent {
            shard: self.core.shard(),
            clk,
            pc,
            next_pc,
            opcode,
            hi: hi_or_prev_a.unwrap_or(0),
            a,
            b,
            c,
            a_record: record.a,
            b_record: record.b,
            c_record: record.c,
        };

        let (hi_access, hi_record_is_real) = match record.hi {
            Some(MemoryRecordEnum::Write(record)) => (record, true),
            _ => (MemoryWriteRecord::default(), false),
        };

        let event_comp = CompAluEvent {
            clk,
            shard: self.core.shard(),
            pc,
            next_pc,
            opcode,
            hi: hi_or_prev_a.unwrap_or(0),
            a,
            b,
            c,
            hi_record: hi_access,
            hi_record_is_real,
            a_record: record.a,
            b_record: record.b,
            c_record: record.c,
        };

        match opcode {
            Opcode::ADD if record.c.is_none() && !imm_b => {
                self.core.record.addi_events.push(event);
            }
            Opcode::ADD => {
                self.core.record.add_events.push(event);
            }
            Opcode::SUB => {
                self.core.record.sub_events.push(event);
            }
            Opcode::XOR | Opcode::OR | Opcode::AND | Opcode::NOR => {
                self.core.record.bitwise_events.push(event);
            }
            Opcode::SLL => {
                self.core.record.shift_left_events.push(event);
            }
            Opcode::SRL | Opcode::SRA | Opcode::ROR => {
                self.core.record.shift_right_events.push(event);
            }
            Opcode::SLT | Opcode::SLTU => {
                self.core.record.lt_events.push(event);
            }
            Opcode::MUL | Opcode::MULT | Opcode::MULTU => {
                self.core.record.mul_events.push(event_comp);
            }
            Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU => {
                self.core.record.divrem_events.push(event_comp);
                emit_divrem_dependencies(&mut self.core.record, event);
            }
            Opcode::CLZ | Opcode::CLO => {
                self.core.record.cloclz_events.push(event);
                emit_cloclz_dependencies(&mut self.core.record, event);
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_mem_instr_event(
        &mut self,
        clk: u64,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        prev_a_val: u32,
        record: MemoryAccessRecord,
    ) {
        let event = MemInstrEvent {
            shard: self.core.shard(),
            clk,
            pc,
            next_pc,
            opcode,
            a,
            b,
            c,
            mem_access: record.memory.expect("Must have memory access"),
            prev_a_val,
            a_record: record.a,
            b_record: record.b,
            c_record: record.c,
        };

        match opcode {
            Opcode::LW => self.core.record.load_word_events.push(event),
            Opcode::SW => self.core.record.store_word_events.push(event),
            _ => self.core.record.memory_instr_events.push(event),
        }
        emit_memory_dependencies(
            &mut self.core.record,
            event,
            record.memory.expect("Must have memory access").current_record(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_branch_event(
        &mut self,
        clk: u64,
        pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        next_pc: u32,
        next_next_pc: u32,
        record: MemoryAccessRecord,
    ) {
        let event = BranchEvent {
            shard: self.core.shard(),
            clk,
            pc,
            next_pc,
            next_next_pc,
            opcode,
            a,
            b,
            c,
            a_record: record.a,
            b_record: record.b,
            c_record: record.c,
        };
        self.core.record.branch_events.push(event);
        emit_branch_dependencies(&mut self.core.record, event);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_jump_event(
        &mut self,
        clk: u64,
        pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        next_pc: u32,
        next_next_pc: u32,
        record: MemoryAccessRecord,
    ) {
        let mut event =
            JumpEvent::new(self.core.shard(), clk, pc, next_pc, next_next_pc, opcode, a, b, c);
        event.a_record = record.a;
        event.b_record = record.b;
        event.c_record = record.c;
        self.core.record.jump_events.push(event);
        emit_jump_dependencies(&mut self.core.record, event);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_misc_event(
        &mut self,
        clk: u64,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        prev_a: u32,
        record: MemoryAccessRecord,
    ) {
        if matches!(opcode, Opcode::MNE | Opcode::MEQ | Opcode::WSBH) {
            let mut event =
                MovCondEvent::new(self.core.shard(), clk, pc, next_pc, opcode, a, b, c, prev_a);
            event.a_record = record.a;
            event.b_record = record.b;
            event.c_record = record.c;
            self.core.record.movcond_events.push(event);
        } else {
            let hi_access = match record.hi {
                Some(MemoryRecordEnum::Write(record)) => record,
                _ => MemoryWriteRecord::default(),
            };

            let mut event = MiscEvent::new(
                clk,
                self.core.shard(),
                pc,
                next_pc,
                opcode,
                a,
                b,
                c,
                prev_a,
                hi_access,
            );
            event.a_record = record.a;
            event.b_record = record.b;
            event.c_record = record.c;
            self.core.record.misc_events.push(event);
            emit_misc_dependencies(&mut self.core.record, event);
        }
    }

    fn emit_syscall_event(
        &mut self,
        clk: u64,
        pc: u32,
        record: MemoryAccessRecord,
        syscall_id: u32,
        arg1: u32,
        arg2: u32,
        next_pc: u32,
    ) {
        // Built directly (not via the `SyscallRuntime::syscall_event` trait method) because that
        // method reads `self.pc()` live -- correct when precompiles call it *during* `step()`
        // (before this instruction's pc gets advanced), but wrong here, where `emit_events` (and
        // so this call) always runs *after* `step()` has already advanced `self.core.pc`.
        let (write, is_real) = match record.a {
            Some(crate::events::MemoryRecordEnum::Write(record)) => (record, true),
            _ => (MemoryWriteRecord::default(), false),
        };
        let mut syscall_event = crate::events::SyscallEvent {
            pc,
            next_pc,
            shard: self.core.shard(),
            clk,
            a_record: write,
            a_record_is_real: is_real,
            b_record: None,
            c_record: None,
            syscall_id,
            arg1,
            arg2,
        };
        syscall_event.b_record = record.b;
        syscall_event.c_record = record.c;

        self.core.record.syscall_events.push(syscall_event);
    }
}
