//! `CoreVM`: the shared instruction-semantics core underlying `generate_records`'s `SplicingVM`
//! and `TracingVM` (see `splicing.rs`/`tracing_chunk.rs`). `MinimalRunner` uses its own,
//! independent `MinimalExecutor` instead (see `minimal/executor.rs`) -- only `SplicingVM`/
//! `TracingVM` replay an already-recorded [`MemValue`] log via [`Oracle`].
//!
//! `CoreVM<M: MemSource>` implements per-instruction semantics, generalized over where memory
//! reads/writes come from -- currently always a linear cursor over an oracle log (`Oracle`), but
//! kept generic since `MemSource` documents the exact contract a source must satisfy. It does not
//! construct typed chip events itself -- callers get a [`StepOutcome`] back from [`CoreVM::step`]
//! and decide what to do with it (cost accounting only, or full event construction).
//!
//! `Executor` keeps its own independent instruction-dispatch implementation for
//! `run()`/`run_fast()`/other non-`generate_records` callers and does not use `CoreVM`.

use std::sync::Arc;

use hashbrown::HashMap;

use crate::{
    events::{
        MemoryAccessPosition, MemoryLocalEvent, MemoryReadRecord, MemoryRecord, MemoryWriteRecord,
    },
    executor::LocalCounts,
    program::MAX_MEMORY,
    record::{ExecutionRecord, MemoryAccessRecord},
    register::NUM_REGISTERS,
    syscalls::{default_syscall_map, Syscall, SyscallCode, SyscallContext, SyscallRuntime},
    utils::sign_extend as sign_extend_fn,
    ExecutionError, Instruction, Opcode, Program, Register,
};

/// Where a [`CoreVM`]'s memory/register reads and writes come from.
///
/// `access` returns the record as it was *immediately before* this access -- the same
/// `prev_record` every existing `Executor` accessor already captures -- and `commit` is where the
/// *new* record (post-access) gets persisted, if this source has anywhere to persist it.
pub trait MemSource {
    /// The record at `addr` immediately before this access.
    fn access(&mut self, addr: u32) -> MemoryRecord;
    /// Persist the new record at `addr` after this access. No-op for [`Oracle`] -- there is
    /// nothing to persist during a replay pass, since every future access at this address already
    /// has its own oracle entry recorded by `MinimalRunner`.
    fn commit(&mut self, addr: u32, record: MemoryRecord);

    /// Seed the value `addr` should read as the first time it's touched (`SYSHINTREAD`). No-op
    /// for [`Oracle`] -- the oracle already encodes the correct first-touch value.
    fn seed_uninitialized(&mut self, addr: u32, value: u32) -> Result<(), ExecutionError>;

    /// Enter an unconstrained block: snapshot whatever's needed to undo memory writes made during
    /// it. No-op for [`Oracle`] -- nothing persists there in the first place, so there's nothing
    /// to undo.
    fn enter_unconstrained(&mut self) {}
    /// Exit an unconstrained block, undoing memory writes made since the matching
    /// `enter_unconstrained`.
    fn exit_unconstrained(&mut self) {}
}

/// A single buffered oracle entry: the record immediately before one memory/register access.
/// `value` is `u32`, matching `MemoryRecord` and the rest of this crate's MIPS32 words.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemValue {
    pub clk: u64,
    pub value: u32,
}

impl From<MemValue> for MemoryRecord {
    fn from(value: MemValue) -> Self {
        MemoryRecord { value: value.value, timestamp: value.clk }
    }
}

/// A replay of an already-recorded [`MemValue`] stream -- used by `SplicingVM`/`TracingVM`.
///
/// Each access just pops the next `MemValue` and turns it directly into a `MemoryRecord` -- no
/// bookkeeping of its own. This works because `MinimalExecutor` (`minimal/executor.rs`) already
/// buffered the *final* timestamp for every access, computed from real execution; there is
/// nothing left for `Oracle` to reconstruct. `commit` is correspondingly a no-op: with the
/// timestamp already correct in the buffer, there's nothing to persist for a later access at the
/// same address to look up.
pub struct Oracle {
    values: std::vec::IntoIter<MemValue>,
}

impl Oracle {
    #[must_use]
    pub fn new(values: Vec<MemValue>) -> Self {
        Self { values: values.into_iter() }
    }

    /// Number of values not yet consumed -- used by `SplicingVM` to know how many oracle entries
    /// a shard-cut piece spans.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.values.len()
    }
}

impl MemSource for Oracle {
    fn access(&mut self, addr: u32) -> MemoryRecord {
        let MemValue { clk, value } = self
            .values
            .next()
            .unwrap_or_else(|| panic!("oracle exhausted while accessing address {addr}"));
        MemoryRecord { value, timestamp: clk }
    }

    fn commit(&mut self, _addr: u32, _record: MemoryRecord) {}

    fn seed_uninitialized(&mut self, _addr: u32, _value: u32) -> Result<(), ExecutionError> {
        Ok(())
    }
}

/// Snapshot taken on entering an unconstrained block, restored on exit. `registers` is a full
/// snapshot rather than a sparse diff (unlike `MinimalExecutor`'s `unconstrained_reg_diff`) --
/// the array is small and `Copy`, so there's no real cost to simplicity here.
#[derive(Debug, Clone, Copy)]
struct UnconstrainedCtx {
    pc: u32,
    clk: u64,
    global_clk: u64,
    registers: [MemValue; NUM_REGISTERS],
}

/// Everything [`CoreVM::step`] produces about one retired instruction, for a caller to turn into
/// cost accounting ([`crate::splicing::SplicingVM`]) or typed events
/// ([`crate::tracing_chunk::TracingVM`]). Mirrors the parameter bundle `Executor::emit_events`
/// takes today.
pub struct StepOutcome {
    pub instruction: Instruction,
    pub clk: u64,
    pub pc: u32,
    pub next_pc: u32,
    pub next_next_pc: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub hi_or_prev_a: Option<u32>,
    pub memory_accesses: MemoryAccessRecord,
    pub exit_code: u32,
    pub syscall_code: u32,
    pub num_extra_cycles: u32,
    /// Whether the program halted on this instruction.
    pub done: bool,
}

/// The shared instruction-semantics core. See the module doc comment.
pub struct CoreVM<M: MemSource> {
    pub program: Arc<Program>,
    pub mem: M,
    /// Registers, kept separate from `mem`/`oracle_out`: a register's value is always a
    /// deterministic function of the instruction stream and already-oracle-buffered memory
    /// reads, so both `SplicingVM` (continuous across the whole execution) and `TracingVM`
    /// (fresh per shard, seeded from `SplicedChunk::registers_start`) recompute registers
    /// locally instead of needing them pre-recorded in the oracle.
    pub registers: [MemValue; NUM_REGISTERS],

    pub pc: u32,
    pub next_pc: u32,
    pub clk: u64,
    pub global_clk: u64,
    pub current_shard: u32,
    /// The value `clk` had when the current shard started. See `ExecutionState::initial_timestamp`'s
    /// doc comment (`Executor`'s equivalent) -- same role here.
    pub initial_timestamp: u64,
    pub next_is_delayslot: bool,
    pub exited: bool,

    pub unconstrained: bool,
    unconstrained_ctx: Option<UnconstrainedCtx>,

    /// Whether this `CoreVM` is being driven by `TracingVM` (building a real `ExecutionRecord`
    /// that ends up in the proof) as opposed to `MinimalRunner`/`SplicingVM` (which only need
    /// `record` as scratch space for syscalls like `COMMIT`). Gates
    /// `SyscallRuntime::is_recording_events` -- `SyscallContext::postprocess`'s outer/inner
    /// local-memory-access chain split (see its doc comment) must run for `TracingVM`, since
    /// skipping it merges a syscall's own memory touches into the surrounding chain instead of
    /// closing it at the syscall boundary.
    pub is_tracing: bool,

    /// Local-memory-access bookkeeping for the shard currently being stepped through. Consumed by
    /// `TracingVM`; ignored (but harmlessly maintained) by `MinimalRunner`/`SplicingVM`.
    pub local_memory_access: HashMap<u32, MemoryLocalEvent>,
    /// Event-count/cost bookkeeping for the shard currently being stepped through. Consumed by
    /// `SplicingVM`'s shard-cut decision; ignored by `MinimalRunner`/`TracingVM` (`TracingVM`
    /// already knows its shard boundaries from `SplicedChunk`).
    pub local_counts: LocalCounts,

    /// Scratch/real execution record: `TracingVM` uses this for real (its per-shard
    /// `ExecutionRecord`); `MinimalRunner`/`SplicingVM` still need it to satisfy syscalls like
    /// `COMMIT` that write straight into `record.public_values`, but never inspect it themselves.
    pub record: ExecutionRecord,
    /// The memory-access slots (a/b/c/hi/memory) touched by the instruction currently being
    /// stepped. Reset at the start of every `step()` call.
    pub memory_accesses: MemoryAccessRecord,

    pub syscall_map: HashMap<SyscallCode, Arc<dyn Syscall<Self>>>,

    pub input_stream: std::collections::VecDeque<Vec<u8>>,
    pub public_values_stream: Vec<u8>,
    pub io_buf: HashMap<u32, String>,

    pub deferred_proof_verification: crate::DeferredProofVerification,
}

impl<M: MemSource> CoreVM<M> {
    #[must_use]
    pub fn new(program: Arc<Program>, mem: M) -> Self {
        let pc = program.pc_start;
        let next_pc = program.next_pc;
        let record = ExecutionRecord::new(program.clone());
        let syscall_map = default_syscall_map::<Self>();
        // Seed registers from the program's initial image (e.g. `$sp`/`$brk`/`$heap`) --
        // deterministic and identical regardless of who's asking, so `SplicingVM` (which starts
        // truly at the beginning) gets the real values for free; `TracingVM` overwrites this
        // right after construction from `SplicedChunk::registers_start` since it doesn't start
        // at the beginning.
        let mut registers = [MemValue::default(); NUM_REGISTERS];
        for (&addr, &value) in &program.image {
            if addr < NUM_REGISTERS as u32 {
                registers[addr as usize] = MemValue { clk: 0, value };
            }
        }
        Self {
            program,
            mem,
            registers,
            pc,
            next_pc,
            // 1, not 0: `0` is the "never touched" sentinel `MemoryRecord::timestamp` (mirrors
            // `ExecutionState::new`). `MinimalRunner` is the only consumer that keeps this value
            // (`SplicingVM`/`TracingVM` overwrite it right after construction from `Chunk`/
            // `SplicedChunk` fields) -- its buffered `MemValue`s carry real timestamps computed
            // from this starting point, so it must agree with the `1`-based convention
            // `SplicingVM`'s own `pending_initial_timestamp` and `TracingVM`'s seeded
            // `initial_timestamp` already use for a program's first shard.
            clk: 1,
            global_clk: 0,
            current_shard: 1,
            initial_timestamp: 1,
            next_is_delayslot: false,
            exited: false,
            unconstrained: false,
            unconstrained_ctx: None,
            is_tracing: false,
            local_memory_access: HashMap::new(),
            local_counts: LocalCounts::default(),
            record,
            memory_accesses: MemoryAccessRecord::default(),
            syscall_map,
            input_stream: std::collections::VecDeque::new(),
            public_values_stream: Vec::new(),
            io_buf: HashMap::new(),
            deferred_proof_verification: crate::DeferredProofVerification::Enabled,
        }
    }

    #[must_use]
    pub fn shard(&self) -> u32 {
        self.current_shard
    }

    /// The current memory-access timestamp for a given access position. See `Executor::timestamp`'s
    /// doc comment -- same role here.
    #[must_use]
    pub const fn timestamp(&self, position: &MemoryAccessPosition) -> u64 {
        self.clk + *position as u64
    }

    fn fetch(&self) -> Instruction {
        self.program.fetch(self.pc)
    }

    // ---- primitive accessors, mirroring Executor::{mr,mw,rr_traced,rw_traced,mr_cpu,rr_cpu,mw_cpu,rw_cpu,register,word,byte} ----

    /// Read a word from memory and create an access record, tracking it into `local_memory_access`.
    pub fn mr(
        &mut self,
        addr: u32,
        external: bool,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        let is_register = addr < NUM_REGISTERS as u32;
        let prev_record: MemoryRecord = if is_register {
            self.registers[addr as usize].into()
        } else {
            self.mem.access(addr)
        };
        if !self.unconstrained
            && (prev_record.timestamp < self.initial_timestamp || external)
        {
            self.local_counts.local_mem += 1;
        }
        let record = MemoryRecord { value: prev_record.value, timestamp };
        if is_register {
            self.registers[addr as usize] = MemValue { clk: record.timestamp, value: record.value };
        } else {
            self.mem.commit(addr, record);
        }
        if !self.unconstrained {
            let local_memory_access =
                local_memory_access.unwrap_or(&mut self.local_memory_access);
            local_memory_access
                .entry(addr)
                .and_modify(|e| e.final_mem_access = record)
                .or_insert(MemoryLocalEvent { addr, initial_mem_access: prev_record, final_mem_access: record });
        }
        MemoryReadRecord::new(record.value, record.timestamp, prev_record.timestamp)
    }

    /// Write a word to memory and create an access record, tracking it into `local_memory_access`.
    pub fn mw(
        &mut self,
        addr: u32,
        value: u32,
        external: bool,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        let is_register = addr < NUM_REGISTERS as u32;
        let prev_record: MemoryRecord =
            if is_register { self.registers[addr as usize].into() } else { self.mem.access(addr) };
        if !self.unconstrained
            && (prev_record.timestamp < self.initial_timestamp || external)
        {
            self.local_counts.local_mem += 1;
        }
        let record = MemoryRecord { value, timestamp };
        if is_register {
            self.registers[addr as usize] = MemValue { clk: record.timestamp, value: record.value };
        } else {
            self.mem.commit(addr, record);
        }
        if !self.unconstrained {
            let local_memory_access =
                local_memory_access.unwrap_or(&mut self.local_memory_access);
            local_memory_access
                .entry(addr)
                .and_modify(|e| e.final_mem_access = record)
                .or_insert(MemoryLocalEvent { addr, initial_mem_access: prev_record, final_mem_access: record });
        }
        MemoryWriteRecord::new(record.value, record.timestamp, prev_record.value, prev_record.timestamp)
    }

    /// Read a register and create an access record (same shape as `mr`, registers are just
    /// addresses `< NUM_REGISTERS`, routed to `self.registers` instead of `self.mem`).
    pub fn rr_traced(
        &mut self,
        register: Register,
        external: bool,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        self.mr(register as u32, external, timestamp, local_memory_access)
    }

    /// Write a register and create an access record.
    pub fn rw_traced(
        &mut self,
        register: Register,
        value: u32,
        external: bool,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        let value = if register == Register::ZERO { 0 } else { value };
        self.mw(register as u32, value, external, timestamp, local_memory_access)
    }

    /// Get the current value of a register, without creating an access record. Doesn't consume
    /// an oracle entry (see `registers`'s doc comment) or participate in local-memory-access/cost
    /// bookkeeping.
    pub fn register(&mut self, register: Register) -> u32 {
        self.registers[register as u32 as usize].value
    }

    /// Get the current value of a word, without creating an access record.
    pub fn word(&mut self, addr: u32) -> u32 {
        self.mem.access(addr).value
    }

    /// Get the current value of a byte, without creating an access record.
    pub fn byte(&mut self, addr: u32) -> u8 {
        let word = self.word(addr - addr % 4);
        (word >> ((addr % 4) * 8)) as u8
    }

    fn mr_cpu(&mut self, addr: u32) -> u32 {
        let record = self.mr(addr, false, self.timestamp(&MemoryAccessPosition::Memory), None);
        self.memory_accesses.memory = Some(record.into());
        record.value
    }

    fn rr_cpu(&mut self, register: Register, position: MemoryAccessPosition) -> u32 {
        let record = self.rr_traced(register, false, self.timestamp(&position), None);
        match position {
            MemoryAccessPosition::A => self.memory_accesses.a = Some(record.into()),
            MemoryAccessPosition::B => self.memory_accesses.b = Some(record.into()),
            MemoryAccessPosition::C => self.memory_accesses.c = Some(record.into()),
            _ => unreachable!(),
        }
        record.value
    }

    fn mw_cpu(&mut self, addr: u32, value: u32) {
        let record = self.mw(addr, value, false, self.timestamp(&MemoryAccessPosition::Memory), None);
        self.memory_accesses.memory = Some(record.into());
    }

    fn rw_cpu(&mut self, register: Register, value: u32, position: MemoryAccessPosition) {
        let value = if register == Register::ZERO { 0 } else { value };
        let record = self.rw_traced(register, value, false, self.timestamp(&position), None);
        match position {
            MemoryAccessPosition::A => self.memory_accesses.a = Some(record.into()),
            MemoryAccessPosition::HI => self.memory_accesses.hi = Some(record.into()),
            _ => unreachable!(),
        }
    }

    // ---- instruction-family helpers, ported verbatim from Executor ----

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
            (rd, self.rr_cpu(rs1, MemoryAccessPosition::B), imm)
        } else {
            debug_assert!(instruction.imm_b && instruction.imm_c);
            (instruction.op_a.into(), instruction.op_b, instruction.op_c)
        }
    }

    fn alu_rw(&mut self, op: &Instruction, rd: Register, hi: u32, a: u32, b: u32, c: u32) -> (Option<u32>, u32, u32, u32) {
        let hi = if op.opcode.is_use_lo_hi_alu() {
            self.rw_cpu(Register::LO, a, MemoryAccessPosition::A);
            self.rw_cpu(Register::HI, hi, MemoryAccessPosition::HI);
            Some(hi)
        } else {
            self.rw_cpu(rd, a, MemoryAccessPosition::A);
            None
        };
        (hi, a, b, c)
    }

    fn branch_rr(&mut self, instruction: &Instruction) -> (u32, u32, u32) {
        let (src1, src2, target) =
            (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = if instruction.opcode.only_one_operand() {
            0
        } else {
            self.rr_cpu(src2, MemoryAccessPosition::B)
        };
        let a = self.rr_cpu(src1, MemoryAccessPosition::A);
        (a, b, target)
    }

    fn execute_alu(&mut self, instruction: &Instruction) -> Result<(Option<u32>, u32, u32, u32), ExecutionError> {
        let (rd, b, c) = self.alu_rr(instruction);
        if matches!(instruction.opcode, Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU) && c == 0 {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        let (a, hi) = match instruction.opcode {
            Opcode::ADD => (b.overflowing_add(c).0, 0),
            Opcode::SUB => (b.overflowing_sub(c).0, 0),
            Opcode::SLL => (b << (c & 0x1f), 0),
            Opcode::SRL => (b >> (c & 0x1F), 0),
            Opcode::SRA => {
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
                if b < c { (1, 0) } else { (0, 0) }
            }
            Opcode::SLT => {
                if (b as i32) < (c as i32) { (1, 0) } else { (0, 0) }
            }
            Opcode::MULT => {
                let out = (((b as i32) as i64) * ((c as i32) as i64)) as u64;
                (out as u32, (out >> 32) as u32)
            }
            Opcode::MULTU => {
                let out = b as u64 * c as u64;
                (out as u32, (out >> 32) as u32)
            }
            Opcode::DIV => (((b as i32) / (c as i32)) as u32, ((b as i32) % (c as i32)) as u32),
            Opcode::DIVU => (b / c, b % c),
            Opcode::MOD => (((b as i32) % (c as i32)) as u32, 0),
            Opcode::MODU => (b % c, 0),
            Opcode::AND => (b & c, 0),
            Opcode::OR => (b | c, 0),
            Opcode::XOR => (b ^ c, 0),
            Opcode::NOR => (!(b | c), 0),
            Opcode::CLZ => (b.leading_zeros(), 0),
            Opcode::CLO => (b.leading_ones(), 0),
            _ => unreachable!(),
        };
        Ok(self.alu_rw(instruction, rd, hi, a, b, c))
    }

    fn execute_load(&mut self, instruction: &Instruction) -> Result<(Option<u32>, u32, u32, u32), ExecutionError> {
        let (rt_reg, rs_reg, offset_ext) =
            (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let rs_raw = self.rr_cpu(rs_reg, MemoryAccessPosition::B);
        let rt = self.register(rt_reg);

        let addr = rs_raw.wrapping_add(offset_ext);
        let aligned_addr = addr & 0xFFFF_FFFC;

        if aligned_addr + 3 > MAX_MEMORY as u32 {
            return Err(ExecutionError::MemoryOutOfBoundsAccess(addr as u64));
        }

        let mem = self.mr_cpu(aligned_addr);
        let rs = addr;

        let val = match instruction.opcode {
            Opcode::LH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LH, addr));
                }
                let mem_fc = |i: u32| -> u32 { sign_extend_fn::<16>((mem >> (i * 8)) & 0xffff) };
                mem_fc(rs & 2)
            }
            Opcode::LWL => {
                let out = |i: u32| -> u32 {
                    let val = mem << (24 - i * 8);
                    let mask: u32 = 0xFFFFFFFF_u32 << (24 - i * 8);
                    (rt & (!mask)) | val
                };
                out(rs & 3)
            }
            Opcode::LW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LW, addr));
                }
                mem
            }
            Opcode::LBU => {
                let out = |i: u32| -> u32 { (mem >> (i * 8)) & 0xff };
                out(rs & 3)
            }
            Opcode::LHU => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LHU, addr));
                }
                let mem_fc = |i: u32| -> u32 { (mem >> (i * 8)) & 0xffff };
                mem_fc(rs & 2)
            }
            Opcode::LWR => {
                let out = |i: u32| -> u32 {
                    let val = mem >> (i * 8);
                    let mask = 0xFFFFFFFF_u32 >> (i * 8);
                    (rt & (!mask)) | val
                };
                out(rs & 3)
            }
            Opcode::LL => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LL, addr));
                }
                mem
            }
            Opcode::LB => {
                let out = |i: u32| -> u32 { sign_extend_fn::<8>((mem >> (i * 8)) & 0xff) };
                out(rs & 3)
            }
            _ => unreachable!(),
        };
        self.rw_cpu(rt_reg, val, MemoryAccessPosition::A);
        Ok((Some(rt), val, rs_raw, offset_ext))
    }

    fn execute_store(&mut self, instruction: &Instruction) -> Result<(Option<u32>, u32, u32, u32), ExecutionError> {
        let (rt_reg, rs_reg, offset_ext) =
            (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let rs = self.rr_cpu(rs_reg, MemoryAccessPosition::B);
        let rt = if instruction.opcode == Opcode::SC {
            self.register(rt_reg)
        } else {
            self.rr_cpu(rt_reg, MemoryAccessPosition::A)
        };

        let addr = rs.wrapping_add(offset_ext);
        let aligned_addr = addr & 0xFFFF_FFFC;

        let mem = self.word(aligned_addr);

        let val = match instruction.opcode {
            Opcode::SB => {
                let out = |i: u32| -> u32 {
                    let val = (rt & 0xff) << (i * 8);
                    let mask = 0xFFFFFFFF_u32 ^ (0xff << (i * 8));
                    (mem & mask) | val
                };
                out(addr & 3)
            }
            Opcode::SH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SH, addr));
                }
                let mem_fc = |i: u32| -> u32 {
                    let val = (rt & 0xffff) << (i * 8);
                    let mask = 0xFFFFFFFF_u32 ^ (0xffff << (i * 8));
                    (mem & mask) | val
                };
                mem_fc(addr & 2)
            }
            Opcode::SWL => {
                let out = |i: u32| -> u32 {
                    let val = rt >> (24 - i * 8);
                    let mask = 0xFFFFFFFF_u32 >> (24 - i * 8);
                    (mem & (!mask)) | val
                };
                out(addr & 3)
            }
            Opcode::SW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SW, addr));
                }
                rt
            }
            Opcode::SWR => {
                let out = |i: u32| -> u32 {
                    let val = rt << (i * 8);
                    let mask = 0xFFFFFFFF_u32 << (i * 8);
                    (mem & (!mask)) | val
                };
                out(addr & 3)
            }
            Opcode::SC => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SC, addr));
                }
                rt
            }
            _ => unreachable!("unexpected store opcode: {:?}", instruction.opcode),
        };

        if aligned_addr + 3 > MAX_MEMORY as u32 {
            return Err(ExecutionError::MemoryOutOfBoundsAccess(addr as u64));
        }

        self.mw_cpu(aligned_addr, val);
        if instruction.opcode == Opcode::SC {
            self.rw_cpu(rt_reg, 1, MemoryAccessPosition::A);
            Ok((Some(rt), 1, rs, offset_ext))
        } else {
            Ok((Some(rt), rt, rs, offset_ext))
        }
    }

    fn execute_branch(&mut self, instruction: &Instruction, next_pc: u32, mut next_next_pc: u32) -> (u32, u32, u32, u32) {
        let (src1, src2, offset) = self.branch_rr(instruction);
        let should_jump = match instruction.opcode {
            Opcode::BEQ => src1 == src2,
            Opcode::BNE => src1 != src2,
            Opcode::BGEZ => (src1 as i32) >= 0,
            Opcode::BLEZ => (src1 as i32) <= 0,
            Opcode::BGTZ => (src1 as i32) > 0,
            Opcode::BLTZ => (src1 as i32) < 0,
            _ => unreachable!(),
        };
        if should_jump {
            next_next_pc = offset.wrapping_add(next_pc);
        }
        (src1, src2, offset, next_next_pc)
    }

    fn execute_jump(&mut self, instruction: &Instruction) -> (u32, u32, u32, u32) {
        let (link, target) = (instruction.op_a.into(), (instruction.op_b as u8).into());
        let target_pc = self.rr_cpu(target, MemoryAccessPosition::B);
        let return_pc = self.next_pc.wrapping_add(4);
        self.rw_cpu(link, return_pc, MemoryAccessPosition::A);
        (return_pc, target_pc, 0, target_pc)
    }

    fn execute_jumpi(&mut self, instruction: &Instruction) -> (u32, u32, u32, u32) {
        let (link, target_pc) = (instruction.op_a.into(), instruction.op_b);
        let return_pc = self.next_pc.wrapping_add(4);
        self.rw_cpu(link, return_pc, MemoryAccessPosition::A);
        (return_pc, target_pc, 0, target_pc)
    }

    fn execute_jump_direct(&mut self, instruction: &Instruction) -> (u32, u32, u32, u32) {
        let (link, offset) = (instruction.op_a.into(), instruction.op_b);
        let target_pc = offset.wrapping_add(self.next_pc);
        let return_pc = self.next_pc.wrapping_add(4);
        self.rw_cpu(link, return_pc, MemoryAccessPosition::A);
        (return_pc, offset, 0, target_pc)
    }

    fn execute_condmov(&mut self, instruction: &Instruction) -> (Option<u32>, u32, u32, u32) {
        let (rd, rs, rt) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let a = self.register(rd);
        let prev_a = a;
        let c = self.rr_cpu(rt, MemoryAccessPosition::C);
        let b = self.rr_cpu(rs, MemoryAccessPosition::B);
        let mov = match instruction.opcode {
            Opcode::MEQ => c == 0,
            Opcode::MNE => c != 0,
            _ => unreachable!(),
        };
        let a = if mov { b } else { a };
        self.rw_cpu(rd, a, MemoryAccessPosition::A);
        (Some(prev_a), a, b, c)
    }

    fn execute_wsbh(&mut self, instruction: &Instruction) -> (u32, u32, u32) {
        let (rd, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let a = (((b >> 16) & 0xFF) << 24)
            | (((b >> 24) & 0xFF) << 16)
            | ((b & 0xFF) << 8)
            | ((b >> 8) & 0xFF);
        self.rw_cpu(rd, a, MemoryAccessPosition::A);
        (a, b, 0)
    }

    fn execute_ext(&mut self, instruction: &Instruction) -> Result<(u32, u32, u32), ExecutionError> {
        let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let msbd = c >> 5;
        let lsb = c & 0x1f;
        if msbd + lsb >= 32 {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        let mask_msb = if msbd + lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msbd + lsb + 1)) - 1 };
        let a = (b & mask_msb) >> lsb;
        self.rw_cpu(rd, a, MemoryAccessPosition::A);
        Ok((a, b, c))
    }

    fn execute_ins(&mut self, instruction: &Instruction) -> Result<(Option<u32>, u32, u32, u32), ExecutionError> {
        let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let a = self.register(rd);
        let prev_a = a;
        let msb = c >> 5;
        let lsb = c & 0x1f;
        if msb < lsb {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        let mask = if msb - lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msb - lsb + 1)) - 1 };
        let mask_field = mask << lsb;
        let a = (a & !mask_field) | ((b << lsb) & mask_field);
        self.rw_cpu(rd, a, MemoryAccessPosition::A);
        Ok((Some(prev_a), a, b, c))
    }

    fn execute_sext(&mut self, instruction: &Instruction) -> (u32, u32, u32) {
        let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let a = if c > 0 { (b & 0xffff) as i16 as i32 as u32 } else { (b & 0xff) as i8 as i32 as u32 };
        self.rw_cpu(rd, a, MemoryAccessPosition::A);
        (a, b, c)
    }

    fn execute_teq(&mut self, instruction: &Instruction) -> Result<(u32, u32, u32), ExecutionError> {
        let (rs, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());
        let src2 = self.rr_cpu(rt, MemoryAccessPosition::B);
        let src1 = self.rr_cpu(rs, MemoryAccessPosition::A);
        if src1 == src2 {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        Ok((src1, src2, 0))
    }

    fn execute_maddu(&mut self, instruction: &Instruction) -> (Option<u32>, u32, u32, u32) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr_cpu(rs, MemoryAccessPosition::C);
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let multiply = b as u64 * c as u64;
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = multiply.wrapping_add(addend);
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw_cpu(lo, out_lo, MemoryAccessPosition::A);
        self.rw_cpu(Register::HI, out_hi, MemoryAccessPosition::HI);
        (Some(lo_val), out_lo, b, c)
    }

    fn execute_msubu(&mut self, instruction: &Instruction) -> (Option<u32>, u32, u32, u32) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr_cpu(rs, MemoryAccessPosition::C);
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let multiply = b as u64 * c as u64;
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = addend.wrapping_sub(multiply);
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw_cpu(lo, out_lo, MemoryAccessPosition::A);
        self.rw_cpu(Register::HI, out_hi, MemoryAccessPosition::HI);
        (Some(lo_val), out_lo, b, c)
    }

    fn execute_madd(&mut self, instruction: &Instruction) -> (Option<u32>, u32, u32, u32) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr_cpu(rs, MemoryAccessPosition::C);
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let multiply = (b as i32 as i64) * (c as i32 as i64);
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = multiply.wrapping_add(addend as i64) as u64;
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw_cpu(lo, out_lo, MemoryAccessPosition::A);
        self.rw_cpu(Register::HI, out_hi, MemoryAccessPosition::HI);
        (Some(lo_val), out_lo, b, c)
    }

    fn execute_msub(&mut self, instruction: &Instruction) -> (Option<u32>, u32, u32, u32) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr_cpu(rs, MemoryAccessPosition::C);
        let b = self.rr_cpu(rt, MemoryAccessPosition::B);
        let multiply = (b as i32 as i64) * (c as i32 as i64);
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = (addend as i64).wrapping_sub(multiply) as u64;
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw_cpu(lo, out_lo, MemoryAccessPosition::A);
        self.rw_cpu(Register::HI, out_hi, MemoryAccessPosition::HI);
        (Some(lo_val), out_lo, b, c)
    }

    fn get_syscall(&self, code: SyscallCode) -> Option<Arc<dyn Syscall<Self>>> {
        self.syscall_map.get(&code).cloned()
    }

    /// Step one instruction, returning what happened (for the caller to cost-account or emit
    /// typed events from) and whether the program halted.
    ///
    /// # Errors
    /// Returns an error if the instruction is invalid or the program traps.
    pub fn step(&mut self) -> Result<StepOutcome, ExecutionError> {
        let instruction = self.fetch();

        let pc = self.pc;
        let clk = self.clk;
        let mut exit_code = 0u32;
        let mut next_pc = self.next_pc;
        let mut next_next_pc = self.next_pc + 4;
        let (mut a, mut b, mut c) = (0, 0, 0);
        let mut hi_or_prev_a = None;
        let mut syscall_code = 0u32;
        let mut num_extra_cycles = 0u32;

        self.next_is_delayslot = false;
        self.memory_accesses = MemoryAccessRecord::default();

        if !self.unconstrained {
            self.local_counts.event_counts[instruction.opcode] += 1;
            if instruction.opcode == Opcode::ADD && instruction.imm_c && !instruction.imm_b {
                self.local_counts.addi_events += 1;
            }
            if instruction.is_memory_load_instruction() {
                self.local_counts.event_counts[Opcode::ADD] += 2;
            } else if instruction.is_branch_cmp_instruction() {
                self.local_counts.event_counts[Opcode::ADD] += 1;
                self.local_counts.event_counts[Opcode::SLT] += 2;
            } else if instruction.is_mov_cond_instruction() {
                self.local_counts.event_counts[Opcode::ADD] += 1;
            } else if instruction.opcode == Opcode::EXT {
                self.local_counts.event_counts[Opcode::SLL] += 1;
                self.local_counts.event_counts[Opcode::SRL] += 1;
            } else if instruction.is_cloclz_instruction() {
                self.local_counts.event_counts[Opcode::SRL] += 1;
            } else if instruction.is_maddsubu_instruction() {
                self.local_counts.event_counts[Opcode::MULTU] += 1;
            } else if instruction.opcode == Opcode::INS {
                self.local_counts.event_counts[Opcode::ROR] += 2;
                self.local_counts.event_counts[Opcode::SLL] += 1;
                self.local_counts.event_counts[Opcode::SRL] += 2;
                self.local_counts.event_counts[Opcode::ADD] += 1;
            } else if instruction.opcode == Opcode::DIV {
                self.local_counts.event_counts[Opcode::MULT] += 2;
                self.local_counts.event_counts[Opcode::ADD] += 2;
                self.local_counts.event_counts[Opcode::SLTU] += 1;
            } else if instruction.opcode == Opcode::DIVU {
                self.local_counts.event_counts[Opcode::MULTU] += 2;
                self.local_counts.event_counts[Opcode::ADD] += 2;
                self.local_counts.event_counts[Opcode::SLTU] += 1;
            } else if instruction.is_maddsub_instruction() {
                self.local_counts.event_counts[Opcode::MULT] += 1;
            } else if instruction.opcode == Opcode::JumpDirect {
                self.local_counts.event_counts[Opcode::ADD] += 1;
            }
        }

        if instruction.is_alu_instruction() {
            (hi_or_prev_a, a, b, c) = self.execute_alu(&instruction)?;
        } else if instruction.is_memory_load_instruction() {
            (hi_or_prev_a, a, b, c) = self.execute_load(&instruction)?;
        } else if instruction.is_memory_store_instruction() {
            (hi_or_prev_a, a, b, c) = self.execute_store(&instruction)?;
        } else if instruction.is_branch_instruction() {
            (a, b, c, next_next_pc) = self.execute_branch(&instruction, next_pc, next_next_pc);
            self.next_is_delayslot = true;
        } else if instruction.is_jump_instruction() {
            (a, b, c, next_next_pc) = if instruction.opcode == Opcode::Jump {
                self.execute_jump(&instruction)
            } else if instruction.opcode == Opcode::Jumpi {
                self.execute_jumpi(&instruction)
            } else {
                self.execute_jump_direct(&instruction)
            };
            self.next_is_delayslot = true;
        } else if instruction.is_mov_cond_instruction() {
            (hi_or_prev_a, a, b, c) = self.execute_condmov(&instruction);
        } else if instruction.is_misc_instruction() {
            if instruction.opcode == Opcode::WSBH {
                (a, b, c) = self.execute_wsbh(&instruction);
            } else if instruction.opcode == Opcode::EXT {
                (a, b, c) = self.execute_ext(&instruction)?;
            } else if instruction.opcode == Opcode::MADDU {
                (hi_or_prev_a, a, b, c) = self.execute_maddu(&instruction);
            } else if instruction.opcode == Opcode::INS {
                (hi_or_prev_a, a, b, c) = self.execute_ins(&instruction)?;
            } else if instruction.opcode == Opcode::SEXT {
                (a, b, c) = self.execute_sext(&instruction);
            } else if instruction.opcode == Opcode::TEQ {
                (a, b, c) = self.execute_teq(&instruction)?;
            } else if instruction.opcode == Opcode::MSUBU {
                (hi_or_prev_a, a, b, c) = self.execute_msubu(&instruction);
            } else if instruction.opcode == Opcode::MADD {
                (hi_or_prev_a, a, b, c) = self.execute_madd(&instruction);
            } else if instruction.opcode == Opcode::MSUB {
                (hi_or_prev_a, a, b, c) = self.execute_msub(&instruction);
            }
        } else if instruction.opcode == Opcode::SYSCALL {
            let syscall_id = self.register(Register::V0);
            c = self.rr_cpu(Register::A1, MemoryAccessPosition::C);
            b = self.rr_cpu(Register::A0, MemoryAccessPosition::B);
            let syscall = SyscallCode::from_u32(syscall_id);
            let mut prev_a = syscall_id;

            if self.unconstrained
                && (syscall != SyscallCode::EXIT_UNCONSTRAINED && syscall != SyscallCode::WRITE)
            {
                return Err(ExecutionError::InvalidSyscallUsage(syscall_id as u64));
            }

            let syscall_impl = self.get_syscall(syscall);
            syscall_code = syscall.syscall_id();
            let mut precompile_rt = SyscallContext::new(self);
            let (precompile_next_pc, precompile_cycles, returned_exit_code) =
                if let Some(syscall_impl) = syscall_impl {
                    let res = syscall_impl.execute(&mut precompile_rt, syscall, b, c)?;
                    a = res.unwrap_or(syscall_id);

                    if syscall == SyscallCode::HALT && precompile_rt.exit_code != 0 {
                        return Err(ExecutionError::HaltWithNonZeroExitCode(precompile_rt.exit_code));
                    }

                    (precompile_rt.next_pc, syscall_impl.num_extra_cycles(), precompile_rt.exit_code)
                } else {
                    return Err(ExecutionError::UnsupportedSyscall(syscall_id));
                };

            if syscall == SyscallCode::HALT && returned_exit_code == 0 {
                self.exited = true;
            }

            if syscall == SyscallCode::EXIT_UNCONSTRAINED {
                b = self.register(Register::A0);
                c = self.register(Register::A1);
                prev_a = self.register(Register::V0);
            }

            self.rw_cpu(Register::V0, a, MemoryAccessPosition::A);
            next_pc = precompile_next_pc;
            next_next_pc = precompile_next_pc + 4;
            self.clk += u64::from(precompile_cycles);
            num_extra_cycles = precompile_cycles;
            exit_code = returned_exit_code;
            hi_or_prev_a = Some(prev_a);
        } else if instruction.opcode == Opcode::UNIMPL {
            return Err(ExecutionError::UnsupportedInstruction(instruction.op_c));
        } else {
            unreachable!()
        }

        if next_next_pc == 0 {
            return Err(ExecutionError::NullPointerReference());
        }

        self.pc = next_pc;
        self.next_pc = next_next_pc;
        self.clk += 5;
        self.global_clk += 1;

        let done = self.pc == 0
            || self.exited
            || self.pc.wrapping_sub(self.program.pc_base) >= (self.program.instructions.len() * 4) as u32;
        if done && self.unconstrained {
            return Err(ExecutionError::EndInUnconstrained());
        }

        Ok(StepOutcome {
            instruction,
            clk,
            pc,
            next_pc,
            next_next_pc,
            a,
            b,
            c,
            hi_or_prev_a,
            memory_accesses: self.memory_accesses,
            exit_code,
            syscall_code,
            num_extra_cycles,
            done,
        })
    }
}

impl<M: MemSource> SyscallRuntime for CoreVM<M> {
    fn shard(&self) -> u32 {
        self.current_shard
    }

    fn clk(&self) -> u64 {
        self.clk
    }

    fn timestamp(&self) -> u64 {
        self.clk
    }

    fn pc(&self) -> u32 {
        self.pc
    }

    fn is_unconstrained(&self) -> bool {
        self.unconstrained
    }

    fn program(&self) -> &Program {
        &self.program
    }

    fn global_clk(&self) -> u64 {
        self.global_clk
    }

    fn mr(
        &mut self,
        addr: u32,
        external: bool,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        CoreVM::mr(self, addr, external, clk, local_memory_access)
    }

    fn mw(
        &mut self,
        addr: u32,
        value: u32,
        external: bool,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        CoreVM::mw(self, addr, value, external, clk, local_memory_access)
    }

    fn rr_traced(
        &mut self,
        register: Register,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        CoreVM::rr_traced(self, register, true, clk, local_memory_access)
    }

    fn rw_traced(
        &mut self,
        register: Register,
        value: u32,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        CoreVM::rw_traced(self, register, value, true, clk, local_memory_access)
    }

    fn register(&mut self, register: Register) -> u32 {
        CoreVM::register(self, register)
    }

    fn word(&mut self, addr: u32) -> u32 {
        CoreVM::word(self, addr)
    }

    fn byte(&mut self, addr: u32) -> u8 {
        CoreVM::byte(self, addr)
    }

    fn record_mut(&mut self) -> &mut ExecutionRecord {
        &mut self.record
    }

    fn is_recording_events(&self) -> bool {
        self.is_tracing
    }

    fn outer_local_memory_access(&mut self) -> &mut HashMap<u32, MemoryLocalEvent> {
        &mut self.local_memory_access
    }

    fn syscall_event(
        &self,
        clk: u64,
        a_record: Option<crate::events::MemoryRecordEnum>,
        next_pc: u32,
        syscall_id: u32,
        arg1: u32,
        arg2: u32,
    ) -> crate::events::SyscallEvent {
        let (write, is_real) = match a_record {
            Some(crate::events::MemoryRecordEnum::Write(record)) => (record, true),
            _ => (MemoryWriteRecord::default(), false),
        };
        crate::events::SyscallEvent {
            pc: self.pc,
            next_pc,
            shard: self.current_shard,
            clk,
            a_record: write,
            a_record_is_real: is_real,
            b_record: None,
            c_record: None,
            syscall_id,
            arg1,
            arg2,
        }
    }

    fn enter_unconstrained(&mut self) {
        assert!(!self.unconstrained, "Unconstrained block is already active.");
        self.unconstrained = true;
        self.unconstrained_ctx = Some(UnconstrainedCtx {
            pc: self.pc,
            clk: self.clk,
            global_clk: self.global_clk,
            registers: self.registers,
        });
        self.mem.enter_unconstrained();
    }

    fn exit_unconstrained(&mut self) {
        if let Some(ctx) = self.unconstrained_ctx.take() {
            self.pc = ctx.pc;
            self.clk = ctx.clk;
            self.global_clk = ctx.global_clk;
            self.registers = ctx.registers;
            self.mem.exit_unconstrained();
            self.unconstrained = false;
        }
    }

    fn seed_uninitialized(&mut self, addr: u32, value: u32) -> Result<(), ExecutionError> {
        self.mem.seed_uninitialized(addr, value)
    }

    fn peek_input(&self) -> Option<&Vec<u8>> {
        self.input_stream.front()
    }

    fn consume_input(&mut self) -> Option<Vec<u8>> {
        self.input_stream.pop_front()
    }

    fn push_hint_input(&mut self, bytes: Vec<u8>) {
        self.input_stream.push_front(bytes);
    }

    fn write_public_values(&mut self, bytes: &[u8]) {
        self.public_values_stream.extend_from_slice(bytes);
    }

    // `stdout_line`/`stderr_line`/`cycle_tracker_start`/`cycle_tracker_end` stay at
    // `SyscallRuntime`'s no-op defaults: `CoreVM<M>` only ever replays an already-recorded
    // instruction stream (`M = Oracle`, used by `SplicingVM`/`TracingVM`), so these host-visible
    // side effects (which must happen exactly once) belong to `MinimalExecutor`'s real pass, not
    // here.

    fn io_buf_push(&mut self, fd: u32, s: &str) -> Vec<String> {
        let entry = self.io_buf.entry(fd).or_default();
        entry.push_str(s);
        if entry.contains('\n') {
            let prev_buf = std::mem::take(entry);
            let mut lines = prev_buf.split('\n').collect::<Vec<&str>>();
            let last = lines.pop().unwrap_or("");
            *entry = last.to_string();
            lines.into_iter().map(std::string::ToString::to_string).collect::<Vec<String>>()
        } else {
            vec![]
        }
    }

    fn invoke_hook(&mut self, _fd: u32, _buf: &[u8]) -> Result<Option<Vec<Vec<u8>>>, ExecutionError> {
        // Custom host hooks are not supported here. Any `WRITE` to a hook-registered fd falls
        // through to the "unknown file descriptor" warning.
        Ok(None)
    }

    fn cycle_tracker_report(&mut self, _name: &str, _total_cycles: u64) {}

    fn verify_deferred_proof(&mut self, _vkey: [u32; 8], _pv_digest: [u32; 8]) -> Result<(), ExecutionError> {
        // `CoreVM<Oracle>` doesn't carry a proof stream at all (no `generate_records` caller
        // currently feeds one in); this is a no-op until that's wired up.
        Ok(())
    }
}
