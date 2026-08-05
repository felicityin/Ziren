//! `MinimalExecutor` -- a portable (plain Rust interpreter, no JIT) real, full-speed interpreter
//! that runs a program once and emits an oracle log ([`crate::trace::TraceChunk`]) instead of AIR
//! events. Every register/ALU computation is genuinely executed (never faked); only RAM
//! (page-table) accesses are logged, as preimages -- registers need zero logging, a plain
//! load/store needs exactly one preimage entry (the new value is a pure function of a live
//! register + the preimage, recomputable by any later replayer).
//!
//! # Syscall coverage
//!
//! The full ISA (every opcode family covered by the pure-compute functions in `crate::vm`) is
//! implemented for real. Syscalls are **not** all implemented yet: `HALT` and the `WRITE`
//! syscall's `FD_PUBLIC_VALUES` case are real; every other syscall (precompiles, Linux shims,
//! hints, unconstrained enter/exit) is a documented no-op fallback (`a0 = syscall_id`, no side
//! effects) for now. Extend the dispatch in [`ecall`] as needed.

#![allow(dead_code)]

mod ecall;

use std::sync::Arc;

use crate::{
    events::MemoryRecord,
    memory::PagedMemory,
    opcode::Opcode,
    register::{Register, NUM_REGISTERS},
    syscalls::default_syscall_map,
    trace::{MemValue, TraceChunk},
    vm, ExecutionError, Instruction, Program,
};

/// Bound on how many oracle-log entries are pre-reserved for a chunk. `max_trace_size` is a
/// cutoff, not a reservation hint -- callers may pass a very large or sentinel value to mean "no
/// cutoff", so the reservation itself is capped independently of it.
const MAX_PREALLOCATED_ORACLE_LOG: u64 = 1 << 20;

/// A live, full-speed interpreter. See the module doc for scope.
pub(crate) struct MinimalExecutor {
    program: Arc<Program>,
    registers: [u32; NUM_REGISTERS],
    page_table: PagedMemory<MemoryRecord>,
    pc: u32,
    next_pc: u32,
    clk: u64,
    next_is_delayslot: bool,
    exited: bool,
    /// Set once `is_done()` has fired *after* at least one instruction has retired -- see
    /// `try_execute_chunk`'s doc comment on why this must be distinct from `is_done()` itself
    /// (checking `is_done()` before ever running an instruction is wrong for any program whose
    /// `pc_start == 0`, since `0` is also the halt-pc sentinel).
    finished: bool,
    max_syscall_cycles: u32,
    /// The oracle log accumulated so far in the *current* chunk (reset by `try_execute_chunk`
    /// after each chunk is taken).
    oracle_log: Vec<MemValue>,
    max_trace_size: u64,
    public_values_stream: Vec<u8>,
}

impl MinimalExecutor {
    #[must_use]
    pub(crate) fn new(program: Arc<Program>, max_trace_size: u64) -> Self {
        // `program.image` addresses below `NUM_REGISTERS` are register-index seeds (e.g. the
        // ELF loader's computed initial `BRK`), not RAM -- mirrors `Memory::insert`'s dispatch,
        // which `Executor::initialize` relies on for exactly this. Routing everything into the
        // page table unconditionally (as an earlier version of this function did) leaves those
        // registers at `0`, which is wrong and was caught by `SysBrk`-adjacent real-ELF programs
        // computing wild addresses from a bogus zero `BRK`.
        let mut registers = [0u32; NUM_REGISTERS];
        let mut page_table = PagedMemory::new_preallocated();
        for (&addr, &value) in &program.image {
            if (addr as usize) < NUM_REGISTERS {
                registers[addr as usize] = value;
            } else {
                page_table.insert(addr, MemoryRecord { value, timestamp: 0 });
            }
        }

        // Mirrors `Executor::with_context`'s exact computation so `bump_clk_high_if_need`'s
        // window-advance threshold matches the legacy executor byte-for-byte (see the migration
        // plan's note on `7df6fd1d`) -- reusing the *value*, not the dispatch table itself.
        let max_syscall_cycles =
            default_syscall_map().values().map(|s| s.num_extra_cycles()).max().unwrap_or(0);

        Self {
            pc: program.pc_start,
            next_pc: program.pc_start.wrapping_add(4),
            program,
            registers,
            page_table,
            clk: 1, // `clk == 0` is the "never touched" sentinel, matches `ExecutionState::new`.
            next_is_delayslot: false,
            exited: false,
            finished: false,
            max_syscall_cycles,
            oracle_log: Vec::with_capacity(
                max_trace_size.min(MAX_PREALLOCATED_ORACLE_LOG) as usize
            ),
            max_trace_size,
            public_values_stream: Vec::new(),
        }
    }

    #[must_use]
    pub(crate) fn registers(&self) -> [u32; NUM_REGISTERS] {
        self.registers
    }

    #[must_use]
    pub(crate) fn pc(&self) -> u32 {
        self.pc
    }

    #[must_use]
    pub(crate) fn clk(&self) -> u64 {
        self.clk
    }

    /// The `max_syscall_cycles` this run used for `bump_clk_high_if_need`. `CoreVM` must be
    /// constructed with the *same* value or the two can silently disagree on `clk_high` window
    /// boundaries -- exposed so tests/callers never have to independently recompute (and
    /// potentially drift from) it.
    #[must_use]
    pub(crate) fn max_syscall_cycles(&self) -> u32 {
        self.max_syscall_cycles
    }

    #[must_use]
    pub(crate) fn is_done(&self) -> bool {
        self.pc == 0
            || self.exited
            || self.pc.wrapping_sub(self.program.pc_base)
                >= (self.program.instructions.len() * 4) as u32
    }

    #[must_use]
    pub(crate) fn public_values_stream(&self) -> &[u8] {
        &self.public_values_stream
    }

    /// Same digest formula as `golden::run_golden`'s `memory_digest` (page-table only,
    /// `timestamp != 0`, sorted by address) -- lets tests compare the two directly.
    #[must_use]
    pub(crate) fn memory_digest(&self) -> u64 {
        use std::hash::{DefaultHasher, Hash, Hasher};
        let mut touched: Vec<(u32, u32, u64)> = self
            .page_table
            .clone()
            .into_iter()
            .filter(|(_, record)| record.timestamp != 0)
            .map(|(addr, record)| (addr, record.value, record.timestamp))
            .collect();
        touched.sort_unstable_by_key(|&(addr, _, _)| addr);
        let mut hasher = DefaultHasher::new();
        touched.hash(&mut hasher);
        hasher.finish()
    }

    // ---- register file (never oracle-logged) ----

    fn reg(&self, r: Register) -> u32 {
        self.registers[r as usize]
    }

    fn set_reg(&mut self, r: Register, value: u32) {
        // `$zero` is hardware-wired to 0 (mirrors `Executor::rw_cpu`'s identical guard) -- real
        // code does write to it sometimes (`op_a == 0` idioms like `add $zero, ...`, handled by
        // `AluX0Chip` on the AIR side), and every later read of `$zero` must still see 0 or
        // downstream computation silently corrupts.
        let value = if r == Register::ZERO { 0 } else { value };
        self.registers[r as usize] = value;
    }

    // ---- RAM (oracle-logged) ----

    /// Read a word from RAM, logging its preimage. Matches `Executor::mr_cpu`'s semantics
    /// (timestamped at `MemoryAccessPosition::Memory`, i.e. exactly `self.clk`).
    fn mr(&mut self, addr: u32) -> u32 {
        let record = self.page_table.entry(addr).or_insert(MemoryRecord { value: 0, timestamp: 0 });
        self.oracle_log.push(MemValue { clk: record.timestamp, value: record.value });
        record.timestamp = self.clk;
        record.value
    }

    /// Peek a word from RAM without updating its timestamp *or* logging anything -- matches
    /// `Executor::word`'s untracked-peek semantics exactly. Used **only** by store instructions
    /// to read the current word for byte/half merging, where the immediately-following `mw` logs
    /// that exact same preimage anyway -- logging here too would double-log an identical entry.
    fn word_peek(&self, addr: u32) -> u32 {
        self.page_table.get(addr).map_or(0, |r| r.value)
    }

    /// Write a word to RAM, logging only its preimage (the new value is always a pure function of
    /// a live register + this preimage, recomputable by any replayer -- see the module doc).
    fn mw(&mut self, addr: u32, value: u32) {
        let record = self.page_table.entry(addr).or_insert(MemoryRecord { value: 0, timestamp: 0 });
        self.oracle_log.push(MemValue { clk: record.timestamp, value: record.value });
        record.value = value;
        record.timestamp = self.clk;
    }

    /// Read a word from RAM for a syscall (`WRITE`'s memory-to-bytes extraction), *without*
    /// updating its timestamp (matches `Executor::word`'s untracked-peek semantics exactly) but
    /// *does* log the preimage as an oracle entry -- unlike `word_peek`, nothing else logs this
    /// value, and `CoreVM` has no backing RAM at all to recover it from otherwise.
    fn mr_log_only(&mut self, addr: u32) -> u32 {
        let record = self.page_table.entry(addr).or_insert(MemoryRecord { value: 0, timestamp: 0 });
        self.oracle_log.push(MemValue { clk: record.timestamp, value: record.value });
        record.value
    }

    fn byte_peek(&mut self, addr: u32) -> u8 {
        let word = self.mr_log_only(addr - addr % 4);
        (word >> ((addr % 4) * 8)) as u8
    }

    // ---- clk / delay-slot-safe chunk cutoff ----

    /// Whether the trace buffer is full *and* it is safe to cut here: a chunk boundary must never
    /// split a branch/jump from its delay slot, mirroring the same `!next_is_delayslot` gate the
    /// legacy executor's shard-cut check uses.
    fn chunk_full(&self) -> bool {
        self.oracle_log.len() as u64 >= self.max_trace_size && !self.next_is_delayslot
    }

    /// Runs until the trace buffer fills (at a delay-slot-safe boundary) or the program halts,
    /// returning the resulting chunk, or `None` if the program was already done.
    ///
    /// # Errors
    ///
    /// Propagates any [`ExecutionError`] from executing an instruction.
    pub(crate) fn try_execute_chunk(&mut self) -> Result<Option<TraceChunk>, ExecutionError> {
        if self.finished {
            return Ok(None);
        }
        let start_registers = self.registers;
        let pc_start = self.pc;
        let clk_start = self.clk;

        // Do-while, matching `Executor::execute`'s `loop { if self.execute_cycle()? {...} }`:
        // `is_done()`'s `pc == 0` arm is also the *initial* `pc` for any program whose
        // `pc_start == 0` (i.e. most of this crate's synthetic test programs), so it must never
        // be consulted before at least one instruction has actually retired.
        loop {
            self.execute_instruction()?;
            if self.is_done() {
                self.finished = true;
                break;
            }
            if self.chunk_full() {
                break;
            }
        }

        let chunk = TraceChunk {
            start_registers,
            pc_start,
            clk_start,
            clk_end: self.clk,
            mem_reads: std::mem::replace(
                &mut self.oracle_log,
                Vec::with_capacity(self.max_trace_size.min(MAX_PREALLOCATED_ORACLE_LOG) as usize),
            )
            .into(),
        };
        Ok(Some(chunk))
    }

    // ---- main interpreter loop ----

    fn execute_instruction(&mut self) -> Result<(), ExecutionError> {
        let instruction = self.program.fetch(self.pc);
        self.clk = vm::bump_clk_high_if_need(self.clk, self.max_syscall_cycles);
        self.execute_operation(&instruction)?;
        self.clk += 5;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_operation(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let next_pc_in = self.next_pc;
        let mut next_next_pc = self.next_pc.wrapping_add(4);
        self.next_is_delayslot = false;

        if instruction.is_alu_instruction() {
            let (rd, b, c) = self.alu_operands(instruction);
            let (a, hi) = vm::alu_compute(instruction.opcode, b, c)?;
            self.alu_write(instruction.opcode, rd, a, hi);
        } else if instruction.is_memory_load_instruction() {
            self.execute_load(instruction)?;
        } else if instruction.is_memory_store_instruction() {
            self.execute_store(instruction)?;
        } else if instruction.is_branch_instruction() {
            let rs: Register = instruction.op_a.into();
            let src1 = self.reg(rs);
            let src2 = if instruction.opcode.only_one_operand() {
                0
            } else {
                self.reg((instruction.op_b as u8).into())
            };
            let offset = instruction.op_c;
            if vm::branch_taken(instruction.opcode, src1, src2) {
                next_next_pc = vm::branch_target(next_pc_in, offset);
            }
            self.next_is_delayslot = true;
        } else if instruction.is_jump_instruction() {
            let link: Register = instruction.op_a.into();
            let (return_pc, target) = match instruction.opcode {
                Opcode::Jump => {
                    let target_reg: Register = (instruction.op_b as u8).into();
                    let target_pc = self.reg(target_reg);
                    vm::jump_jr_result(next_pc_in, target_pc)
                }
                Opcode::Jumpi => vm::jump_jumpi_result(next_pc_in, instruction.op_b),
                Opcode::JumpDirect => vm::jump_direct_result(next_pc_in, instruction.op_b),
                _ => unreachable!("not a jump opcode: {:?}", instruction.opcode),
            };
            self.set_reg(link, return_pc);
            next_next_pc = target;
            self.next_is_delayslot = true;
        } else if instruction.is_mov_cond_instruction() {
            let rd: Register = instruction.op_a.into();
            let rs: Register = (instruction.op_b as u8).into();
            let rt: Register = (instruction.op_c as u8).into();
            let prev_a = self.reg(rd);
            let b = self.reg(rs);
            let c = self.reg(rt);
            let a = vm::condmov_result(instruction.opcode, prev_a, b, c);
            self.set_reg(rd, a);
        } else if instruction.is_misc_instruction() {
            self.execute_misc(instruction)?;
        } else if instruction.is_syscall_instruction() {
            let syscall_next_pc = self.execute_syscall()?;
            next_next_pc = syscall_next_pc.wrapping_add(4);
            self.pc = syscall_next_pc;
            self.next_pc = next_next_pc;
            return Ok(());
        } else {
            return Err(ExecutionError::UnsupportedInstruction(instruction.opcode as u32));
        }

        if next_next_pc == 0 {
            return Err(ExecutionError::NullPointerReference());
        }
        self.pc = next_pc_in;
        self.next_pc = next_next_pc;
        Ok(())
    }

    /// Mirrors `Executor::alu_rr`'s three operand-decoding shapes (register-register,
    /// register-immediate, immediate-immediate).
    fn alu_operands(&self, instruction: &Instruction) -> (Register, u32, u32) {
        if !instruction.imm_c {
            let rd = instruction.op_a.into();
            let b = self.reg((instruction.op_b as u8).into());
            let c = self.reg((instruction.op_c as u8).into());
            (rd, b, c)
        } else if !instruction.imm_b {
            let rd = instruction.op_a.into();
            let b = self.reg((instruction.op_b as u8).into());
            (rd, b, instruction.op_c)
        } else {
            (instruction.op_a.into(), instruction.op_b, instruction.op_c)
        }
    }

    /// Mirrors `Executor::alu_rw`: dual-result opcodes write LO/HI, everything else writes `rd`.
    fn alu_write(&mut self, opcode: Opcode, rd: Register, a: u32, hi: u32) {
        if opcode.is_use_lo_hi_alu() {
            self.set_reg(Register::LO, a);
            self.set_reg(Register::HI, hi);
        } else {
            self.set_reg(rd, a);
        }
    }

    fn execute_load(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs_raw = self.reg(rs_reg);
        let rt = self.reg(rt_reg);

        let addr = rs_raw.wrapping_add(offset);
        let aligned_addr = addr & 0xFFFF_FFFC;
        if aligned_addr as usize + 3 > crate::program::MAX_MEMORY {
            return Err(ExecutionError::MemoryOutOfBoundsAccess(addr as u64));
        }
        let mem = self.mr(aligned_addr);
        let rs = addr;

        let val = match instruction.opcode {
            Opcode::LH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LH, addr));
                }
                sign_extend::<16>((mem >> ((rs & 2) * 8)) & 0xffff)
            }
            Opcode::LWL => {
                let i = rs & 3;
                let val = mem << (24 - i * 8);
                let mask: u32 = 0xFFFF_FFFF_u32 << (24 - i * 8);
                (rt & (!mask)) | val
            }
            Opcode::LW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LW, addr));
                }
                mem
            }
            Opcode::LBU => (mem >> ((rs & 3) * 8)) & 0xff,
            Opcode::LHU => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LHU, addr));
                }
                (mem >> ((rs & 2) * 8)) & 0xffff
            }
            Opcode::LWR => {
                let i = rs & 3;
                let val = mem >> (i * 8);
                let mask = 0xFFFF_FFFF_u32 >> (i * 8);
                (rt & (!mask)) | val
            }
            Opcode::LL => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LL, addr));
                }
                mem
            }
            Opcode::LB => sign_extend::<8>((mem >> ((rs & 3) * 8)) & 0xff),
            _ => unreachable!("not a load opcode: {:?}", instruction.opcode),
        };
        self.set_reg(rt_reg, val);
        Ok(())
    }

    fn execute_store(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs = self.reg(rs_reg);
        let rt = self.reg(rt_reg);

        let addr = rs.wrapping_add(offset);
        let aligned_addr = addr & 0xFFFF_FFFC;
        let mem = self.word_peek(aligned_addr);

        let val = match instruction.opcode {
            Opcode::SB => {
                let i = addr & 3;
                let val = (rt & 0xff) << (i * 8);
                let mask = 0xFFFF_FFFF_u32 ^ (0xff << (i * 8));
                (mem & mask) | val
            }
            Opcode::SH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SH, addr));
                }
                let i = addr & 2;
                let val = (rt & 0xffff) << (i * 8);
                let mask = 0xFFFF_FFFF_u32 ^ (0xffff << (i * 8));
                (mem & mask) | val
            }
            Opcode::SWL => {
                let i = addr & 3;
                let val = rt >> (24 - i * 8);
                let mask = 0xFFFF_FFFF_u32 >> (24 - i * 8);
                (mem & (!mask)) | val
            }
            Opcode::SW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SW, addr));
                }
                rt
            }
            Opcode::SWR => {
                let i = addr & 3;
                let val = rt << (i * 8);
                let mask = 0xFFFF_FFFF_u32 << (i * 8);
                (mem & (!mask)) | val
            }
            Opcode::SC => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SC, addr));
                }
                rt
            }
            _ => unreachable!("not a store opcode: {:?}", instruction.opcode),
        };

        if aligned_addr as usize + 3 > crate::program::MAX_MEMORY {
            return Err(ExecutionError::MemoryOutOfBoundsAccess(addr as u64));
        }
        self.mw(aligned_addr, val);
        if instruction.opcode == Opcode::SC {
            self.set_reg(rt_reg, 1);
        }
        Ok(())
    }

    fn execute_misc(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        if instruction.opcode == Opcode::WSBH {
            let rd: Register = instruction.op_a.into();
            let rt: Register = (instruction.op_b as u8).into();
            let b = self.reg(rt);
            self.set_reg(rd, vm::wsbh(b));
            return Ok(());
        }

        let rd: Register = instruction.op_a.into();
        let rt: Register = (instruction.op_b as u8).into();
        let c = instruction.op_c;
        match instruction.opcode {
            Opcode::SEXT => {
                let b = self.reg(rt);
                self.set_reg(rd, vm::sext(b, c));
            }
            Opcode::EXT => {
                let b = self.reg(rt);
                self.set_reg(rd, vm::ext(b, c)?);
            }
            Opcode::INS => {
                let b = self.reg(rt);
                let a = self.reg(rd);
                self.set_reg(rd, vm::ins(a, b, c)?);
            }
            Opcode::TEQ => {
                // `execute_teq`'s unusual encoding: `rs = op_a`, `rt = op_b` (no destination).
                let rs: Register = instruction.op_a.into();
                let rt: Register = (instruction.op_b as u8).into();
                let src2 = self.reg(rt);
                let src1 = self.reg(rs);
                vm::teq(src1, src2)?;
            }
            Opcode::MADDU | Opcode::MSUBU | Opcode::MADD | Opcode::MSUB => {
                let lo_reg: Register = instruction.op_a.into();
                let rs: Register = (instruction.op_c as u8).into();
                let c = self.reg(rs);
                let b = self.reg(rt);
                let lo = self.reg(Register::LO);
                let hi = self.reg(Register::HI);
                let (out_lo, out_hi) = match instruction.opcode {
                    Opcode::MADDU => vm::maddu(b, c, lo, hi),
                    Opcode::MSUBU => vm::msubu(b, c, lo, hi),
                    Opcode::MADD => vm::madd(b, c, lo, hi),
                    Opcode::MSUB => vm::msub(b, c, lo, hi),
                    _ => unreachable!(),
                };
                self.set_reg(lo_reg, out_lo);
                self.set_reg(Register::HI, out_hi);
            }
            _ => unreachable!("not a misc opcode: {:?}", instruction.opcode),
        }
        Ok(())
    }
}

/// Mirrors `executor.rs`'s free `sign_extend` helper.
fn sign_extend<const BITS: u32>(value: u32) -> u32 {
    let shift = 32 - BITS;
    (((value << shift) as i32) >> shift) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        golden::run_golden,
        programs::tests::{fibonacci_program, halt_only_program, hello_world_program, simple_program},
        Program,
    };

    /// Runs a `MinimalExecutor` to completion (looping `try_execute_chunk` with a small buffer so
    /// the multi-chunk path is exercised even by these tiny programs, not just the single-chunk
    /// case) and returns the final `(registers, pc, clk, public_values_stream, memory_digest,
    /// num_chunks)`.
    fn run_minimal(program: Program) -> ([u32; NUM_REGISTERS], u32, u64, Vec<u8>, u64, usize) {
        let mut exec = MinimalExecutor::new(Arc::new(program), 4);
        let mut num_chunks = 0;
        while exec.try_execute_chunk().unwrap().is_some() {
            num_chunks += 1;
        }
        (
            exec.registers(),
            exec.pc(),
            exec.clk(),
            exec.public_values_stream().to_vec(),
            exec.memory_digest(),
            num_chunks,
        )
    }

    fn assert_matches_golden(program: impl Fn() -> Program, name: &str) {
        let golden = run_golden(program());
        let (registers, pc, clk, public_values_stream, memory_digest, num_chunks) =
            run_minimal(program());
        assert_eq!(registers, golden.final_registers, "{name}: final registers mismatch");
        assert_eq!(pc, golden.final_pc, "{name}: final pc mismatch");
        assert_eq!(clk, golden.final_clk, "{name}: final clk mismatch");
        assert_eq!(
            public_values_stream, golden.public_values_stream,
            "{name}: public values stream mismatch"
        );
        assert_eq!(memory_digest, golden.memory_digest, "{name}: memory digest mismatch");
        assert!(num_chunks >= 1, "{name}: expected at least one chunk");
    }

    #[test]
    fn matches_golden_simple_program() {
        assert_matches_golden(simple_program, "simple_program");
    }

    #[test]
    fn matches_golden_halt_only_program() {
        assert_matches_golden(halt_only_program, "halt_only_program");
    }

    #[test]
    fn matches_golden_fibonacci_real_elf() {
        assert_matches_golden(fibonacci_program, "fibonacci_program");
    }

    #[test]
    fn matches_golden_hello_world_real_elf() {
        assert_matches_golden(hello_world_program, "hello_world_program");
    }

    /// The chunk-size cutoff must never split a branch/jump from its delay slot. `fibonacci` is
    /// long/branchy enough that a small `max_trace_size` will otherwise hit this constantly; a
    /// single assertion inside the loop that fires on ANY violation is a strong test even though
    /// it's not testing one specific instance.
    #[test]
    fn chunk_boundaries_are_never_mid_delay_slot() {
        let mut exec = MinimalExecutor::new(Arc::new(fibonacci_program()), 4);
        while let Some(_chunk) = exec.try_execute_chunk().unwrap() {
            assert!(
                !exec.next_is_delayslot,
                "chunk boundary landed with next_is_delayslot == true (pc={:#x}, clk={})",
                exec.pc, exec.clk
            );
        }
    }
}
