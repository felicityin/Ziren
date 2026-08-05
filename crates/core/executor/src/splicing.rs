//! `SplicingVM` -- walks a [`MinimalTrace`] via `CoreVM`, deciding where shard (proof) boundaries
//! fall, and slices out one [`SplicedMinimalTrace`] per shard. This is the *only* place shard cuts
//! get decided; `TracingVM` just replays exactly `[clk_start, clk_end)` of whatever
//! `SplicedMinimalTrace` it's handed.
//!
//! [`ShapeChecker`] accumulates a running `trace_area`/per-chip `heights` off the existing
//! `mips_costs.json` cost table, checked with one comparison every instruction, and cuts a shard
//! when either crosses its configured threshold. This is unpadded (it does not apply the legacy
//! `inc_shard_if_need`'s per-chip `next_power_of_two` rounding), so shard boundaries may differ
//! from the legacy executor's; only total-execution correctness across all shards is required to
//! match (verified by this file's own tests).
//!
//! A `SplicingVM<'a>` borrows one `MinimalExecutor` chunk's oracle log at a time (via `CoreVM`'s
//! own lifetime-bound cursor), but a shard's own instructions are not required to fit within a
//! single chunk -- chunk boundaries (a memory-management buffer-size cutoff) and shard boundaries
//! (a proof-size decision) are deliberately independent. When `execute()` returns `TraceEnd`
//! (this chunk's log is exhausted but the in-progress shard hasn't been cut yet), the caller
//! extracts a lifetime-free [`SplicingCarry`] via `into_carry`, fetches the next chunk from
//! `MinimalExecutor`, and constructs a fresh `SplicingVM` over it via `resume` -- the shard's own
//! start-state snapshot, accumulated `ShapeChecker` totals, and oracle-log window-so-far all
//! carry over transparently.

#![allow(dead_code)]

use hashbrown::HashMap;
use std::sync::Arc;

use crate::{
    cost::PRECOMPILE_AIR_IDS,
    instruction::Instruction,
    mips_costs,
    opcode::Opcode,
    register::{Register, NUM_REGISTERS},
    syscalls::SyscallCode,
    trace::{MemReads, MemValue, MinimalTrace},
    vm::CoreVM,
    ExecutionError, MipsAirId, Program,
};

const BYTE_NUM_ROWS: u64 = 1 << 16;

/// Per-instruction shard-cut heuristic. Accumulates a running (unpadded -- see the module doc's
/// precision note) `trace_area` and per-chip `heights`, checked with a single comparison every
/// instruction.
pub(crate) struct ShapeChecker {
    costs: HashMap<MipsAirId, u64>,
    baseline: u64,
    trace_area: u64,
    heights: HashMap<MipsAirId, u64>,
    element_threshold: u64,
    height_threshold: u64,
}

impl ShapeChecker {
    #[must_use]
    pub(crate) fn new(program_size: u64, element_threshold: u64, height_threshold: u64) -> Self {
        let costs: HashMap<MipsAirId, u64> =
            mips_costs().into_iter().map(|(k, v)| (k, v as u64)).collect();
        // Fixed per-shard overhead the legacy `estimate_mips_lde_size` also charges unconditionally
        // (the byte-lookup chip's always-full table, and the program chip's one row per program
        // word) -- constant across every shard of a given program, so computed once here rather
        // than re-derived per instruction.
        let baseline = BYTE_NUM_ROWS * costs.get(&MipsAirId::Byte).copied().unwrap_or(0)
            + program_size * costs.get(&MipsAirId::Program).copied().unwrap_or(0);
        Self {
            costs,
            baseline,
            trace_area: baseline,
            heights: HashMap::new(),
            element_threshold,
            height_threshold,
        }
    }

    fn add(&mut self, air_id: MipsAirId) {
        let cost = self.costs.get(&air_id).copied().unwrap_or(0);
        self.trace_area += cost;
        *self.heights.entry(air_id).or_insert(0) += 1;
    }

    /// Classifies and accounts for one retired instruction, plus its syscall code if it was a
    /// `SYSCALL`. A pragmatic classifier (see the module doc): covers the common ADD/ADDI split
    /// (the only one that matters for realistic cost -- fully-immediate `ADD $rd, $zero, imm`,
    /// used constantly by this crate's own `load_imm`-style constant-loading, stays on `Add` per
    /// `execute_operation`'s own routing comment) and one entry per opcode/chip elsewhere; it does
    /// **not** replicate every AIR chip's `op_a == 0`-destination special case (`AluX0`/`LoadX0`)
    /// -- those exist purely to shrink *proving* cost for a rare real-code idiom and have no
    /// bearing on whether a shard-sizing estimate is safe, which is the only thing this function
    /// needs to get right (per-chip attribution doesn't need to be exact, only monotonic and
    /// safe).
    pub(crate) fn record_instruction(
        &mut self,
        instruction: &Instruction,
        syscall_code: Option<SyscallCode>,
    ) {
        self.add(opcode_air_id(instruction));
        if let Some(code) = syscall_code {
            for &air_id in precompile_air_ids(code) {
                self.add(air_id);
            }
        }
    }

    #[must_use]
    pub(crate) fn check_shard_limit(&self) -> bool {
        self.trace_area >= self.element_threshold
            || self.heights.values().any(|&h| h >= self.height_threshold)
    }

    /// Current running totals -- exposed only for tests to confirm the shard-cut check fires as
    /// soon as a threshold is actually crossed.
    #[must_use]
    pub(crate) fn current_totals(&self) -> (u64, u64) {
        (self.trace_area, self.heights.values().copied().max().unwrap_or(0))
    }

    fn start_new_shard(&mut self) {
        self.trace_area = self.baseline;
        self.heights.clear();
    }
}

/// Maps a retired instruction to the core `MipsAirId` chip whose row count it drives.
fn opcode_air_id(instruction: &Instruction) -> MipsAirId {
    match instruction.opcode {
        Opcode::ADD | Opcode::SUB => {
            // `Addi` iff register+immediate form (`imm_c && !imm_b`); fully-immediate
            // (`imm_b && imm_c`, e.g. this crate's own `load_imm` helper) and register-register
            // both stay on `Add`/`Sub`, mirroring `execute_operation`'s own
            // `local_counts.addi_events` routing comment.
            if instruction.opcode == Opcode::ADD && instruction.imm_c && !instruction.imm_b {
                MipsAirId::Addi
            } else if instruction.opcode == Opcode::ADD {
                MipsAirId::Add
            } else {
                MipsAirId::Sub
            }
        }
        Opcode::MUL => MipsAirId::Mul,
        Opcode::MULT | Opcode::MULTU | Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU => {
            MipsAirId::DivRem
        }
        Opcode::SLL => MipsAirId::ShiftLeft,
        Opcode::SRL | Opcode::SRA | Opcode::ROR => MipsAirId::ShiftRight,
        Opcode::SLT | Opcode::SLTU => {
            if instruction.imm_c {
                MipsAirId::Slti
            } else {
                MipsAirId::Lt
            }
        }
        Opcode::AND | Opcode::OR | Opcode::XOR | Opcode::NOR => MipsAirId::Bitwise,
        Opcode::CLZ | Opcode::CLO => MipsAirId::CloClz,
        Opcode::BEQ | Opcode::BGEZ | Opcode::BGTZ | Opcode::BLEZ | Opcode::BLTZ | Opcode::BNE => {
            MipsAirId::Branch
        }
        Opcode::Jump => MipsAirId::Jump,
        Opcode::Jumpi => MipsAirId::Jumpi,
        Opcode::JumpDirect => MipsAirId::JumpDirect,
        Opcode::SYSCALL => MipsAirId::SyscallInstrs,
        Opcode::LW => MipsAirId::LoadWord,
        Opcode::SW => MipsAirId::StoreWord,
        Opcode::LB | Opcode::LBU => MipsAirId::LoadByte,
        Opcode::LH | Opcode::LHU => MipsAirId::LoadHalf,
        Opcode::LWL | Opcode::LWR => MipsAirId::LoadWordUnaligned,
        Opcode::SB => MipsAirId::StoreByte,
        Opcode::SH => MipsAirId::StoreHalf,
        Opcode::SWL | Opcode::SWR => MipsAirId::StoreWordUnaligned,
        Opcode::SC | Opcode::LL => MipsAirId::StoreConditional,
        Opcode::MEQ | Opcode::MNE | Opcode::WSBH => MipsAirId::MovCond,
        Opcode::SEXT => MipsAirId::Sext,
        Opcode::INS => MipsAirId::Ins,
        Opcode::EXT => MipsAirId::Ext,
        Opcode::MADDU | Opcode::MSUBU | Opcode::MADD | Opcode::MSUB => MipsAirId::Maddsub,
        Opcode::TEQ => MipsAirId::Teq,
        Opcode::UNIMPL => MipsAirId::SyscallInstrs, // never actually retired; harmless fallback
    }
}

/// Precompile-chip costs a given syscall additionally drives, beyond the base `SyscallInstrs`
/// cost every `SYSCALL` instruction already gets via [`opcode_air_id`]. Only fires for syscalls
/// this crate actually implements (see `minimal/syscall.rs`'s module doc on scope); extend this
/// table as each additional precompile is wired up for real.
fn precompile_air_ids(code: SyscallCode) -> &'static [MipsAirId] {
    if matches!(
        code,
        SyscallCode::SYS_BRK
            | SyscallCode::SYS_MMAP
            | SyscallCode::SYS_MMAP2
            | SyscallCode::SYS_CLONE
            | SyscallCode::SYS_READ
            | SyscallCode::SYS_WRITE
            | SyscallCode::SYS_FCNTL
    ) {
        return &[MipsAirId::SysLinux];
    }
    PRECOMPILE_AIR_IDS
        .iter()
        .find(|(sc, _)| *sc == code)
        .map_or(&[] as &[MipsAirId], |(_, ids)| ids)
}

/// A single shard's oracle-log window and starting-state snapshot (the latter reconstructed from
/// `CoreVM`'s live state at the cut point, not re-derived), plus this shard's own `clk_end` (the
/// cut point, *not* whichever chunk(s) it was spliced from).
///
/// Self-contained -- no generic/lifetime parameter, unlike an earlier version of this type that
/// borrowed directly from one backing chunk -- because a shard's instructions can span more than
/// one `MinimalExecutor` chunk (see `SplicingVM::into_carry`/`resume`), so its oracle-log window
/// may need to be a concatenation of more than one chunk's data. This also makes it trivially
/// sendable across a channel to a separate trace-generation worker.
#[derive(Debug, Clone)]
pub struct SplicedMinimalTrace {
    mem_reads: Arc<[MemValue]>,
    start_registers: [u32; NUM_REGISTERS],
    start_register_timestamps: [u64; NUM_REGISTERS],
    start_pc: u32,
    start_clk: u64,
    end_clk: u64,
    start_input_stream_ptr: usize,
}

impl MinimalTrace for SplicedMinimalTrace {
    fn start_registers(&self) -> [u32; NUM_REGISTERS] {
        self.start_registers
    }

    fn start_register_timestamps(&self) -> [u64; NUM_REGISTERS] {
        self.start_register_timestamps
    }

    fn pc_start(&self) -> u32 {
        self.start_pc
    }

    fn clk_start(&self) -> u64 {
        self.start_clk
    }

    fn start_input_stream_ptr(&self) -> usize {
        self.start_input_stream_ptr
    }

    fn clk_end(&self) -> u64 {
        self.end_clk
    }

    fn num_mem_reads(&self) -> u64 {
        self.mem_reads.len() as u64
    }

    fn mem_reads(&self) -> MemReads<'_> {
        MemReads::new(&self.mem_reads)
    }

    fn mem_reads_slice(&self) -> &[MemValue] {
        &self.mem_reads
    }
}

/// Shard-tracking state carried across a `MinimalExecutor` chunk boundary -- see
/// `SplicingVM::into_carry`/`resume`. Lifetime-free (unlike `SplicingVM` itself), so it survives
/// past the point where the exhausted chunk that produced it goes out of scope (e.g. a `while let
/// Some(chunk) = minimal.try_execute_chunk()?` loop, where each `chunk` is a fresh local).
pub(crate) struct SplicingCarry {
    shard_start_registers: [u32; NUM_REGISTERS],
    shard_start_register_timestamps: [u64; NUM_REGISTERS],
    shard_start_pc: u32,
    shard_start_clk: u64,
    shard_start_input_stream_ptr: usize,
    /// This in-progress shard's oracle-log entries from every chunk it has already spanned
    /// (i.e. every chunk before the one currently being replayed).
    shard_mem_reads_carried: Vec<MemValue>,
    shape_checker: ShapeChecker,
}

/// The result of one `SplicingVM::execute()` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SplicingStatus {
    /// The program halted for real; no more shards follow.
    Done,
    /// This `MinimalTrace`'s oracle log is exhausted (`clk == clk_end`), but the program hasn't
    /// halted -- a later chunk (from a subsequent `MinimalExecutor::try_execute_chunk` call)
    /// continues it. No `SplicedMinimalTrace` is pending in this case (unlike `ShardBoundary`):
    /// whatever ran since the last `splice()` is incomplete and must be re-walked once the next
    /// chunk's data is available.
    TraceEnd,
    /// A shard boundary was reached; call `splice()` before calling `execute()` again.
    ShardBoundary,
}

pub(crate) struct SplicingVM<'a> {
    core: CoreVM<'a>,
    shape_checker: ShapeChecker,
    shard_start_registers: [u32; NUM_REGISTERS],
    shard_start_register_timestamps: [u64; NUM_REGISTERS],
    shard_start_pc: u32,
    shard_start_clk: u64,
    shard_start_input_stream_ptr: usize,
    shard_start_mem_reads_consumed: usize,
    /// See `SplicingCarry`'s identically-named field -- empty except right after `Self::resume`.
    shard_mem_reads_carried: Vec<MemValue>,
}

impl<'a> SplicingVM<'a> {
    #[must_use]
    pub(crate) fn new<T: MinimalTrace>(
        trace: &'a T,
        program: Arc<Program>,
        max_syscall_cycles: u32,
        stdin: Arc<[Vec<u8>]>,
        element_threshold: u64,
        height_threshold: u64,
    ) -> Self {
        let program_size = program.instructions.len() as u64;
        let core = CoreVM::new(trace, program, max_syscall_cycles, stdin);
        Self {
            shard_start_registers: core.registers(),
            shard_start_register_timestamps: core.register_timestamps(),
            shard_start_pc: core.pc(),
            shard_start_clk: core.clk(),
            shard_start_input_stream_ptr: core.input_stream_ptr(),
            shard_start_mem_reads_consumed: 0,
            shard_mem_reads_carried: Vec::new(),
            core,
            shape_checker: ShapeChecker::new(program_size, element_threshold, height_threshold),
        }
    }

    /// Resumes replaying an in-progress shard -- one whose state was extracted via
    /// `Self::into_carry` when the *previous* chunk's oracle log ran out (`execute()` returned
    /// `TraceEnd`) before the shard itself ended -- against a newly available chunk. The shard's
    /// own start-state snapshot and accumulated `ShapeChecker` cost totals carry over unchanged;
    /// only the underlying `CoreVM` is rebuilt, over `trace` (whose own start state is wherever
    /// `MinimalExecutor` left off, i.e. exactly where the previous chunk's replay ended, so
    /// resuming execution here is seamless -- only the shard-window bookkeeping needs threading
    /// through by hand).
    #[must_use]
    pub(crate) fn resume<T: MinimalTrace>(
        carry: SplicingCarry,
        trace: &'a T,
        program: Arc<Program>,
        max_syscall_cycles: u32,
        stdin: Arc<[Vec<u8>]>,
    ) -> Self {
        Self {
            core: CoreVM::new(trace, program, max_syscall_cycles, stdin),
            shape_checker: carry.shape_checker,
            shard_start_registers: carry.shard_start_registers,
            shard_start_register_timestamps: carry.shard_start_register_timestamps,
            shard_start_pc: carry.shard_start_pc,
            shard_start_clk: carry.shard_start_clk,
            shard_start_input_stream_ptr: carry.shard_start_input_stream_ptr,
            shard_start_mem_reads_consumed: 0,
            shard_mem_reads_carried: carry.shard_mem_reads_carried,
        }
    }

    /// Extracts this in-progress shard's carryable state after `execute()` returns `TraceEnd`.
    /// Consumes `self` since the underlying `CoreVM`/chunk borrow is now spent; pass the result to
    /// `Self::resume` once the next chunk is available. `trace` must be the same trace this
    /// `SplicingVM` was constructed (or last resumed) with.
    #[must_use]
    pub(crate) fn into_carry<T: MinimalTrace>(self, trace: &T) -> SplicingCarry {
        let mem_reads_consumed = trace.num_mem_reads() as usize - self.core.mem_reads_remaining();
        let mut shard_mem_reads_carried = self.shard_mem_reads_carried;
        shard_mem_reads_carried
            .extend_from_slice(&trace.mem_reads_slice()[self.shard_start_mem_reads_consumed..mem_reads_consumed]);
        SplicingCarry {
            shard_start_registers: self.shard_start_registers,
            shard_start_register_timestamps: self.shard_start_register_timestamps,
            shard_start_pc: self.shard_start_pc,
            shard_start_clk: self.shard_start_clk,
            shard_start_input_stream_ptr: self.shard_start_input_stream_ptr,
            shard_mem_reads_carried,
            shape_checker: self.shape_checker,
        }
    }

    #[must_use]
    pub(crate) fn registers(&self) -> [u32; NUM_REGISTERS] {
        self.core.registers()
    }

    #[must_use]
    pub(crate) fn pc(&self) -> u32 {
        self.core.pc()
    }

    #[must_use]
    pub(crate) fn clk(&self) -> u64 {
        self.core.clk()
    }

    /// See `ShapeChecker::current_totals` -- test-only sanity-check surface.
    #[must_use]
    pub(crate) fn current_shape_totals(&self) -> (u64, u64) {
        self.shape_checker.current_totals()
    }

    /// Runs until the program halts, this chunk's oracle log is exhausted, or a shard boundary is
    /// reached.
    ///
    /// # Errors
    ///
    /// Propagates any [`ExecutionError`] from executing an instruction.
    pub(crate) fn execute(&mut self) -> Result<SplicingStatus, ExecutionError> {
        // Do-while, matching `CoreVM::execute`/`MinimalExecutor::try_execute_chunk`'s identical
        // fix: none of `is_halted()`/`clk >= clk_end`/the shard-cut check may be consulted before
        // at least one instruction has actually retired.
        loop {
            let instruction = self.core.program().fetch(self.core.pc());
            let syscall_code = instruction
                .is_syscall_instruction()
                .then(|| SyscallCode::from_u32(self.core.reg_peek(Register::V0)));

            self.core.execute_instruction()?;
            self.shape_checker.record_instruction(&instruction, syscall_code);

            if self.core.is_halted() {
                return Ok(SplicingStatus::Done);
            }
            if self.core.clk() >= self.core.clk_end() {
                return Ok(SplicingStatus::TraceEnd);
            }
            if !self.core.next_is_delayslot() && self.shape_checker.check_shard_limit() {
                return Ok(SplicingStatus::ShardBoundary);
            }
        }
    }

    /// Slices out the shard that just ended (the span from the previous `splice()`/construction/
    /// `resume()` up to the current, just-reached boundary), and resets internal bookkeeping for
    /// the next one. Call only immediately after `execute()` returns `ShardBoundary` or `Done`.
    #[must_use]
    pub(crate) fn splice<T: MinimalTrace>(&mut self, trace: &T) -> SplicedMinimalTrace {
        // The delay-slot invariant: a cut is only ever legal when `!next_is_delayslot`, which
        // implies `next_pc == pc + 4` exactly -- i.e. this shard's starting snapshot never needs
        // to carry a separate `next_pc`/pending-branch field. Checked here, not just in a test,
        // because a violation would silently produce a shard `TracingVM` replays incorrectly
        // with no other signal.
        debug_assert!(
            !self.core.next_is_delayslot(),
            "splice() called while next_is_delayslot == true -- caller must only call this \
             immediately after execute() returns ShardBoundary"
        );
        let mem_reads_consumed = trace.num_mem_reads() as usize - self.core.mem_reads_remaining();
        let this_chunk_slice =
            &trace.mem_reads_slice()[self.shard_start_mem_reads_consumed..mem_reads_consumed];
        // Common case (a shard fits entirely within one chunk): just copy this one window. Rare
        // case (a shard spanned one or more earlier chunks via `resume()`): concatenate onto what
        // was already carried over. Either way this is a real copy, not a zero-copy borrow -- an
        // earlier version of this type borrowed directly from a single backing chunk instead, but
        // that can't represent the multi-chunk case at all (`mem_reads_slice()` must return one
        // contiguous `&[MemValue]`). Acceptable for now; revisit if profiling shows it matters.
        let mem_reads: Arc<[MemValue]> = if self.shard_mem_reads_carried.is_empty() {
            this_chunk_slice.into()
        } else {
            let mut combined = std::mem::take(&mut self.shard_mem_reads_carried);
            combined.extend_from_slice(this_chunk_slice);
            combined.into()
        };
        let spliced = SplicedMinimalTrace {
            mem_reads,
            start_registers: self.shard_start_registers,
            start_register_timestamps: self.shard_start_register_timestamps,
            start_pc: self.shard_start_pc,
            start_clk: self.shard_start_clk,
            end_clk: self.core.clk(),
            start_input_stream_ptr: self.shard_start_input_stream_ptr,
        };

        self.shard_start_registers = self.core.registers();
        self.shard_start_register_timestamps = self.core.register_timestamps();
        self.shard_start_pc = self.core.pc();
        self.shard_start_clk = self.core.clk();
        self.shard_start_input_stream_ptr = self.core.input_stream_ptr();
        self.shard_start_mem_reads_consumed = mem_reads_consumed;
        self.shape_checker.start_new_shard();

        spliced
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        minimal::MinimalExecutor,
        programs::tests::{fibonacci_program, halt_only_program, hello_world_program, simple_program},
        Program,
    };

    /// Runs `program` through `MinimalExecutor` (whole program in one chunk) then
    /// `SplicingVM`/`ShapeChecker` with the given thresholds, driving `execute()`/`splice()` to
    /// completion. Returns `(num_shards, final_registers, final_pc, final_clk)`.
    fn run_splicing(
        program: impl Fn() -> Program,
        element_threshold: u64,
        height_threshold: u64,
    ) -> (usize, [u32; NUM_REGISTERS], u32, u64) {
        let mut minimal = MinimalExecutor::new(Arc::new(program()), u64::MAX / 2);
        let chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");
        let max_syscall_cycles = minimal.max_syscall_cycles();

        let mut splicing = SplicingVM::new(
            &chunk,
            Arc::new(program()),
            max_syscall_cycles,
            Arc::from([]),
            element_threshold,
            height_threshold,
        );
        let mut num_shards = 0;
        loop {
            match splicing.execute().unwrap() {
                SplicingStatus::ShardBoundary => {
                    // The shard-cut check fires on the first instruction whose accumulated cost
                    // crosses a threshold (checked every instruction) -- confirmed directly
                    // rather than via a numeric margin, since the fixed per-shard baseline
                    // (`ShapeChecker::new`'s `BYTE_NUM_ROWS * cost[Byte]` term) can legitimately
                    // dwarf a small test threshold on its own, making a "close to threshold"
                    // margin meaningless.
                    assert!(
                        splicing.current_shape_totals().0 >= element_threshold
                            || splicing.current_shape_totals().1 >= height_threshold,
                        "ShardBoundary returned but neither threshold was actually crossed"
                    );
                    let _spliced = splicing.splice(&chunk); // delay-slot invariant asserted inside splice() itself
                    num_shards += 1;
                }
                SplicingStatus::Done => {
                    // The final segment (ending in real completion, not a shape-cut) is still a
                    // real shard and must still be spliced -- `TracingVM` needs it too.
                    let _spliced = splicing.splice(&chunk);
                    num_shards += 1;
                    break;
                }
                SplicingStatus::TraceEnd => {
                    panic!("expected the whole run to fit in a single MinimalExecutor chunk")
                }
            }
        }
        (num_shards, splicing.registers(), splicing.pc(), splicing.clk())
    }

    /// With thresholds far larger than any of these tiny/small programs could ever reach,
    /// splicing must produce exactly one shard and reproduce the exact same final state
    /// `MinimalExecutor` itself already reached (avoiding a second dependency on `golden.rs`'s
    /// legacy-`Executor` comparison for this specific check).
    fn assert_single_shard_matches_minimal(program: impl Fn() -> Program, name: &str) {
        let mut minimal = MinimalExecutor::new(Arc::new(program()), u64::MAX / 2);
        let _ = minimal.try_execute_chunk().unwrap();
        let (num_shards, registers, pc, clk) = run_splicing(program, u64::MAX / 2, u64::MAX / 2);
        assert_eq!(num_shards, 1, "{name}: expected exactly one shard with generous thresholds");
        assert_eq!(registers, minimal.registers(), "{name}: register mismatch");
        assert_eq!(pc, minimal.pc(), "{name}: pc mismatch");
        assert_eq!(clk, minimal.clk(), "{name}: clk mismatch");
    }

    #[test]
    fn splicing_single_shard_matches_minimal_simple_program() {
        assert_single_shard_matches_minimal(simple_program, "simple_program");
    }

    #[test]
    fn splicing_single_shard_matches_minimal_halt_only_program() {
        assert_single_shard_matches_minimal(halt_only_program, "halt_only_program");
    }

    #[test]
    fn splicing_single_shard_matches_minimal_fibonacci_real_elf() {
        assert_single_shard_matches_minimal(fibonacci_program, "fibonacci_program");
    }

    #[test]
    fn splicing_single_shard_matches_minimal_hello_world_real_elf() {
        assert_single_shard_matches_minimal(hello_world_program, "hello_world_program");
    }

    /// With tight thresholds, `fibonacci_program` (long/branchy enough to matter) must actually
    /// split into multiple shards, and the concatenated total execution must still reach the
    /// exact same final state as `MinimalExecutor`'s own single-chunk run -- shard *boundaries*
    /// are explicitly allowed to differ from anything else (there's no legacy reference for them
    /// any more, see the module doc), only total-execution correctness is required.
    #[test]
    fn splicing_many_shards_still_matches_minimal_fibonacci_real_elf() {
        let mut minimal = MinimalExecutor::new(Arc::new(fibonacci_program()), u64::MAX / 2);
        let _ = minimal.try_execute_chunk().unwrap();

        // `element_threshold` left effectively disabled (`u64::MAX/2`, matching the single-shard
        // tests): `ShapeChecker::new`'s fixed per-shard baseline (the byte chip's mandatory full
        // `BYTE_NUM_ROWS`-row table) alone can be far larger than a small `trace_area` threshold,
        // so a tiny `element_threshold` doesn't exercise "accumulates over many instructions" at
        // all -- it just fires on shard 1's first instruction, repeatedly. `height_threshold`
        // starts at 0 every shard with no such baseline, so a small value here is what actually
        // forces the intended "many small shards" scenario.
        let (num_shards, registers, pc, clk) = run_splicing(fibonacci_program, u64::MAX / 2, 40);
        assert!(num_shards > 1, "expected a tight height_threshold to force multiple shards, got 1");
        assert_eq!(registers, minimal.registers(), "register mismatch");
        assert_eq!(pc, minimal.pc(), "pc mismatch");
        assert_eq!(clk, minimal.clk(), "clk mismatch");
    }

    /// Same as above but also cross-checked against `golden.rs`'s independent legacy-`Executor`
    /// reference.
    #[test]
    fn splicing_many_shards_matches_golden_fibonacci_real_elf() {
        let golden = crate::golden::run_golden(fibonacci_program());
        let (num_shards, registers, pc, clk) = run_splicing(fibonacci_program, u64::MAX / 2, 40);
        assert!(num_shards > 1, "expected tight thresholds to force multiple shards, got 1");
        assert_eq!(registers, golden.final_registers, "register mismatch vs. golden");
        assert_eq!(pc, golden.final_pc, "pc mismatch vs. golden");
        assert_eq!(clk, golden.final_clk, "clk mismatch vs. golden");
    }

    /// Exercises `SplicingVM::into_carry`/`resume`: a tiny `max_trace_size` forces
    /// `MinimalExecutor` to cut many small chunks, while generous shard thresholds keep the whole
    /// run to a single shard -- meaning that one shard's oracle-log window must be correctly
    /// concatenated across every chunk transition. Also feeds the resulting `SplicedMinimalTrace`
    /// through a real `TracingVM` replay (not just checking `SplicingVM`'s own final state), since
    /// the risk here is specifically a wrong/misordered concatenation that `SplicingVM`'s own
    /// bookkeeping wouldn't catch but a downstream replay reading garbage values would.
    #[test]
    fn splicing_resumes_across_many_chunks_and_tracing_vm_replays_it_matching_golden() {
        let golden = crate::golden::run_golden(fibonacci_program());
        let program = Arc::new(fibonacci_program());

        let mut minimal = MinimalExecutor::new(Arc::new(fibonacci_program()), 4);
        let max_syscall_cycles = minimal.max_syscall_cycles();
        let mut chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");
        let mut splicing = SplicingVM::new(
            &chunk,
            program.clone(),
            max_syscall_cycles,
            Arc::from([]),
            u64::MAX / 2,
            u64::MAX / 2,
        );
        let mut num_shards = 0;
        let mut num_chunk_resumes = 0;
        let spliced = loop {
            match splicing.execute().unwrap() {
                SplicingStatus::ShardBoundary => {
                    let _spliced = splicing.splice(&chunk);
                    num_shards += 1;
                }
                SplicingStatus::Done => {
                    let spliced = splicing.splice(&chunk);
                    num_shards += 1;
                    break spliced;
                }
                SplicingStatus::TraceEnd => {
                    let carry = splicing.into_carry(&chunk);
                    chunk = minimal.try_execute_chunk().unwrap().expect("expected another chunk");
                    splicing = SplicingVM::resume(
                        carry,
                        &chunk,
                        program.clone(),
                        max_syscall_cycles,
                        Arc::from([]),
                    );
                    num_chunk_resumes += 1;
                }
            }
        };
        assert!(
            num_chunk_resumes > 1,
            "expected a tiny max_trace_size to force many chunk resumes, got {num_chunk_resumes}"
        );
        assert_eq!(num_shards, 1, "generous thresholds should keep this to a single shard spanning every chunk");

        let mut record = crate::record::ExecutionRecord::new(program.clone());
        let mut tracing = crate::tracing::TracingVM::new(
            &spliced,
            program,
            max_syscall_cycles,
            Arc::from([]),
            &mut record,
        );
        assert_eq!(
            tracing.execute().unwrap(),
            crate::vm::CoreVMStatus::Done,
            "TracingVM must be able to replay the concatenated shard to completion"
        );
        assert_eq!(tracing.registers(), golden.final_registers, "register mismatch vs. golden");
        assert_eq!(tracing.pc(), golden.final_pc, "pc mismatch vs. golden");
        assert_eq!(tracing.clk(), golden.final_clk, "clk mismatch vs. golden");
    }
}
