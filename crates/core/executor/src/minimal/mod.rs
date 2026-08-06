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
//! implemented for real. Most syscalls are too: `HALT`, `WRITE`'s `FD_PUBLIC_VALUES` case, every
//! precompile, the Linux syscall shims, `ENTER_UNCONSTRAINED`/`EXIT_UNCONSTRAINED` (see
//! `Self::enter_unconstrained`'s doc comment), `HINT_LEN`/`HINT_READ` (reading from `ZKMStdin`'s
//! buffer, see `Self::hint_seed`'s doc comment), and `SYSVERIFY` (`verify_zkm_proof`, see
//! `syscall::verify_dispatch`'s doc comment) are all real. A hint computed *inside* an
//! `unconstrained!{}` block via `hint()`/`hint_slice()` (or a hook fd) is not yet supported --
//! `WRITE`'s `FD_HINT` case remains a documented no-op (see `syscall.rs`), and hooks
//! (`ZKMContext::hook_registry`) aren't consulted anywhere yet either. Extend the dispatch in
//! [`syscall`] as needed.

#![allow(dead_code)]

mod syscall;

use std::sync::Arc;

use enum_map::EnumMap;
use hashbrown::HashMap;

use crate::{
    events::{MemoryAccessPosition, MemoryInitializeFinalizeEvent, MemoryRecord},
    memory::{MaybeCowMemory, PagedMemory},
    opcode::Opcode,
    register::{Register, NUM_REGISTERS},
    subproof::SubproofVerifier,
    syscalls::{default_syscall_map, SyscallCode},
    trace::{MemValue, TraceChunk},
    vm, ExecutionError, ExecutionReport, Instruction, Program, ZKMReduceProof,
};
use zkm_hypercube::{
    config::ZkmGlobalContext, verifier::ZkmPcsProofInner, MachineVerifyingKey,
};

/// Register/pc state snapshotted on `ENTER_UNCONSTRAINED`, restored on `EXIT_UNCONSTRAINED` --
/// see `MinimalExecutor::enter_unconstrained`'s doc comment. `clk` is deliberately *not* here:
/// unlike the legacy `Executor` (which lets `clk` advance through the whole block and rolls it
/// back afterward), `clk` here is simply gated from advancing at all while unconstrained (see
/// `execute_instruction`), so there is nothing to restore.
struct UnconstrainedSnapshot {
    registers: [u32; NUM_REGISTERS],
    register_timestamps: [u64; NUM_REGISTERS],
    pc: u32,
}

/// Bound on how many oracle-log entries are pre-reserved for a chunk. `max_trace_size` is a
/// cutoff, not a reservation hint -- callers may pass a very large or sentinel value to mean "no
/// cutoff", so the reservation itself is capped independently of it.
const MAX_PREALLOCATED_ORACLE_LOG: u64 = 1 << 20;

/// A live, full-speed interpreter. See the module doc for scope.
pub(crate) struct MinimalExecutor<'a> {
    program: Arc<Program>,
    registers: [u32; NUM_REGISTERS],
    /// The `clk` each register was last written at -- `0` means "never touched" (matches
    /// `MemoryRecord::timestamp`'s sentinel). Tracked purely so a later chunk's `TracingVM` can
    /// recover a real `prev_timestamp` for a register last written in an *earlier* chunk;
    /// register *values* never need this (always recomputed live, never oracle-logged).
    register_timestamps: [u64; NUM_REGISTERS],
    page_table: MaybeCowMemory<MemoryRecord>,
    pc: u32,
    next_pc: u32,
    clk: u64,
    next_is_delayslot: bool,
    exited: bool,
    /// Whether we're currently inside an `unconstrained!{}` block -- see
    /// `Self::enter_unconstrained`'s doc comment.
    unconstrained: bool,
    /// Set by `Self::enter_unconstrained`, consumed by `Self::exit_unconstrained`. `None` outside
    /// an unconstrained block.
    unconstrained_snapshot: Option<UnconstrainedSnapshot>,
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
    /// `ZKMStdin`'s buffer -- whole-run constant, like `program`, never mutated after
    /// construction. `HINT_LEN` reads `stdin[input_stream_ptr].len()`; `HINT_READ` pops the front
    /// entry (`input_stream_ptr += 1`) and seeds `hint_seed`. Mirrors `Executor::state.input_stream`.
    stdin: Arc<[Vec<u8>]>,
    /// Live cursor into `stdin`, advanced by `HINT_READ`. Threaded across chunk/shard boundaries
    /// as a carried snapshot (`TraceChunk::start_input_stream_ptr`), exactly like `pc`/`clk`.
    input_stream_ptr: usize,
    /// Pre-seeded values for addresses `HINT_READ` has written but no real load/store has touched
    /// yet -- mirrors `Executor::state.uninitialized_memory`. `HINT_READ` never calls `mw`
    /// directly (that would mark the address "touched" at the hint's own `clk`, pricing an
    /// address that's hinted but never actually accessed, which legacy never does); instead it
    /// only records here, and `mr`/`mw`/`mr_log_only`'s own `Entry::Vacant` case consults this
    /// (via `remove`, so a value is only ever consumed once) instead of defaulting to `0` -- the
    /// hinted value then flows through the *existing* oracle log automatically, the moment any
    /// real instruction first touches that address, exactly like every other RAM preimage.
    /// Deliberately *not* consulted by `word_peek` (an untracked, non-mutating peek used only for
    /// a store's byte/half merge) -- matches `Executor::word`'s identical behavior: it doesn't
    /// consult `uninitialized_memory` either, so a partial store to a hint-seeded-but-untouched
    /// address merges against `0`, not the hinted bytes. That's legacy's real, established
    /// behavior (not a bug this port should "fix"), and must be matched bit-for-bit.
    hint_seed: HashMap<u32, u32>,
    /// Per-opcode/syscall dispatch counts, for `Self::execution_report`. Mirrors
    /// `Executor::report`'s identical fields and gating (`!self.unconstrained`) -- purely
    /// informational, consumed only by `prove.rs`'s end-of-run summary logging.
    opcode_counts: Box<EnumMap<Opcode, u64>>,
    syscall_counts: Box<EnumMap<SyscallCode, u64>>,
    /// `ZKMStdin`'s proof stream -- whole-run constant, mirrors `Executor::state.proof_stream`.
    /// `SYSVERIFY` (`verify_zkm_proof` in the guest) pops the front entry on each call.
    proof_stream: Vec<(ZKMReduceProof<ZkmGlobalContext, ZkmPcsProofInner>, MachineVerifyingKey<ZkmGlobalContext>)>,
    /// Live cursor into `proof_stream`, advanced by `SYSVERIFY`. Mirrors `input_stream_ptr`, but
    /// is never threaded across a chunk/shard boundary: unlike `HINT_LEN`/`HINT_READ`, `SYSVERIFY`
    /// is only ever dispatched by `MinimalExecutor` itself (see `verify_dispatch`'s doc comment on
    /// why `CoreVM`/`TracingVM` never need to replay it), so there is no later stage that needs to
    /// recover this cursor.
    proof_stream_ptr: usize,
    /// Host-side sanity check invoked by `SYSVERIFY`, verifying that a `verify_zkm_proof` call in
    /// the guest was passed a real, valid proof -- mirrors `Executor::subproof_verifier`. `None`
    /// skips the check entirely (matches `Executor`'s own `rt.subproof_verifier` being `None`).
    subproof_verifier: Option<&'a dyn SubproofVerifier>,
    /// Mirrors `Executor::deferred_proof_verification`: `SYSVERIFY` is a no-op (beyond advancing
    /// `proof_stream_ptr`) when this is `false`.
    deferred_proof_verification_enabled: bool,
}

impl<'a> MinimalExecutor<'a> {
    /// Convenience constructor for callers that never feed stdin -- equivalent to
    /// `Self::new_with_stdin(program, max_trace_size, Arc::from([]))`.
    #[must_use]
    pub(crate) fn new(program: Arc<Program>, max_trace_size: u64) -> Self {
        Self::new_with_stdin(program, max_trace_size, Arc::from([]))
    }

    /// Convenience constructor for callers that feed stdin but never subproofs -- equivalent to
    /// `Self::new_with_context(program, max_trace_size, stdin, Vec::new(), None, true)`.
    #[must_use]
    pub(crate) fn new_with_stdin(
        program: Arc<Program>,
        max_trace_size: u64,
        stdin: Arc<[Vec<u8>]>,
    ) -> Self {
        Self::new_with_context(program, max_trace_size, stdin, Vec::new(), None, true)
    }

    /// The full constructor, taking every piece a real `ZKMContext`/`ZKMStdin` can carry that
    /// `MinimalExecutor` needs. `deferred_proof_verification_enabled` mirrors
    /// `!context.skip_deferred_proof_verification`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_context(
        program: Arc<Program>,
        max_trace_size: u64,
        stdin: Arc<[Vec<u8>]>,
        proof_stream: Vec<(ZKMReduceProof<ZkmGlobalContext, ZkmPcsProofInner>, MachineVerifyingKey<ZkmGlobalContext>)>,
        subproof_verifier: Option<&'a dyn SubproofVerifier>,
        deferred_proof_verification_enabled: bool,
    ) -> Self {
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
            register_timestamps: [0; NUM_REGISTERS],
            page_table: MaybeCowMemory::Owned(page_table),
            clk: 1, // `clk == 0` is the "never touched" sentinel, matches `ExecutionState::new`.
            next_is_delayslot: false,
            unconstrained: false,
            unconstrained_snapshot: None,
            exited: false,
            finished: false,
            max_syscall_cycles,
            oracle_log: Vec::with_capacity(
                max_trace_size.min(MAX_PREALLOCATED_ORACLE_LOG) as usize
            ),
            max_trace_size,
            public_values_stream: Vec::new(),
            stdin,
            input_stream_ptr: 0,
            hint_seed: HashMap::new(),
            opcode_counts: Box::default(),
            syscall_counts: Box::default(),
            proof_stream,
            proof_stream_ptr: 0,
            subproof_verifier,
            deferred_proof_verification_enabled,
        }
    }

    #[must_use]
    pub(crate) fn registers(&self) -> [u32; NUM_REGISTERS] {
        self.registers
    }

    #[must_use]
    pub(crate) fn register_timestamps(&self) -> [u64; NUM_REGISTERS] {
        self.register_timestamps
    }

    #[must_use]
    pub(crate) fn pc(&self) -> u32 {
        self.pc
    }

    #[must_use]
    pub(crate) fn clk(&self) -> u64 {
        self.clk
    }

    /// Current cursor into `stdin` -- snapshotted into `TraceChunk::start_input_stream_ptr` at
    /// each chunk boundary, exactly like `pc()`/`clk()`.
    #[must_use]
    pub(crate) fn input_stream_ptr(&self) -> usize {
        self.input_stream_ptr
    }

    /// `stdin` itself -- whole-run constant, threaded into `CoreVM`/`TracingVM` alongside
    /// `program` so replay can independently recompute `HINT_LEN`'s return value.
    #[must_use]
    pub(crate) fn stdin(&self) -> Arc<[Vec<u8>]> {
        self.stdin.clone()
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
            .keys()
            .filter_map(|addr| {
                let record = self.page_table.get(addr).unwrap();
                (record.timestamp != 0).then_some((addr, record.value, record.timestamp))
            })
            .collect();
        touched.sort_unstable_by_key(|&(addr, _, _)| addr);
        let mut hasher = DefaultHasher::new();
        touched.hash(&mut hasher);
        hasher.finish()
    }

    /// Computes `global_memory_initialize_events`/`global_memory_finalize_events` for the whole
    /// run, mirroring `Executor::postprocess()`'s memory-event section exactly. Only meaningful
    /// once the whole run has finished (`Self::is_done()`), since this walks the *final*
    /// register/RAM state -- addresses touched by only *some* chunks wouldn't be visible from an
    /// in-progress run.
    ///
    /// Every initialize event currently uses value `0` unconditionally. `Executor::postprocess()`
    /// instead consults `uninitialized_memory.get(addr)` here, so a hint-seeded-and-then-really-
    /// touched address (outside `program.image`) gets its real hinted value, not `0` -- a gap this
    /// port has not yet closed (`Self::hint_seed`'s own entries are consumed via `remove` on first
    /// touch, so by the time this runs there is nothing left to consult for such an address; a fix
    /// needs a separate, non-consumed snapshot of hint-seeded initial values).
    #[must_use]
    pub(crate) fn global_memory_events(
        &self,
    ) -> (Vec<MemoryInitializeFinalizeEvent>, Vec<MemoryInitializeFinalizeEvent>) {
        let mut initialize_events = Vec::new();
        let mut finalize_events = Vec::new();

        // addr 0 (register `ZERO`) is handled unconditionally and first, regardless of whether
        // it was ever touched -- the finalize table's AIR constrains its first row to address 0.
        let zero_ts = self.register_timestamps[Register::ZERO as usize];
        let addr_0_final = MemoryRecord { timestamp: if zero_ts != 0 { zero_ts } else { 1 }, value: 0 };
        finalize_events.push(MemoryInitializeFinalizeEvent::finalize_from_record(0, &addr_0_final));
        initialize_events.push(MemoryInitializeFinalizeEvent::initialize(0, 0));

        for addr in 1..NUM_REGISTERS as u32 {
            // A register seeded from `program.image` (e.g. the ELF loader's computed initial
            // `BRK`) is "touched" from program load onward, even if never accessed at runtime --
            // mirrors `Memory::insert`'s dispatch, which `Executor::initialize` relies on to
            // create a real `self.state.memory.registers` entry for these addresses up front
            // (see `Self::new`'s identical comment on why register-index seeds bypass the page
            // table). Without this, a register like `BRK` that a real program's runtime resolves
            // once but the guest code never re-reads/re-writes would be silently dropped here.
            let seeded = self.program.image.contains_key(&addr);
            let timestamp = self.register_timestamps[addr as usize];
            if timestamp == 0 && !seeded {
                continue;
            }
            // Program memory is initialized in the `MemoryProgramChip` and doesn't require any
            // events, so we only send init events for other addresses.
            if !seeded {
                initialize_events.push(MemoryInitializeFinalizeEvent::initialize(addr, 0));
            }
            let record = MemoryRecord { timestamp, value: self.registers[addr as usize] };
            finalize_events.push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &record));
        }

        for addr in self.page_table.keys() {
            if addr == 0 {
                // Handled above.
                continue;
            }
            if !self.program.image.contains_key(&addr) {
                initialize_events.push(MemoryInitializeFinalizeEvent::initialize(addr, 0));
            }
            let record = self.page_table.get(addr).unwrap();
            finalize_events.push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, record));
        }

        (initialize_events, finalize_events)
    }

    /// Snapshots the opcode/syscall dispatch counts accumulated so far into an [`ExecutionReport`],
    /// for `prove.rs`'s end-of-run summary logging. `touched_memory_addresses` reuses
    /// `Self::global_memory_events`'s finalize-event count, since that walks the exact same
    /// touched-address set this field is meant to describe.
    #[must_use]
    pub(crate) fn execution_report(&self) -> ExecutionReport {
        ExecutionReport {
            opcode_counts: self.opcode_counts.clone(),
            syscall_counts: self.syscall_counts.clone(),
            cycle_tracker: Default::default(),
            touched_memory_addresses: self.global_memory_events().1.len() as u64,
        }
    }

    // ---- register file (values never oracle-logged; timestamps tracked -- see the struct doc
    // comment on `register_timestamps`) ----

    fn reg(&self, r: Register) -> u32 {
        self.registers[r as usize]
    }

    /// Reads register `r`'s value, re-stamping its own consistency-timestamp at `clk + position`
    /// in the process -- mirrors `Executor::rr_traced`'s unconditional `record.timestamp =
    /// timestamp` (a read participates in the same access-timestamp chain as a write, since the
    /// timestamp tracked here is later used to seed the *next* chunk's `start_register_timestamps`
    /// via `Self::register_timestamps()`; if reads didn't re-stamp it, a register whose last touch
    /// in this chunk was a read -- not a write -- would hand the next chunk a stale timestamp).
    /// Only call this at the exact position an op_a/op_b/op_c slot reads `r` (never for an
    /// untracked "peek" -- see the callers' own comments for which reads are peeks).
    fn reg_read(&mut self, r: Register, position: MemoryAccessPosition) -> u32 {
        let value = self.reg(r);
        self.set_reg(r, value, position);
        value
    }

    /// Writes `value` to register `r`, tagging its consistency-timestamp at `clk + position` --
    /// matches `Executor::rw_cpu`'s exact scheme for the instruction's own op_a/op_b/op_c/hi
    /// slots (the only registers this ISA ever writes through those slots; `op_b`/`op_c` are
    /// always source-only, so there is no `set_reg` counterpart for those positions).
    fn set_reg(&mut self, r: Register, value: u32, position: MemoryAccessPosition) {
        // `$zero` is hardware-wired to 0 (mirrors `Executor::rw_cpu`'s identical guard) -- real
        // code does write to it sometimes (`op_a == 0` idioms like `add $zero, ...`, handled by
        // `AluX0Chip` on the AIR side), and every later read of `$zero` must still see 0 or
        // downstream computation silently corrupts.
        let value = if r == Register::ZERO { 0 } else { value };
        self.registers[r as usize] = value;
        self.register_timestamps[r as usize] = self.clk + position as u64;
    }

    /// Writes `value` to register `r` for a syscall-internal auxiliary register access -- one not
    /// part of the `SYSCALL` instruction's own op_a/op_b/op_c encoding (e.g. `$a3`/`$heap`).
    /// Mirrors `Executor::rw_traced`'s scheme via `SyscallContext`, which tags these at the
    /// syscall's base `clk` with no position offset (`SyscallContext::clk` is captured once at
    /// dispatch, before any of a syscall's own intra-dispatch `clk` bumps).
    fn set_reg_aux(&mut self, r: Register, value: u32) {
        let value = if r == Register::ZERO { 0 } else { value };
        self.registers[r as usize] = value;
        self.register_timestamps[r as usize] = self.clk;
    }

    /// Auxiliary-register counterpart to `reg_read` (bare `clk`, no position offset) -- see both
    /// doc comments.
    fn reg_read_aux(&mut self, r: Register) -> u32 {
        let value = self.reg(r);
        self.set_reg_aux(r, value);
        value
    }

    // ---- RAM (oracle-logged) ----

    /// Read a word from RAM, logging its preimage. Matches `Executor::mr_cpu`'s semantics
    /// (timestamped at `MemoryAccessPosition::Memory`, i.e. exactly `self.clk`). Logging is
    /// suppressed while unconstrained -- see `Self::enter_unconstrained`'s doc comment: nothing
    /// computed inside the block should be observable to a later replay.
    fn mr(&mut self, addr: u32) -> u32 {
        let hint_seed = &mut self.hint_seed;
        let record = self.page_table.entry(addr).or_insert_with(|| {
            MemoryRecord { value: hint_seed.remove(&addr).unwrap_or(0), timestamp: 0 }
        });
        if !self.unconstrained {
            self.oracle_log.push(MemValue { clk: record.timestamp, value: record.value });
        }
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
    /// Logging is suppressed while unconstrained -- see `mr`'s identical comment.
    fn mw(&mut self, addr: u32, value: u32) {
        let hint_seed = &mut self.hint_seed;
        let record = self.page_table.entry(addr).or_insert_with(|| {
            MemoryRecord { value: hint_seed.remove(&addr).unwrap_or(0), timestamp: 0 }
        });
        if !self.unconstrained {
            self.oracle_log.push(MemValue { clk: record.timestamp, value: record.value });
        }
        record.value = value;
        record.timestamp = self.clk;
    }

    /// Read a word from RAM for a syscall (`WRITE`'s memory-to-bytes extraction), *without*
    /// updating its timestamp (matches `Executor::word`'s untracked-peek semantics exactly) but
    /// *does* log the preimage as an oracle entry -- unlike `word_peek`, nothing else logs this
    /// value, and `CoreVM` has no backing RAM at all to recover it from otherwise. Logging is
    /// suppressed while unconstrained -- see `mr`'s identical comment (`WRITE` is one of the only
    /// two syscalls allowed inside a block, so this path is real and reachable there).
    fn mr_log_only(&mut self, addr: u32) -> u32 {
        let record = self.page_table.entry(addr).or_insert(MemoryRecord { value: 0, timestamp: 0 });
        if !self.unconstrained {
            self.oracle_log.push(MemValue { clk: record.timestamp, value: record.value });
        }
        record.value
    }

    /// Reads `len` consecutive words starting at `addr`, each logged individually via `mr`.
    fn mr_slice(&mut self, addr: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| self.mr(addr + i * 4)).collect()
    }

    /// Writes `values` starting at `addr`, each logged individually via `mw`.
    fn mw_slice(&mut self, addr: u32, values: &[u32]) {
        for (i, &value) in values.iter().enumerate() {
            self.mw(addr + i as u32 * 4, value);
        }
    }

    /// Peeks `len` consecutive words starting at `addr`, untracked -- see `word_peek`'s doc
    /// comment on when this is safe to use (only when an immediately-following `mw`/`mw_slice` to
    /// the same addresses logs the same preimage anyway).
    fn slice_peek(&self, addr: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| self.word_peek(addr + i * 4)).collect()
    }

    fn byte_peek(&mut self, addr: u32) -> u8 {
        let word = self.mr_log_only(addr - addr % 4);
        (word >> ((addr % 4) * 8)) as u8
    }

    // ---- clk / delay-slot-safe chunk cutoff ----

    /// Whether the trace buffer is full *and* it is safe to cut here: a chunk boundary must never
    /// split a branch/jump from its delay slot, mirroring the same `!next_is_delayslot` gate the
    /// legacy executor's shard-cut check uses. Likewise, a chunk boundary must never fall inside
    /// an unconstrained block -- a later chunk's `start_registers`/`start_pc` snapshot would then
    /// capture mid-block (not-yet-rolled-back) state that `CoreVM`'s replay, which never visits
    /// any instruction inside the block at all, could never independently reproduce.
    fn chunk_full(&self) -> bool {
        self.oracle_log.len() as u64 >= self.max_trace_size
            && !self.next_is_delayslot
            && !self.unconstrained
    }

    // ---- unconstrained mode (`unconstrained!{}` guest blocks) ----

    /// Enters an unconstrained block: snapshots registers/register-timestamps/`pc` (see
    /// `UnconstrainedSnapshot`) and switches `page_table` into COW mode (`Self::exit_unconstrained`
    /// discards the overlay in O(1), unlike a diff-based undo log -- see
    /// `crate::memory::MaybeCowMemory`'s doc comment). Everything computed for real from this
    /// point until the matching `EXIT_UNCONSTRAINED` -- register writes, RAM writes, even further
    /// nested control flow -- is real, genuine execution (this is *not* a no-op skip), but is
    /// guaranteed to be fully discarded on exit and is never oracle-logged in the meantime (see
    /// `mr`/`mw`/`mr_log_only`'s matching guards), so it's invisible to any later replay.
    ///
    /// Returns `1` (matches `Executor::EnterUnconstrainedSyscall`'s `Ok(Some(1))`) -- the
    /// guest-side `unconstrained!{}` macro branches on this value to
    /// decide whether to execute the block for real (see the macro's own doc comment for why this
    /// is also what makes the whole block invisible to `CoreVM`'s replay: replay unconditionally
    /// computes `0` for this syscall instead, so it always takes the "don't enter" branch).
    ///
    /// # Panics
    ///
    /// If called while already inside an unconstrained block (guest bug, mirrors `Executor`'s
    /// identical panic).
    fn enter_unconstrained(&mut self) -> u32 {
        assert!(
            self.unconstrained_snapshot.is_none(),
            "ENTER_UNCONSTRAINED called while already inside an unconstrained block"
        );
        self.unconstrained_snapshot = Some(UnconstrainedSnapshot {
            registers: self.registers,
            register_timestamps: self.register_timestamps,
            pc: self.pc,
        });
        self.unconstrained = true;
        self.page_table.copy_on_write();
        1
    }

    /// Exits an unconstrained block: restores the register/pc snapshot taken on entry, discards
    /// the `page_table` COW overlay (reverting to its pre-block state), and returns the `next_pc`
    /// the caller should resume at -- `snapshot.pc + 4`, i.e. the instruction immediately after
    /// the *original* `ENTER_UNCONSTRAINED` syscall, not after this `EXIT_UNCONSTRAINED` one. This
    /// is what makes the whole block (including the exit syscall's own instruction) disappear
    /// from view: execution "resumes" exactly where the guest macro's own branch-on-`enter`'s-
    /// return-value check lives, and since registers were just rolled back, that check now
    /// evaluates false and falls straight through, never re-entering the block.
    ///
    /// A no-op returning a garbage `next_pc` if called outside an unconstrained block would be a
    /// guest bug; like the legacy `Executor`, other syscalls already reject running inside a block
    /// (see `syscall.rs`), so this is only ever reached in the correct state -- still, mirrors
    /// `Executor::ExitUnconstrainedSyscall`'s own defensive `if ctx.rt.unconstrained` shape via the
    /// `expect` below rather than silently computing nonsense.
    fn exit_unconstrained(&mut self) -> u32 {
        let snapshot = self
            .unconstrained_snapshot
            .take()
            .expect("EXIT_UNCONSTRAINED called while not inside an unconstrained block");
        self.registers = snapshot.registers;
        self.register_timestamps = snapshot.register_timestamps;
        self.page_table.discard_cow();
        self.unconstrained = false;
        snapshot.pc.wrapping_add(4)
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
        let start_register_timestamps = self.register_timestamps;
        let pc_start = self.pc;
        let clk_start = self.clk;
        let start_input_stream_ptr = self.input_stream_ptr;

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
            start_register_timestamps,
            pc_start,
            clk_start,
            clk_end: self.clk,
            start_input_stream_ptr,
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
        // `clk` only advances for instructions *outside* an unconstrained block. Gated on the
        // state *before* this instruction's own dispatch (not after): `ENTER_UNCONSTRAINED`
        // itself is the last instruction that still advances `clk` (its own dispatch sets
        // `unconstrained = true` only partway through), while `EXIT_UNCONSTRAINED` -- despite
        // clearing the flag partway through its own dispatch -- is the last one that does *not*.
        // This means `clk` never needs to be snapshotted/rolled back at all: it simply never
        // moved during the whole block, so it's already exactly where a replayer (which never
        // visits any instruction inside the block) independently computes it to be. See
        // `Self::enter_unconstrained`'s doc comment for the full picture.
        let was_unconstrained = self.unconstrained;
        if !was_unconstrained {
            self.clk = vm::bump_clk_high_if_need(self.clk, self.max_syscall_cycles);
        }
        self.execute_operation(&instruction)?;
        if !was_unconstrained {
            self.clk += 5;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_operation(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let next_pc_in = self.next_pc;
        let mut next_next_pc = self.next_pc.wrapping_add(4);
        self.next_is_delayslot = false;

        if !self.unconstrained {
            self.opcode_counts[instruction.opcode] += 1;
        }

        if instruction.is_alu_instruction() {
            let (rd, b, c) = self.alu_operands(instruction);
            let (a, hi) = vm::alu_compute(instruction.opcode, b, c)?;
            self.alu_write(instruction.opcode, rd, a, hi);
        } else if instruction.is_memory_load_instruction() {
            self.execute_load(instruction)?;
        } else if instruction.is_memory_store_instruction() {
            self.execute_store(instruction)?;
        } else if instruction.is_branch_instruction() {
            // Must read B before A (mirrors `Executor::branch_rr`'s exact order -- matches
            // `MemoryAccessPosition`'s documented C-B-A ordering: if the two operands alias the
            // same register, B's record must chain from the pre-instruction state and A's from
            // B's).
            let rs: Register = instruction.op_a.into();
            let src2 = if instruction.opcode.only_one_operand() {
                0
            } else {
                self.reg_read((instruction.op_b as u8).into(), MemoryAccessPosition::B)
            };
            let src1 = self.reg_read(rs, MemoryAccessPosition::A);
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
                    let target_pc = self.reg_read(target_reg, MemoryAccessPosition::B);
                    vm::jump_jr_result(next_pc_in, target_pc)
                }
                Opcode::Jumpi => vm::jump_jumpi_result(next_pc_in, instruction.op_b),
                Opcode::JumpDirect => vm::jump_direct_result(next_pc_in, instruction.op_b),
                _ => unreachable!("not a jump opcode: {:?}", instruction.opcode),
            };
            self.set_reg(link, return_pc, MemoryAccessPosition::A);
            next_next_pc = target;
            self.next_is_delayslot = true;
        } else if instruction.is_mov_cond_instruction() {
            let rd: Register = instruction.op_a.into();
            let rs: Register = (instruction.op_b as u8).into();
            let rt: Register = (instruction.op_c as u8).into();
            // `prev_a` is an untracked live peek -- matches `TracingVM`'s identical pattern (the
            // real `a_record` comes entirely from the write below, which captures this same value
            // as its own `prev_value`), so it must NOT re-stamp `rd`'s timestamp here.
            let prev_a = self.reg(rd);
            // Must read C before B (mirrors `Executor::execute_condmov`'s exact order -- see the
            // branch case's identical comment on why this matters when `rs == rt`).
            let c = self.reg_read(rt, MemoryAccessPosition::C);
            let b = self.reg_read(rs, MemoryAccessPosition::B);
            let a = vm::condmov_result(instruction.opcode, prev_a, b, c);
            self.set_reg(rd, a, MemoryAccessPosition::A);
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
    fn alu_operands(&mut self, instruction: &Instruction) -> (Register, u32, u32) {
        if !instruction.imm_c {
            let rd = instruction.op_a.into();
            // Must read C before B (mirrors `Executor::alu_rr`'s exact order) -- if the two
            // operands alias the same register, C's record must chain from the pre-instruction
            // state and B's from C's, matching `MemoryAccessPosition`'s documented C-B-A order.
            let c = self.reg_read((instruction.op_c as u8).into(), MemoryAccessPosition::C);
            let b = self.reg_read((instruction.op_b as u8).into(), MemoryAccessPosition::B);
            (rd, b, c)
        } else if !instruction.imm_b {
            let rd = instruction.op_a.into();
            let b = self.reg_read((instruction.op_b as u8).into(), MemoryAccessPosition::B);
            (rd, b, instruction.op_c)
        } else {
            (instruction.op_a.into(), instruction.op_b, instruction.op_c)
        }
    }

    /// Mirrors `Executor::alu_rw`: dual-result opcodes write LO/HI, everything else writes `rd`.
    fn alu_write(&mut self, opcode: Opcode, rd: Register, a: u32, hi: u32) {
        if opcode.is_use_lo_hi_alu() {
            self.set_reg(Register::LO, a, MemoryAccessPosition::A);
            self.set_reg(Register::HI, hi, MemoryAccessPosition::HI);
        } else {
            self.set_reg(rd, a, MemoryAccessPosition::A);
        }
    }

    fn execute_load(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs_raw = self.reg_read(rs_reg, MemoryAccessPosition::B);
        // `rt`'s current value is an untracked live peek (only needed for LWL/LWR's byte-merge) --
        // matches `TracingVM`'s identical comment: the real `a_record` comes entirely from the
        // write below, which captures this same value as its own `prev_value`.
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
        self.set_reg(rt_reg, val, MemoryAccessPosition::A);
        Ok(())
    }

    fn execute_store(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs = self.reg_read(rs_reg, MemoryAccessPosition::B);
        // `SC`'s `rt` (the value about to be stored) is an untracked live peek -- unlike every
        // other store, which reads it as a real record at A (see `TracingVM::execute_store`'s
        // identical comment: SC's own A-slot is a *write* of the success flag, not a read of
        // `rt`).
        let rt = self.reg(rt_reg);
        if instruction.opcode != Opcode::SC {
            self.set_reg(rt_reg, rt, MemoryAccessPosition::A);
        }

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
            self.set_reg(rt_reg, 1, MemoryAccessPosition::A);
        }
        Ok(())
    }

    fn execute_misc(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        if instruction.opcode == Opcode::WSBH {
            let rd: Register = instruction.op_a.into();
            let rt: Register = (instruction.op_b as u8).into();
            let b = self.reg_read(rt, MemoryAccessPosition::B);
            self.set_reg(rd, vm::wsbh(b), MemoryAccessPosition::A);
            return Ok(());
        }

        let rd: Register = instruction.op_a.into();
        let rt: Register = (instruction.op_b as u8).into();
        let c = instruction.op_c;
        match instruction.opcode {
            Opcode::SEXT => {
                let b = self.reg_read(rt, MemoryAccessPosition::B);
                self.set_reg(rd, vm::sext(b, c), MemoryAccessPosition::A);
            }
            Opcode::EXT => {
                let b = self.reg_read(rt, MemoryAccessPosition::B);
                self.set_reg(rd, vm::ext(b, c)?, MemoryAccessPosition::A);
            }
            Opcode::INS => {
                let b = self.reg_read(rt, MemoryAccessPosition::B);
                // `a` (rd's current value, merged with `b`) is an untracked live peek -- matches
                // `TracingVM`'s identical `INS` pattern.
                let a = self.reg(rd);
                self.set_reg(rd, vm::ins(a, b, c)?, MemoryAccessPosition::A);
            }
            Opcode::TEQ => {
                // `execute_teq`'s unusual encoding: `rs = op_a`, `rt = op_b` (no destination).
                let rs: Register = instruction.op_a.into();
                let rt: Register = (instruction.op_b as u8).into();
                let src2 = self.reg_read(rt, MemoryAccessPosition::B);
                let src1 = self.reg_read(rs, MemoryAccessPosition::A);
                vm::teq(src1, src2)?;
            }
            Opcode::MADDU | Opcode::MSUBU | Opcode::MADD | Opcode::MSUB => {
                let lo_reg: Register = instruction.op_a.into();
                let rs: Register = (instruction.op_c as u8).into();
                let c = self.reg_read(rs, MemoryAccessPosition::C);
                let b = self.reg_read(rt, MemoryAccessPosition::B);
                // `lo`/`hi` are untracked live peeks -- matches `TracingVM`'s identical pattern
                // (only used as computation inputs; the real records come from the writes below).
                let lo = self.reg(Register::LO);
                let hi = self.reg(Register::HI);
                let (out_lo, out_hi) = match instruction.opcode {
                    Opcode::MADDU => vm::maddu(b, c, lo, hi),
                    Opcode::MSUBU => vm::msubu(b, c, lo, hi),
                    Opcode::MADD => vm::madd(b, c, lo, hi),
                    Opcode::MSUB => vm::msub(b, c, lo, hi),
                    _ => unreachable!(),
                };
                self.set_reg(lo_reg, out_lo, MemoryAccessPosition::A);
                self.set_reg(Register::HI, out_hi, MemoryAccessPosition::HI);
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
        programs::tests::{
            ed_decompress_program, fibonacci_program, halt_only_program, hello_world_program,
            simple_memory_program, simple_program,
        },
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

    #[test]
    fn matches_golden_ed_decompress_real_elf() {
        assert_matches_golden(ed_decompress_program, "ed_decompress_program");
    }

    #[test]
    fn matches_golden_simple_memory_program() {
        assert_matches_golden(simple_memory_program, "simple_memory_program");
    }

    /// `golden::run_golden` drives the legacy `Executor` via its single-pass `run()`, whose
    /// `emit_global_memory_events` defaults to `true` -- unlike `golden.rs`'s
    /// `local_memory_access_events` case, this field genuinely IS populated by that path, so its
    /// count is a real cross-check for `global_memory_events`.
    fn assert_global_memory_events_match_golden(program: impl Fn() -> Program, name: &str) {
        let golden = run_golden(program());
        let mut exec = MinimalExecutor::new(Arc::new(program()), 4);
        while exec.try_execute_chunk().unwrap().is_some() {}
        let (initialize_events, finalize_events) = exec.global_memory_events();
        assert_eq!(
            initialize_events.len(),
            golden.event_counts.get("global_memory_initialize_events").copied().unwrap_or(0),
            "{name}: global_memory_initialize_events count mismatch"
        );
        assert_eq!(
            finalize_events.len(),
            golden.event_counts.get("global_memory_finalize_events").copied().unwrap_or(0),
            "{name}: global_memory_finalize_events count mismatch"
        );
    }

    #[test]
    fn global_memory_events_matches_golden_simple_program() {
        assert_global_memory_events_match_golden(simple_program, "simple_program");
    }

    #[test]
    fn global_memory_events_matches_golden_halt_only_program() {
        assert_global_memory_events_match_golden(halt_only_program, "halt_only_program");
    }

    #[test]
    fn global_memory_events_matches_golden_simple_memory_program() {
        assert_global_memory_events_match_golden(simple_memory_program, "simple_memory_program");
    }

    #[test]
    fn global_memory_events_matches_golden_fibonacci_real_elf() {
        assert_global_memory_events_match_golden(fibonacci_program, "fibonacci_program");
    }

    #[test]
    fn global_memory_events_matches_golden_hello_world_real_elf() {
        assert_global_memory_events_match_golden(hello_world_program, "hello_world_program");
    }

    #[test]
    fn global_memory_events_matches_golden_ed_decompress_real_elf() {
        assert_global_memory_events_match_golden(ed_decompress_program, "ed_decompress_program");
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
