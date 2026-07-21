use hashbrown::HashMap;

use crate::{
    events::{MemoryLocalEvent, MemoryReadRecord, MemoryRecordEnum, MemoryWriteRecord, PrecompileEvent, SyscallEvent},
    record::ExecutionRecord,
    ExecutionError, Executor, ExecutorMode, Program, Register,
};

use super::SyscallCode;

/// Everything a [`Syscall`](super::Syscall) implementation needs from whatever engine is running
/// it. Implemented by [`Executor`] (unchanged, real behavior) and by `CoreVM<M>` (used by
/// `MinimalRunner`/`SplicingVM`/`TracingVM`), so every existing precompile keeps working
/// unchanged across all of them.
///
/// Host-visible side effects (`stdout_line`/`stderr_line`/`invoke_hook`/cycle-tracker reporting)
/// are real only for `Executor` and `CoreVM<Live>`; `CoreVM<Oracle>` (which replays the same
/// instruction stream) implements them as no-ops, since those effects must happen exactly once.
pub trait SyscallRuntime {
    /// The current shard.
    fn shard(&self) -> u32;
    /// The current clock cycle.
    fn clk(&self) -> u64;
    /// The current, globalized memory-access timestamp (`initial_timestamp + clk`, no position
    /// offset -- every access within one syscall shares this same value, unlike CPU-level
    /// accesses which use `MemoryAccessPosition` to space out same-instruction sub-accesses).
    fn timestamp(&self) -> u64;
    /// The current program counter.
    fn pc(&self) -> u32;
    /// Whether we're inside an unconstrained block.
    fn is_unconstrained(&self) -> bool;
    /// The program being executed.
    fn program(&self) -> &Program;
    /// The global clock (used for cycle-tracker timing).
    fn global_clk(&self) -> u64;

    /// Read a word from memory and create an access record. `external` marks an access whose
    /// real shard isn't known yet (deferred precompile events) -- see
    /// `ExecutionState::initial_timestamp`'s doc comment for what replaces the old shard-equality
    /// check.
    fn mr(
        &mut self,
        addr: u32,
        external: bool,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord;
    /// Write a word to memory and create an access record.
    fn mw(
        &mut self,
        addr: u32,
        value: u32,
        external: bool,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord;
    /// Read a register and create an access record.
    fn rr_traced(
        &mut self,
        register: Register,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord;
    /// Write a register and create an access record.
    fn rw_traced(
        &mut self,
        register: Register,
        value: u32,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord;

    /// Get the current value of a register, without creating an access record.
    fn register(&mut self, register: Register) -> u32;
    /// Get the current value of a word, without creating an access record.
    fn word(&mut self, addr: u32) -> u32;
    /// Get the current value of a byte, without creating an access record.
    fn byte(&mut self, addr: u32) -> u8;

    /// A mutable reference to the execution record currently being built. `MinimalRunner`
    /// and `SplicingVM` provide a scratch record here (not otherwise inspected): they still
    /// need `record_mut()` to satisfy syscalls like `COMMIT` that write straight into it, but
    /// neither of them is responsible for the final record's public values (that's `TracingVM`
    /// plus the `generate_records` orchestration layer that propagates them shard-to-shard,
    /// exactly like today's `prove.rs` stage-2 loop already does).
    fn record_mut(&mut self) -> &mut ExecutionRecord;
    /// Whether typed events (precompile events, etc.) should actually be recorded. True only for
    /// `Executor` in `Trace` mode and for `TracingVM`.
    fn is_recording_events(&self) -> bool;
    /// The "outer" local-memory-access map used by non-syscall instruction execution in the
    /// current shard (`Executor::local_memory_access` / `TracingVM`'s equivalent). Used by
    /// [`SyscallContext::postprocess`] to correctly chain a syscall's own memory-access span with
    /// whatever span (if any) an earlier, non-syscall instruction in the same shard already
    /// opened for the same address.
    fn outer_local_memory_access(&mut self) -> &mut HashMap<u32, MemoryLocalEvent>;
    /// Add a precompile event to the execution record, if `is_recording_events()`.
    fn add_precompile_event(
        &mut self,
        syscall_code: SyscallCode,
        syscall_event: SyscallEvent,
        event: PrecompileEvent,
    ) {
        if self.is_recording_events() {
            self.record_mut().precompile_events.add_event(syscall_code, syscall_event, event);
        }
    }
    /// Build a [`SyscallEvent`] for the current instruction.
    fn syscall_event(
        &self,
        clk: u64,
        a_record: Option<MemoryRecordEnum>,
        next_pc: u32,
        syscall_id: u32,
        arg1: u32,
        arg2: u32,
    ) -> SyscallEvent;

    /// Enter an unconstrained block: snapshot whatever state needs to be rolled back on exit.
    fn enter_unconstrained(&mut self);
    /// Exit an unconstrained block: roll back to the state at the matching `enter_unconstrained`.
    fn exit_unconstrained(&mut self);

    /// Seed the value an address should read as, the first time it's touched (used by
    /// `SYSHINTREAD`). No-op for oracle-sourced runtimes -- the oracle already encodes the
    /// correct first-touch value.
    fn seed_uninitialized(&mut self, addr: u32, value: u32) -> Result<(), ExecutionError>;
    /// Peek at the next input-stream item without consuming it (`SYSHINTLEN`).
    fn peek_input(&self) -> Option<&Vec<u8>>;
    /// Consume and return the next input-stream item (`SYSHINTREAD`).
    fn consume_input(&mut self) -> Option<Vec<u8>>;
    /// Push a new item onto the input stream (`WRITE` to `FD_HINT`, or a hook's injected result).
    fn push_hint_input(&mut self, bytes: Vec<u8>);

    /// Verify the next deferred proof against `(vkey, pv_digest)` (`VERIFY_ZKM_PROOF`/`SYSVERIFY`).
    /// No-op for oracle-sourced runtimes: verification only needs to happen once, against the real
    /// memory source.
    fn verify_deferred_proof(&mut self, vkey: [u32; 8], pv_digest: [u32; 8]) -> Result<(), ExecutionError> {
        let _ = (vkey, pv_digest);
        Ok(())
    }

    /// Append bytes to the public values stream (`WRITE` to `FD_PUBLIC_VALUES`).
    fn write_public_values(&mut self, bytes: &[u8]);
    /// Emit a complete stdout line. No-op for oracle-sourced runtimes.
    fn stdout_line(&mut self, _line: &str) {}
    /// Emit a complete stderr line. No-op for oracle-sourced runtimes.
    fn stderr_line(&mut self, _line: &str) {}
    /// Buffer partial output for `fd` until a full line is available, returning completed lines.
    fn io_buf_push(&mut self, fd: u32, s: &str) -> Vec<String>;
    /// Invoke a registered hook for `fd`, if one exists. No-op (returns `Ok(None)`) for
    /// oracle-sourced runtimes -- hooks may have external side effects and must run exactly once.
    fn invoke_hook(&mut self, fd: u32, buf: &[u8]) -> Result<Option<Vec<Vec<u8>>>, ExecutionError>;

    /// Start a cycle tracker span. No-op for oracle-sourced runtimes (diagnostic only).
    fn cycle_tracker_start(&mut self, _name: &str) {}
    /// End a cycle tracker span, returning its cycle count. No-op for oracle-sourced runtimes.
    fn cycle_tracker_end(&mut self, _name: &str) -> Option<u64> {
        None
    }
    /// Accumulate a cycle tracker span's cycles into the execution report. No-op for
    /// oracle-sourced runtimes.
    fn cycle_tracker_report(&mut self, _name: &str, _total_cycles: u64) {}
}

impl SyscallRuntime for Executor<'_> {
    fn shard(&self) -> u32 {
        Executor::shard(self)
    }

    fn clk(&self) -> u64 {
        self.state.clk
    }

    fn timestamp(&self) -> u64 {
        self.state.clk
    }

    fn pc(&self) -> u32 {
        self.state.pc
    }

    fn is_unconstrained(&self) -> bool {
        self.unconstrained
    }

    fn program(&self) -> &Program {
        &self.program
    }

    fn global_clk(&self) -> u64 {
        self.state.global_clk
    }

    fn mr(
        &mut self,
        addr: u32,
        external: bool,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        Executor::mr(self, addr, external, clk, local_memory_access)
    }

    fn mw(
        &mut self,
        addr: u32,
        value: u32,
        external: bool,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        Executor::mw(self, addr, value, external, clk, local_memory_access)
    }

    fn rr_traced(
        &mut self,
        register: Register,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        Executor::rr_traced(self, register, clk, local_memory_access)
    }

    fn rw_traced(
        &mut self,
        register: Register,
        value: u32,
        clk: u64,
        local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        Executor::rw_traced(self, register, value, clk, local_memory_access)
    }

    fn register(&mut self, register: Register) -> u32 {
        Executor::register(self, register)
    }

    fn word(&mut self, addr: u32) -> u32 {
        Executor::word(self, addr)
    }

    fn byte(&mut self, addr: u32) -> u8 {
        Executor::byte(self, addr)
    }

    fn record_mut(&mut self) -> &mut ExecutionRecord {
        &mut self.record
    }

    fn is_recording_events(&self) -> bool {
        self.executor_mode == ExecutorMode::Trace
    }

    fn outer_local_memory_access(&mut self) -> &mut HashMap<u32, MemoryLocalEvent> {
        &mut self.local_memory_access
    }

    fn syscall_event(
        &self,
        clk: u64,
        a_record: Option<MemoryRecordEnum>,
        next_pc: u32,
        syscall_id: u32,
        arg1: u32,
        arg2: u32,
    ) -> SyscallEvent {
        Executor::syscall_event(self, clk, a_record, next_pc, syscall_id, arg1, arg2)
    }

    fn enter_unconstrained(&mut self) {
        assert!(!self.unconstrained, "Unconstrained block is already active.");
        self.unconstrained = true;
        self.unconstrained_state = crate::state::ForkState {
            global_clk: self.state.global_clk,
            clk: self.state.clk,
            pc: self.state.pc,
            memory_diff: HashMap::default(),
            record: std::mem::take(&mut self.record),
            op_record: std::mem::take(&mut self.memory_accesses),
            executor_mode: self.executor_mode,
        };
        self.executor_mode = ExecutorMode::Simple;
    }

    fn exit_unconstrained(&mut self) {
        if self.unconstrained {
            self.state.global_clk = self.unconstrained_state.global_clk;
            self.state.clk = self.unconstrained_state.clk;
            self.state.pc = self.unconstrained_state.pc;
            for (addr, value) in self.unconstrained_state.memory_diff.drain() {
                match value {
                    Some(value) => {
                        self.state.memory.insert(addr, value);
                    }
                    None => {
                        self.state.memory.remove(addr);
                    }
                }
            }
            self.record = std::mem::take(&mut self.unconstrained_state.record);
            self.memory_accesses = std::mem::take(&mut self.unconstrained_state.op_record);
            self.executor_mode = self.unconstrained_state.executor_mode;
            self.unconstrained = false;
        }
        self.unconstrained_state = crate::state::ForkState::default();
    }

    fn seed_uninitialized(&mut self, addr: u32, value: u32) -> Result<(), ExecutionError> {
        use crate::memory::Entry;
        match self.state.uninitialized_memory.entry(addr) {
            Entry::Occupied(_) => {
                log::error!("hint read address is initialized already");
                Err(ExecutionError::InvalidSyscallArgs())
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
        }
    }

    fn peek_input(&self) -> Option<&Vec<u8>> {
        self.state.input_stream.get(self.state.input_stream_ptr)
    }

    fn consume_input(&mut self) -> Option<Vec<u8>> {
        let item = self.state.input_stream.get(self.state.input_stream_ptr).cloned();
        if item.is_some() {
            self.state.input_stream_ptr += 1;
        }
        item
    }

    fn push_hint_input(&mut self, bytes: Vec<u8>) {
        let ptr = self.state.input_stream_ptr;
        self.state.input_stream.splice(ptr..ptr, std::iter::once(bytes));
    }

    fn write_public_values(&mut self, bytes: &[u8]) {
        self.state.public_values_stream.extend_from_slice(bytes);
    }

    fn stdout_line(&mut self, line: &str) {
        println!("stdout: {line}");
    }

    fn stderr_line(&mut self, line: &str) {
        println!("stderr: {line}");
    }

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

    fn invoke_hook(&mut self, fd: u32, buf: &[u8]) -> Result<Option<Vec<Vec<u8>>>, ExecutionError> {
        if let Some(mut hook) = self.hook_registry.get(fd) {
            let res = hook.invoke_hook(self.hook_env(), buf)?;
            Ok(Some(res))
        } else {
            Ok(None)
        }
    }

    fn cycle_tracker_start(&mut self, name: &str) {
        let depth = self.cycle_tracker.len() as u32;
        self.cycle_tracker.insert(name.to_string(), (self.state.global_clk, depth));
        let padding = "│ ".repeat(depth as usize);
        log::info!("{padding}┌╴{name}");
    }

    fn cycle_tracker_end(&mut self, name: &str) -> Option<u64> {
        if let Some((start, depth)) = self.cycle_tracker.remove(name) {
            let padding = "│ ".repeat(depth as usize);
            let total_cycles = self.state.global_clk - start;
            log::info!("{}└╴{} cycles", padding, zkm_primitives::consts::num_to_comma_separated(total_cycles));
            return Some(total_cycles);
        }
        None
    }

    fn cycle_tracker_report(&mut self, name: &str, total_cycles: u64) {
        self.report
            .cycle_tracker
            .entry(name.to_string())
            .and_modify(|cycles| *cycles += total_cycles)
            .or_insert(total_cycles);
    }

    fn verify_deferred_proof(&mut self, vkey: [u32; 8], pv_digest: [u32; 8]) -> Result<(), ExecutionError> {
        if self.deferred_proof_verification == crate::DeferredProofVerification::Disabled {
            return Ok(());
        }

        let proof_index = self.state.proof_stream_ptr;
        if proof_index >= self.state.proof_stream.len() {
            panic!("Not enough proofs were written to the runtime.");
        }
        let (proof, proof_vk) = &self.state.proof_stream[proof_index];
        self.state.proof_stream_ptr += 1;

        if let Some(verifier) = self.subproof_verifier {
            if let Err(e) = verifier.verify_deferred_proof(proof, proof_vk, vkey, pv_digest) {
                log::error!(
                    "Failed to verify proof {proof_index} with digest {}: {}",
                    hex::encode(bytemuck::cast_slice(&pv_digest)),
                    e
                );
                return Err(ExecutionError::ExceptionOrTrap());
            }
        } else if self.state.proof_stream_ptr == 1 {
            tracing::info!("Not verifying sub proof during runtime");
        }

        Ok(())
    }
}

/// A runtime for syscalls that is protected so that developers cannot arbitrarily modify the
/// runtime. Generic over the underlying [`SyscallRuntime`] so every precompile implementation
/// works unchanged for `Executor` and for `CoreVM<M>` (phases 1-3 of `generate_records`).
pub struct SyscallContext<'a, R: SyscallRuntime> {
    /// The current shard.
    pub current_shard: u32,
    /// The clock cycle every `mr`/`mw`/`rr_traced`/`rw_traced` call made through this context
    /// uses. Starts at the runtime's own current clk and is otherwise left to the syscall
    /// implementation to manage -- most precompiles never touch it, so every access within the
    /// syscall shares one timestamp, but a few (see e.g. `Sha256CompressSyscall`) bump it
    /// mid-syscall to give a later phase's accesses a distinct, later timestamp, matching their
    /// own AIR's `clk + <phase offset>` expectation.
    pub clk: u64,
    /// The next program counter.
    pub next_pc: u32,
    /// The exit code.
    pub exit_code: u32,
    /// The runtime.
    pub rt: &'a mut R,
    /// The local memory access events for the syscall.
    pub local_memory_access: HashMap<u32, MemoryLocalEvent>,
}

impl<'a, R: SyscallRuntime> SyscallContext<'a, R> {
    /// Create a new [`SyscallContext`].
    pub fn new(runtime: &'a mut R) -> Self {
        let current_shard = runtime.shard();
        let clk = runtime.clk();
        let next_pc = runtime.pc().wrapping_add(4);
        Self {
            current_shard,
            clk,
            next_pc,
            exit_code: 0,
            rt: runtime,
            local_memory_access: HashMap::new(),
        }
    }

    /// Get a mutable reference to the execution record.
    pub fn record_mut(&mut self) -> &mut ExecutionRecord {
        self.rt.record_mut()
    }

    #[inline]
    /// Add a precompile event to the execution record.
    pub fn add_precompile_event(
        &mut self,
        syscall_code: SyscallCode,
        syscall_event: SyscallEvent,
        event: PrecompileEvent,
    ) {
        self.rt.add_precompile_event(syscall_code, syscall_event, event);
    }

    /// Get the current shard.
    #[must_use]
    pub fn current_shard(&self) -> u32 {
        self.current_shard
    }

    /// Read a word from memory.
    pub fn mr(&mut self, addr: u32) -> (MemoryReadRecord, u32) {
        let record =
            self.rt.mr(addr, true, self.clk, Some(&mut self.local_memory_access));
        (record, record.value)
    }

    /// Read a slice of words from memory.
    pub fn mr_slice(&mut self, addr: u32, len: usize) -> (Vec<MemoryReadRecord>, Vec<u32>) {
        let mut records = Vec::with_capacity(len);
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let (record, value) = self.mr(addr + i as u32 * 4);
            records.push(record);
            values.push(value);
        }
        (records, values)
    }

    /// Write a word to memory.
    pub fn mw(&mut self, addr: u32, value: u32) -> MemoryWriteRecord {
        self.rt.mw(addr, value, true, self.clk, Some(&mut self.local_memory_access))
    }

    /// Write a slice of words to memory.
    pub fn mw_slice(&mut self, addr: u32, values: &[u32]) -> Vec<MemoryWriteRecord> {
        let mut records = Vec::with_capacity(values.len());
        #[allow(clippy::needless_range_loop)]
        for i in 0..values.len() {
            let record = self.mw(addr + i as u32 * 4, values[i]);
            records.push(record);
        }
        records
    }

    /// Read a register and record the memory access.
    pub fn rr_traced(&mut self, register: Register) -> (MemoryReadRecord, u32) {
        let record = self.rt.rr_traced(
            register,
            self.clk,
            Some(&mut self.local_memory_access),
        );
        (record, record.value)
    }

    /// Write a register and record the memory access.
    pub fn rw_traced(&mut self, register: Register, value: u32) -> MemoryWriteRecord {
        self.rt.rw_traced(
            register,
            value,
            self.clk,
            Some(&mut self.local_memory_access),
        )
    }

    /// Postprocess the syscall.  Specifically will process the syscall's memory local events.
    pub fn postprocess(&mut self) -> Vec<MemoryLocalEvent> {
        let mut syscall_local_mem_events = Vec::new();

        if !self.rt.is_unconstrained() && self.rt.is_recording_events() {
            // Will need to transfer the existing memory local events in the executor to it's record,
            // and return all the syscall memory local events.  This is similar to what
            // `bump_record` does.
            for (addr, event) in self.local_memory_access.drain() {
                let local_mem_access = self.rt.outer_local_memory_access().remove(&addr);

                if let Some(local_mem_access) = local_mem_access {
                    self.rt.record_mut().cpu_local_memory_access.push(local_mem_access);
                }

                syscall_local_mem_events.push(event);
            }
        }

        syscall_local_mem_events
    }

    /// Get the current value of a register, but doesn't use a memory record.
    /// This is generally unconstrained, so you must be careful using it.
    #[must_use]
    pub fn register_unsafe(&mut self, register: Register) -> u32 {
        self.rt.register(register)
    }

    /// Get the current value of a byte, but doesn't use a memory record.
    #[must_use]
    pub fn byte_unsafe(&mut self, addr: u32) -> u8 {
        self.rt.byte(addr)
    }

    /// Get the current value of a word, but doesn't use a memory record.
    #[must_use]
    pub fn word_unsafe(&mut self, addr: u32) -> u32 {
        self.rt.word(addr)
    }

    /// Get a slice of words, but doesn't use a memory record.
    #[must_use]
    pub fn slice_unsafe(&mut self, addr: u32, len: usize) -> Vec<u32> {
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            values.push(self.rt.word(addr + i as u32 * 4));
        }
        values
    }

    /// Set the next program counter.
    pub fn set_next_pc(&mut self, next_pc: u32) {
        self.next_pc = next_pc;
    }

    /// Set the exit code.
    pub fn set_exit_code(&mut self, exit_code: u32) {
        self.exit_code = exit_code;
    }
}
