//! `MinimalRunner` wraps `CoreVM<Live>` and does nothing but run the program for real (loads,
//! branches, syscalls all need real values and real control flow) and buffer the resulting value
//! stream into bounded [`Chunk`]s. It makes no shard-cut decisions and does no cost accounting,
//! which keeps its contract minimal enough for a different execution engine to implement in its
//! place later without touching `splicing.rs`/`tracing_chunk.rs`/`vm.rs`.

use std::sync::Arc;

use crate::{
    events::{MemoryInitializeFinalizeEvent, MemoryRecord},
    record::ExecutionRecord,
    register::NUM_REGISTERS,
    vm::{CoreVM, Live},
    ExecutionError, Program,
};

/// A bounded slice of the program's execution: enough to later be spliced into shard-sized
/// pieces and traced into `ExecutionRecord`s. `oracle` carries only *values* (see
/// `crate::vm::Oracle`'s doc comment for why shard/timestamp tags are deliberately not included).
pub struct Chunk {
    pub oracle: Vec<u32>,
    pub pc_start: u32,
    pub next_pc_start: u32,
    pub clk_start: u64,
    pub global_clk_start: u64,
    /// True if the program halted during this chunk (it is therefore the last one).
    pub done: bool,
}

/// Produces a stream of [`Chunk`]s from a program + inputs.
pub struct MinimalRunner {
    pub core: CoreVM<Live>,
    /// Number of oracle values to buffer before yielding a chunk. Independent of shard economics
    /// -- this bounds peak memory of the buffered value stream, not shard size (`ZKMCoreOpts`'s
    /// `minimal_trace_chunk_threshold`).
    chunk_threshold: u64,
}

impl MinimalRunner {
    #[must_use]
    pub fn new(program: Arc<Program>, chunk_threshold: u64) -> Self {
        let mut core = CoreVM::new(program, Live::new());
        core.load_image();
        Self { core, chunk_threshold }
    }

    /// Add an item to the input stream (`stdin`).
    pub fn with_input(&mut self, input: &[u8]) {
        self.core.input_stream.push_back(input.to_vec());
    }

    #[must_use]
    pub fn program(&self) -> &Arc<Program> {
        &self.core.program
    }

    #[must_use]
    pub fn global_clk(&self) -> u64 {
        self.core.global_clk
    }

    #[must_use]
    pub fn public_values_stream(&self) -> &[u8] {
        &self.core.public_values_stream
    }

    /// Run until the chunk-size bound is hit or the program halts, returning the resulting
    /// [`Chunk`]. Returns `Ok(None)` if the program has already halted (nothing left to produce).
    ///
    /// # Errors
    /// Returns an error if execution fails (invalid instruction, out-of-bounds access, etc).
    pub fn try_next_chunk(&mut self) -> Result<Option<Chunk>, ExecutionError> {
        if self.core.exited || self.core.pc == 0 {
            return Ok(None);
        }

        let pc_start = self.core.pc;
        let next_pc_start = self.core.next_pc;
        let clk_start = self.core.clk;
        let global_clk_start = self.core.global_clk;

        loop {
            let outcome = self.core.step()?;
            if outcome.done || self.core.mem.oracle_out.len() as u64 >= self.chunk_threshold {
                let oracle = std::mem::take(&mut self.core.mem.oracle_out);
                return Ok(Some(Chunk {
                    oracle,
                    pc_start,
                    next_pc_start,
                    clk_start,
                    global_clk_start,
                    done: outcome.done,
                }));
            }
        }
    }

    /// Emit the program's global memory initialize/finalize events into `record`, for every
    /// address ever touched. Must only be called once, after the final chunk (`done`) -- ported
    /// from `Executor::postprocess`'s memory-events section, reading from this runner's still-live
    /// `Live` memory instead of `Executor`'s.
    pub fn emit_globals(&self, record: &mut ExecutionRecord) {
        let memory = &self.core.mem.memory;
        let uninitialized_memory = &self.core.mem.uninitialized_memory;
        let program = &self.core.program;

        let addr_0_final_record = match memory.get(0) {
            Some(record) => *record,
            None => MemoryRecord { value: 0, timestamp: 1 },
        };
        record
            .global_memory_finalize_events
            .push(MemoryInitializeFinalizeEvent::finalize_from_record(0, &addr_0_final_record));
        record.global_memory_initialize_events.push(MemoryInitializeFinalizeEvent::initialize(0, 0));

        for addr in 1..NUM_REGISTERS as u32 {
            if let Some(reg_record) = memory.registers.get(addr) {
                if !program.image.contains_key(&addr) {
                    let initial_value = uninitialized_memory.registers.get(addr).copied().unwrap_or(0);
                    record
                        .global_memory_initialize_events
                        .push(MemoryInitializeFinalizeEvent::initialize(addr, initial_value));
                }
                record
                    .global_memory_finalize_events
                    .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, reg_record));
            }
        }

        for addr in memory.page_table.keys() {
            if addr == 0 {
                continue;
            }
            if !program.image.contains_key(&addr) {
                let initial_value = uninitialized_memory.get(addr).copied().unwrap_or(0);
                record
                    .global_memory_initialize_events
                    .push(MemoryInitializeFinalizeEvent::initialize(addr, initial_value));
            }
            let mem_record = *memory.get(addr).unwrap();
            record
                .global_memory_finalize_events
                .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &mem_record));
        }
    }
}
