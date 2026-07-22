//! `MinimalRunner` wraps `MinimalExecutor` and does nothing but run the program for real (loads,
//! branches, syscalls all need real values and real control flow) and buffer the resulting value
//! stream into bounded [`Chunk`]s. It makes no shard-cut decisions and does no cost accounting,
//! which keeps its contract minimal enough for a different execution engine to implement in its
//! place later without touching `splicing.rs`/`tracing_chunk.rs`/`vm.rs`.

use std::sync::Arc;

use executor::MinimalExecutor;

use crate::{
    memory::{Memory, PagedMemory},
    register::NUM_REGISTERS,
    vm::MemValue,
    ExecutionError, Program,
};

mod executor;

/// A bounded slice of the program's execution: enough to later be spliced into shard-sized
/// pieces and traced into `ExecutionRecord`s. Each `oracle` entry carries both the value and the
/// timestamp a memory/register access saw immediately before it (see `crate::vm::MemValue`'s doc
/// comment).
pub struct Chunk {
    pub oracle: Vec<MemValue>,
    pub pc_start: u32,
    pub next_pc_start: u32,
    pub clk_start: u64,
    pub global_clk_start: u64,
    /// This chunk's starting register state -- `SplicingVM` replays each chunk independently (no
    /// continuity across chunks, so it can run as a worker pool), so unlike a single continuous
    /// walk it can't fall back on carrying `CoreVM::registers` forward from the previous chunk.
    /// See `SplicedChunk::registers_start`'s doc comment for the same pattern one level down.
    pub registers_start: [MemValue; NUM_REGISTERS],
    /// The `global_clk` value immediately after this chunk's last instruction. `SplicingVM`
    /// replays exactly `global_clk_end - global_clk_start` instructions before treating this
    /// chunk as exhausted -- unlike before registers were split out of the oracle (see
    /// `CoreVM::registers`'s doc comment), oracle exhaustion (`Oracle::remaining() == 0`) is no
    /// longer a reliable proxy for "done with this chunk": a run of register-only instructions
    /// (e.g. plain ALU ops) consumes zero oracle entries yet still needs replaying.
    pub global_clk_end: u64,
    /// True if the program halted during this chunk (it is therefore the last one).
    pub done: bool,
}

/// Produces a stream of [`Chunk`]s from a program + inputs.
pub struct MinimalRunner {
    core: MinimalExecutor,
    /// Number of oracle values to buffer before yielding a chunk. Independent of shard economics
    /// -- this bounds peak memory of the buffered value stream, not shard size (`ZKMCoreOpts`'s
    /// `minimal_trace_chunk_threshold`).
    chunk_threshold: u64,
    /// Whether [`Self::try_next_chunk`] has produced a chunk yet. `HaltSyscall::execute`
    /// unconditionally sets `next_pc` to `0` as the "the program is done" marker (checked
    /// regardless of exit code, unlike `CoreVM::exited`, which is only set for a zero exit code)
    /// -- but `0` is also a perfectly legitimate starting `pc` (every synthetic, non-ELF test
    /// program in this crate uses one). Gating the `pc == 0` check on having already produced at
    /// least one chunk disambiguates "just halted" from "hasn't run yet".
    started: bool,
}

impl MinimalRunner {
    #[must_use]
    pub fn new(program: Arc<Program>, chunk_threshold: u64) -> Self {
        let mut core = MinimalExecutor::new(program);
        core.load_image();
        Self { core, chunk_threshold, started: false }
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

    /// This runner's still-live registers/memory, for a caller (`tracing_chunk::emit_globals`) to
    /// read the final state of every address ever touched from, after the final chunk (`done`).
    /// Read-only: `MinimalRunner` itself never builds `MemoryInitializeFinalizeEvent`s or any
    /// other typed event/record content.
    #[must_use]
    pub fn registers(&self) -> &[MemValue; NUM_REGISTERS] {
        &self.core.registers
    }

    /// Whether each register has ever been touched -- see `MinimalExecutor::registers_touched`'s
    /// doc comment.
    #[must_use]
    pub fn registers_touched(&self) -> &[bool; NUM_REGISTERS] {
        &self.core.registers_touched
    }

    /// See [`Self::registers`].
    #[must_use]
    pub fn memory(&self) -> &PagedMemory<MemValue> {
        &self.core.memory
    }

    /// See [`Self::registers`].
    #[must_use]
    pub fn uninitialized_memory(&self) -> &Memory<u32> {
        &self.core.uninitialized_memory
    }

    /// Run until the chunk-size bound is hit or the program halts, returning the resulting
    /// [`Chunk`]. Returns `Ok(None)` if the program has already halted (nothing left to produce).
    ///
    /// # Errors
    /// Returns an error if execution fails (invalid instruction, out-of-bounds access, etc).
    pub fn try_next_chunk(&mut self) -> Result<Option<Chunk>, ExecutionError> {
        if self.core.exited || (self.started && self.core.pc == 0) {
            return Ok(None);
        }
        self.started = true;

        let pc_start = self.core.pc;
        let next_pc_start = self.core.next_pc;
        let clk_start = self.core.clk;
        let global_clk_start = self.core.global_clk;
        let registers_start = self.core.registers;

        loop {
            let done = self.core.execute_instruction()?;
            if done || self.core.oracle_out.len() as u64 >= self.chunk_threshold {
                let oracle = std::mem::take(&mut self.core.oracle_out);
                return Ok(Some(Chunk {
                    oracle,
                    pc_start,
                    next_pc_start,
                    clk_start,
                    global_clk_start,
                    registers_start,
                    global_clk_end: self.core.global_clk,
                    done,
                }));
            }
        }
    }

}
