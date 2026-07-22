//! `SplicingVM` owns the *real*, cost-model-based shard-cut decision -- deliberately kept out of
//! `MinimalRunner` so that module's contract stays minimal enough for a different execution
//! engine to implement in its place later. It walks a [`Chunk`]'s value stream forward through a
//! fresh `CoreVM<Oracle>` (reusing the same cost model `Executor::inc_shard_if_need` uses:
//! `cpu_exit`/`clk_exit`/shape-check against
//! `estimate_mips_event_counts`/`estimate_mips_lde_size`), and cuts it into shard-sized
//! [`SplicedChunk`] pieces. Each `Chunk` is spliced independently -- if a shard's cost threshold
//! hasn't been reached by the time the chunk's own instruction stream runs out, the in-progress
//! shard is force-closed there rather than carried into a later call, so different `Chunk`s can
//! be spliced concurrently by a worker pool instead of needing one continuous sequential walk.

use hashbrown::HashMap;

use crate::{
    cost::{estimate_mips_event_counts, estimate_mips_lde_size, pad_mips_event_counts},
    executor::{LocalCounts, CORE_SHARD_CLK_LIMIT, CORE_SHARD_HEIGHT_THRESHOLD},
    minimal::Chunk,
    register::NUM_REGISTERS,
    vm::{CoreVM, MemValue, Oracle},
    ExecutionError, MipsAirId, Program,
};

/// One shard's worth of a program's execution, ready to be traced into an `ExecutionRecord` by
/// `TracingVM`.
pub struct SplicedChunk {
    pub oracle: Vec<MemValue>,
    pub pc_start: u32,
    pub next_pc_start: u32,
    pub global_clk_start: u64,
    /// The global, never-resetting memory-access timestamp value this shard's local `clk == 0`
    /// corresponds to. See `ExecutionState::initial_timestamp`'s doc comment.
    pub initial_timestamp: u64,
    /// This shard's starting register state -- `TracingVM` is reconstructed fresh per shard (no
    /// continuity across shards, unlike `SplicingVM`), so unlike `pc_start`/`clk_start` it can't
    /// fall back on `CoreVM::new()`'s own (program-image-only) seeding once execution has moved
    /// past the first shard. See `CoreVM::registers`'s doc comment for why registers need this
    /// threading at all instead of flowing through `oracle` like general memory.
    pub registers_start: [MemValue; NUM_REGISTERS],
    /// Only scoped to the `Chunk` this piece came from -- `SplicingVM::splice_chunk` doesn't know
    /// the true global shard index, since chunks are spliced independently and may be processed
    /// out of order by a worker pool. The caller must overwrite this with the real global index
    /// (tracked sequentially downstream) before constructing a `TracingVM` from it.
    pub shard: u32,
    /// True if the program halted at the end of this piece.
    pub done: bool,
}

/// Decides real shard cuts and slices `Chunk`s into `SplicedChunk`s. Holds only immutable
/// per-program config -- stateless between `splice_chunk` calls, so it can be shared (cloned)
/// across a worker pool splicing multiple `Chunk`s concurrently.
pub struct SplicingVM {
    program: std::sync::Arc<Program>,
    max_syscall_cycles: u32,
    shard_size: u32,
    shape_check_frequency: u64,
    lde_size_threshold: u64,
    costs: HashMap<MipsAirId, u64>,
    program_size: u64,
}

impl SplicingVM {
    #[must_use]
    pub fn new(
        program: std::sync::Arc<Program>,
        max_syscall_cycles: u32,
        shard_size: u32,
        shape_check_frequency: u64,
        lde_size_threshold: u64,
        costs: HashMap<MipsAirId, u64>,
    ) -> Self {
        let program_size = (program.instructions.len() as u64).next_power_of_two();
        Self {
            program,
            max_syscall_cycles,
            shard_size,
            shape_check_frequency,
            lde_size_threshold,
            costs,
            program_size,
        }
    }

    fn should_cut_shard(&self, core: &CoreVM<Oracle>) -> bool {
        // Cycles consumed so far within the shard currently being built -- `clk` itself no
        // longer resets per shard (see `ExecutionState::clk`'s doc comment).
        let cycles_this_shard = core.clk - core.initial_timestamp;
        let cpu_exit =
            u64::from(self.max_syscall_cycles) + cycles_this_shard >= u64::from(self.shard_size);
        let clk_exit = u64::from(self.max_syscall_cycles) + cycles_this_shard
            >= u64::from(CORE_SHARD_CLK_LIMIT);

        // Never let a shard's clk range cross a `clk_high` (top bits above the 28-bit low
        // window) boundary -- keeps `clk_high` constant within a shard's own trace, so the AIR
        // never needs to reason about it changing mid-shard.
        let window_exit = (core.clk >> 28) != (core.initial_timestamp >> 28);

        let mut shape_match_found = true;
        if core.global_clk.is_multiple_of(self.shape_check_frequency) {
            let event_counts = estimate_mips_event_counts(
                core.local_counts.local_mem as u64,
                core.local_counts.syscalls_sent as u64,
                core.local_counts.addi_events,
                *core.local_counts.event_counts,
            );
            let padded_event_counts =
                pad_mips_event_counts(event_counts, self.shape_check_frequency);
            let padded_lde_size =
                estimate_mips_lde_size(padded_event_counts, &self.costs, self.program_size);
            if padded_lde_size > self.lde_size_threshold {
                shape_match_found = false;
            }
            if let Some(max_chip_height) = padded_event_counts.iter().map(|(_, h)| *h).max() {
                if max_chip_height >= CORE_SHARD_HEIGHT_THRESHOLD {
                    shape_match_found = false;
                }
            }
        }

        cpu_exit || clk_exit || window_exit || !shape_match_found
    }

    /// Splice a `Chunk` into shard-sized `SplicedChunk` pieces. May return zero pieces (if the
    /// whole chunk doesn't reach the shard-cost threshold -- rare in practice, since
    /// `minimal_trace_chunk_threshold` is sized well above a typical shard), one, or several. The
    /// last piece is force-closed at the chunk's own end if the cost threshold hasn't naturally
    /// been hit yet (see this module's doc comment).
    ///
    /// # Errors
    /// Returns an error if replaying the chunk's instructions fails.
    pub fn splice_chunk(&self, chunk: Chunk) -> Result<Vec<SplicedChunk>, ExecutionError> {
        // Kept around so we can re-slice `[consumed_before..consumed_now]` after each `step()` --
        // `Oracle` consumes its values via a one-shot `IntoIter` and can't be re-sliced itself,
        // and we can't hold a borrow of it across `step()` (which needs `&mut core.mem`) anyway.
        let chunk_values = chunk.oracle;
        let chunk_len = chunk_values.len();

        let mut core = CoreVM::new(self.program.clone(), Oracle::new(chunk_values.clone()));
        core.pc = chunk.pc_start;
        core.next_pc = chunk.next_pc_start;
        core.clk = chunk.clk_start;
        core.global_clk = chunk.global_clk_start;
        core.registers = chunk.registers_start;
        // `CoreVM::new()` defaults this to `1` (the program's first shard) -- every chunk after
        // the first starts partway through the execution, at whatever clk this chunk's own
        // (already force-closed, per this module's doc comment) predecessor shard ended at.
        core.initial_timestamp = chunk.clk_start;

        let mut pending_oracle = Vec::new();
        let mut pending_pc_start = chunk.pc_start;
        let mut pending_next_pc_start = chunk.next_pc_start;
        let mut pending_global_clk_start = chunk.global_clk_start;
        let mut pending_initial_timestamp = chunk.clk_start;
        let mut pending_registers_start = chunk.registers_start;
        let mut local_shard = 0u32;

        let mut result = Vec::new();
        let mut consumed_before = 0usize;

        loop {
            let outcome = core.step()?;
            let consumed_now = chunk_len - core.mem.remaining();

            if outcome.done {
                pending_oracle.extend_from_slice(&chunk_values[consumed_before..consumed_now]);
                result.push(SplicedChunk {
                    oracle: std::mem::take(&mut pending_oracle),
                    pc_start: pending_pc_start,
                    next_pc_start: pending_next_pc_start,
                    global_clk_start: pending_global_clk_start,
                    initial_timestamp: pending_initial_timestamp,
                    registers_start: pending_registers_start,
                    shard: local_shard,
                    done: true,
                });
                return Ok(result);
            }

            // We restrict the execution of branch/jump and its delay slot to be in the same
            // shard, matching `Executor::execute`'s `!self.state.next_is_delayslot` guard.
            if !core.next_is_delayslot && self.should_cut_shard(&core) {
                pending_oracle.extend_from_slice(&chunk_values[consumed_before..consumed_now]);
                result.push(SplicedChunk {
                    oracle: std::mem::take(&mut pending_oracle),
                    pc_start: pending_pc_start,
                    next_pc_start: pending_next_pc_start,
                    global_clk_start: pending_global_clk_start,
                    initial_timestamp: pending_initial_timestamp,
                    registers_start: pending_registers_start,
                    shard: local_shard,
                    done: false,
                });
                consumed_before = consumed_now;

                local_shard += 1;
                core.initial_timestamp = core.clk;
                core.local_counts = LocalCounts::default();
                pending_pc_start = core.pc;
                pending_next_pc_start = core.next_pc;
                pending_global_clk_start = core.global_clk;
                pending_initial_timestamp = core.initial_timestamp;
                pending_registers_start = core.registers;
            }

            if core.global_clk >= chunk.global_clk_end {
                // Chunk exhausted -- force-close whatever shard is still in progress instead of
                // carrying it into a later call (there's no cross-call continuity anymore: each
                // chunk is spliced independently, see this module's doc comment). Reaching this
                // point always means the program hasn't halted yet: the `outcome.done` branch
                // above already returns immediately on the step that does halt, before this
                // check is ever reached on that same iteration.
                pending_oracle.extend_from_slice(&chunk_values[consumed_before..consumed_now]);
                result.push(SplicedChunk {
                    oracle: pending_oracle,
                    pc_start: pending_pc_start,
                    next_pc_start: pending_next_pc_start,
                    global_clk_start: pending_global_clk_start,
                    initial_timestamp: pending_initial_timestamp,
                    registers_start: pending_registers_start,
                    shard: local_shard,
                    done: false,
                });
                return Ok(result);
            }
        }
    }
}
