//! Phase 2 of `generate_records`: the real shard-cut decision.
//!
//! `SplicingVM` owns the *real*, cost-model-based shard-cut decision -- deliberately kept out of
//! phase 1 (`MinimalRunner`) so phase 1's contract stays minimal enough for a future JIT engine
//! to replace wholesale; see the port plan's "Why shard-cut logic lives in `SplicingVM`" section.
//! It walks a [`Chunk`]'s value stream forward through `CoreVM<Oracle>` (reusing the exact
//! cost-model `Executor::inc_shard_if_need` uses today: `cpu_exit`/`clk_exit`/shape-check against
//! `estimate_mips_event_counts`/`estimate_mips_lde_size`), and cuts it into shard-sized
//! [`SplicedChunk`] pieces, carrying a piece across chunk boundaries when a shard doesn't close
//! within one chunk (mirrors SP1's `last_splice`/`splice_to_emit` carry-over in
//! `splice_chunk_sequential`).

use hashbrown::HashMap;

use crate::{
    cost::{estimate_mips_event_counts, estimate_mips_lde_size, pad_mips_event_counts},
    executor::{CORE_SHARD_CLK_LIMIT, CORE_SHARD_HEIGHT_THRESHOLD},
    minimal::Chunk,
    vm::{CoreVM, Oracle},
    ExecutionError, MipsAirId, Program,
};

/// One shard's worth of a program's execution, ready to be traced into an `ExecutionRecord` by
/// `TracingVM`.
pub struct SplicedChunk {
    pub oracle: Vec<u32>,
    pub pc_start: u32,
    pub next_pc_start: u32,
    pub global_clk_start: u64,
    pub shard: u32,
    /// True if the program halted at the end of this piece.
    pub done: bool,
}

/// Phase 2: decides real shard cuts and slices `Chunk`s into `SplicedChunk`s.
pub struct SplicingVM {
    core: CoreVM<Oracle>,
    /// Oracle values belonging to the shard currently being accumulated, spanning however many
    /// `Chunk`s it takes to close (usually one).
    pending_oracle: Vec<u32>,
    pending_pc_start: u32,
    pending_next_pc_start: u32,
    pending_global_clk_start: u64,

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
        let pc_start = program.pc_start;
        let next_pc_start = program.next_pc;
        let core = CoreVM::new(program, Oracle::new(Vec::new(), HashMap::new()));
        Self {
            core,
            pending_oracle: Vec::new(),
            pending_pc_start: pc_start,
            pending_next_pc_start: next_pc_start,
            pending_global_clk_start: 0,
            max_syscall_cycles,
            shard_size,
            shape_check_frequency,
            lde_size_threshold,
            costs,
            program_size,
        }
    }

    /// Add an item to the input stream (`stdin`) -- must mirror `MinimalRunner::with_input`
    /// exactly (same items, same order), since `HINT_LEN`/`HINT_READ` replay through this same
    /// stream during splicing and must see what phase 1 saw.
    pub fn with_input(&mut self, input: &[u8]) {
        self.core.input_stream.push_back(input.to_vec());
    }

    fn should_cut_shard(&mut self) -> bool {
        let cpu_exit = self.max_syscall_cycles + self.core.clk >= self.shard_size;
        let clk_exit = self.max_syscall_cycles + self.core.clk >= CORE_SHARD_CLK_LIMIT;

        let mut shape_match_found = true;
        if self.core.global_clk.is_multiple_of(self.shape_check_frequency) {
            let event_counts = estimate_mips_event_counts(
                self.core.local_counts.local_mem as u64,
                self.core.local_counts.syscalls_sent as u64,
                self.core.local_counts.addi_events,
                *self.core.local_counts.event_counts,
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

        cpu_exit || clk_exit || !shape_match_found
    }

    /// Splice a `Chunk` into shard-sized `SplicedChunk` pieces. May return zero pieces (if the
    /// chunk doesn't close any shard -- its values are folded into `pending_oracle` for the next
    /// call), one, or several.
    ///
    /// # Errors
    /// Returns an error if replaying the chunk's instructions fails.
    pub fn splice_chunk(&mut self, chunk: Chunk) -> Result<Vec<SplicedChunk>, ExecutionError> {
        // Kept around so we can re-slice `[consumed_before..consumed_now]` after each `step()` --
        // `Oracle` consumes its values via a one-shot `IntoIter` and can't be re-sliced itself,
        // and we can't hold a borrow of it across `step()` (which needs `&mut self.core.mem`)
        // anyway.
        let chunk_values = chunk.oracle;
        let chunk_len = chunk_values.len();

        let tags = std::mem::replace(&mut self.core.mem, Oracle::new(Vec::new(), HashMap::new()))
            .into_tags();
        self.core.mem = Oracle::new(chunk_values.clone(), tags);
        self.core.pc = chunk.pc_start;
        self.core.next_pc = chunk.next_pc_start;
        self.core.clk = chunk.clk_start;
        self.core.global_clk = chunk.global_clk_start;

        let mut result = Vec::new();
        let mut consumed_before = 0usize;

        loop {
            let outcome = self.core.step()?;
            let consumed_now = chunk_len - self.core.mem.remaining();

            if outcome.done {
                self.pending_oracle.extend_from_slice(&chunk_values[consumed_before..consumed_now]);
                result.push(SplicedChunk {
                    oracle: std::mem::take(&mut self.pending_oracle),
                    pc_start: self.pending_pc_start,
                    next_pc_start: self.pending_next_pc_start,
                    global_clk_start: self.pending_global_clk_start,
                    shard: self.core.current_shard,
                    done: true,
                });
                return Ok(result);
            }

            // We restrict the execution of branch/jump and its delay slot to be in the same
            // shard, matching `Executor::execute`'s `!self.state.next_is_delayslot` guard.
            if !self.core.next_is_delayslot && self.should_cut_shard() {
                self.pending_oracle.extend_from_slice(&chunk_values[consumed_before..consumed_now]);
                result.push(SplicedChunk {
                    oracle: std::mem::take(&mut self.pending_oracle),
                    pc_start: self.pending_pc_start,
                    next_pc_start: self.pending_next_pc_start,
                    global_clk_start: self.pending_global_clk_start,
                    shard: self.core.current_shard,
                    done: false,
                });
                consumed_before = consumed_now;

                self.core.current_shard += 1;
                self.core.clk = 0;
                self.core.local_counts = crate::executor::LocalCounts::default();
                self.pending_pc_start = self.core.pc;
                self.pending_next_pc_start = self.core.next_pc;
                self.pending_global_clk_start = self.core.global_clk;
            }

            if self.core.mem.remaining() == 0 {
                // Chunk exhausted without closing the in-progress shard -- carry it into the
                // next chunk.
                self.pending_oracle.extend_from_slice(&chunk_values[consumed_before..consumed_now]);
                return Ok(result);
            }
        }
    }
}
