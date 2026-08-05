//! The oracle-log ("minimal trace") contract.
//!
//! A [`MinimalTrace`] is the minimum information needed to re-execute a program from
//! `pc_start`/`clk_start` to `clk_end`: a starting register snapshot, plus a flat, ordered,
//! address-free log of every value that is genuinely non-deterministic *from a replayer's point
//! of view* -- RAM read/write preimages and syscall/precompile write results. Register-only
//! ALU/branch/jump results and RAM addresses themselves are never logged; they are pure functions
//! of already-known live state, recomputed identically by every consumer (`MinimalExecutor`,
//! `CoreVM`, `SplicingVM`, `TracingVM`).

#![allow(dead_code)]

use std::sync::Arc;

use crate::register::NUM_REGISTERS;

/// One oracle-log entry: a RAM preimage `(timestamp, value)` at the moment it was read, or (for a
/// syscall/precompile write) the literal post-image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemValue {
    /// The timestamp (`clk`) the logged value was last written at. `0` means "never touched",
    /// mirroring `MemoryRecord::timestamp`'s sentinel.
    pub clk: u64,
    /// The logged value.
    pub value: u32,
}

/// A cursor over a [`MinimalTrace`]'s oracle log, consumed strictly in order by `CoreVM`.
/// Bounded (`end` may be short of the backing slice's own length) so `SplicedMinimalTrace` can
/// hand a shard exactly its own window of a larger chunk's log, not a suffix running past its own
/// boundary into the next shard's entries.
#[derive(Debug, Clone)]
pub(crate) struct MemReads<'a> {
    data: &'a [MemValue],
    pos: usize,
    end: usize,
}

impl<'a> MemReads<'a> {
    #[must_use]
    pub(crate) fn new(data: &'a [MemValue]) -> Self {
        Self { data, pos: 0, end: data.len() }
    }

    /// A cursor over `data[start..end]` -- used by `SplicedMinimalTrace` to hand out exactly one
    /// shard's slice of a larger chunk's oracle log.
    #[must_use]
    pub(crate) fn bounded(data: &'a [MemValue], start: usize, end: usize) -> Self {
        Self { data, pos: start, end: end.min(data.len()) }
    }

    /// Remaining (unconsumed) entry count.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.end.saturating_sub(self.pos)
    }
}

impl Iterator for MemReads<'_> {
    type Item = MemValue;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Some(v)
    }
}

/// The oracle-log contract every consumer (`CoreVM`/`SplicingVM`/`TracingVM`) is generic over.
/// `MinimalExecutor` produces the concrete implementation, [`TraceChunk`]; `SplicedMinimalTrace`
/// wraps one to add a starting-state offset without copying the log.
pub(crate) trait MinimalTrace: Clone + Send + Sync + 'static {
    /// The register file at `pc_start`/`clk_start`.
    fn start_registers(&self) -> [u32; NUM_REGISTERS];
    /// The `clk` each register was last written at, as of `pc_start`/`clk_start` -- `0` means
    /// "never touched" (see `MinimalExecutor`'s `register_timestamps` doc comment). Needed so a
    /// replayer can recover a real `prev_timestamp` for a register whose last write happened in
    /// an *earlier* chunk/shard than the one it's currently replaying.
    fn start_register_timestamps(&self) -> [u64; NUM_REGISTERS];
    /// The `pc` to begin replay at.
    fn pc_start(&self) -> u32;
    /// The `clk` to begin replay at.
    fn clk_start(&self) -> u64;
    /// The `clk` replay must stop at (either the chunk's buffer-size cutoff or program halt).
    fn clk_end(&self) -> u64;
    /// Total number of oracle-log entries.
    fn num_mem_reads(&self) -> u64;
    /// A cursor over the oracle log, starting from its first entry.
    fn mem_reads(&self) -> MemReads<'_>;
    /// The oracle log's backing slice. Every `MinimalTrace` implementation here is a real, owned,
    /// contiguous slice -- `SplicedMinimalTrace` uses this directly to build a bounded
    /// [`MemReads`] rather than re-deriving one via `mem_reads()` + N discarded `.next()` calls.
    fn mem_reads_slice(&self) -> &[MemValue];
}

/// A single chunk of a [`crate::minimal::MinimalExecutor`] run: a starting snapshot plus the
/// oracle log accumulated between `clk_start` and `clk_end`. Chunk boundaries are purely a
/// buffer-size cutoff (see `MinimalExecutor::try_execute_chunk`), unrelated to shard-proving
/// boundaries (`SplicingVM`'s job).
#[derive(Debug, Clone)]
pub(crate) struct TraceChunk {
    pub start_registers: [u32; NUM_REGISTERS],
    pub start_register_timestamps: [u64; NUM_REGISTERS],
    pub pc_start: u32,
    pub clk_start: u64,
    pub clk_end: u64,
    pub mem_reads: Arc<[MemValue]>,
}

impl MinimalTrace for TraceChunk {
    fn start_registers(&self) -> [u32; NUM_REGISTERS] {
        self.start_registers
    }

    fn start_register_timestamps(&self) -> [u64; NUM_REGISTERS] {
        self.start_register_timestamps
    }

    fn pc_start(&self) -> u32 {
        self.pc_start
    }

    fn clk_start(&self) -> u64 {
        self.clk_start
    }

    fn clk_end(&self) -> u64 {
        self.clk_end
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
