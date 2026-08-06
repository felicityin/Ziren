//! Drives the whole `MinimalExecutor` -> `SplicingVM` -> `TracingVM` pipeline for one program
//! run, producing one [`ExecutionRecord`] per shard. This mirrors `prove.rs`'s own
//! `checkpoint_generator_handle`/`trace_checkpoint` flow (driven across a channel and a worker
//! pool), but single-threaded and independently testable without any of that concurrency.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::{
    events::MemoryInitializeFinalizeEvent,
    hook::HookRegistry,
    minimal::MinimalExecutor,
    splicing::{SplicingCarry, SplicingStatus, SplicingVM},
    subproof::SubproofVerifier,
    tracing::TracingVM,
    ExecutionError, ExecutionRecord, ExecutionReport, Program, ZKMReduceProof,
};
use zkm_hypercube::{config::ZkmGlobalContext, verifier::ZkmPcsProofInner, MachineVerifyingKey};

pub use crate::splicing::SplicedMinimalTrace;

/// Runs `program` to completion through the new pipeline, returning one [`ExecutionRecord`] per
/// shard. `max_trace_size` bounds `MinimalExecutor`'s oracle-log buffer (a memory-management
/// cutoff, independent of shard boundaries -- see `SplicingVM`'s module doc); `element_threshold`/
/// `height_threshold` are `ShapeChecker`'s shard-cut thresholds.
///
/// # Errors
///
/// Propagates any [`ExecutionError`] from executing an instruction.
pub fn run_full_pipeline(
    program: Arc<Program>,
    max_trace_size: u64,
    stdin: Arc<[Vec<u8>]>,
    element_threshold: u64,
    height_threshold: u64,
) -> Result<Vec<ExecutionRecord>, ExecutionError> {
    let mut minimal = MinimalExecutor::new_with_stdin(program.clone(), max_trace_size, stdin);
    let max_syscall_cycles = minimal.max_syscall_cycles();

    let mut chunk = minimal
        .try_execute_chunk()?
        .expect("the first try_execute_chunk call always retires at least one instruction");
    let mut splicing = SplicingVM::new(
        &chunk,
        program.clone(),
        max_syscall_cycles,
        element_threshold,
        height_threshold,
    );
    let mut records = Vec::new();
    loop {
        match splicing.execute()? {
            SplicingStatus::ShardBoundary => {
                let spliced = splicing.splice(&chunk);
                records.push(trace_shard(program.clone(), &spliced, max_syscall_cycles)?);
            }
            SplicingStatus::Done => {
                let spliced = splicing.splice(&chunk);
                records.push(trace_shard(program.clone(), &spliced, max_syscall_cycles)?);
                break;
            }
            SplicingStatus::TraceEnd => {
                let carry = splicing.into_carry(&chunk);
                chunk = minimal
                    .try_execute_chunk()?
                    .expect("TraceEnd means the program hasn't halted, so another chunk follows");
                splicing = SplicingVM::resume(carry, &chunk, program.clone(), max_syscall_cycles);
            }
        }
    }

    // Global memory events need the *final* register/RAM state, which only `MinimalExecutor`
    // itself has a view of (see `MinimalExecutor::global_memory_events`'s doc comment) -- attached
    // to the last shard, mirroring the legacy executor's postprocess()-on-final-checkpoint
    // behavior.
    let (initialize_events, finalize_events) = minimal.global_memory_events();
    if let Some(last) = records.last_mut() {
        last.global_memory_initialize_events = initialize_events;
        last.global_memory_finalize_events = finalize_events;
    }

    Ok(records)
}

/// Runs `program` to completion *without* tracing -- the new pipeline's equivalent of
/// `Executor::run_fast`, for `ZKMProver::execute`'s dry-run path (no shard proving, just the
/// public-values stream and a final `ExecutionReport`). Unlike [`run_full_pipeline`], this drives
/// `MinimalExecutor` directly: there is no `SplicingVM`/`TracingVM` replay step to keep in sync, so
/// `hook_registry`/`max_cycles` are supported here even though the real proving pipeline doesn't
/// support them yet (see `MinimalExecutor::stdin`'s doc comment for why that's a real, not
/// incidental, restriction). `max_trace_size` only bounds how much oracle log
/// `MinimalExecutor` buffers at a time before this function discards it (a memory-management knob
/// a caller should size the same way it would size `ZKMCoreOpts::shard_size`) -- it has no effect
/// on the returned result.
///
/// # Errors
///
/// Propagates any [`ExecutionError`] from executing an instruction, including
/// `ExecutionError::ExceededCycleLimit` once `max_cycles` is reached.
#[allow(clippy::too_many_arguments)]
pub fn execute_fast<'a>(
    program: Arc<Program>,
    max_trace_size: u64,
    stdin: Arc<[Vec<u8>]>,
    proof_stream: Vec<(ZKMReduceProof<ZkmGlobalContext, ZkmPcsProofInner>, MachineVerifyingKey<ZkmGlobalContext>)>,
    subproof_verifier: Option<&'a dyn SubproofVerifier>,
    deferred_proof_verification_enabled: bool,
    hook_registry: Option<HookRegistry<'a>>,
    max_cycles: Option<u64>,
) -> Result<(Vec<u8>, ExecutionReport), ExecutionError> {
    let mut minimal = MinimalExecutor::new_with_context(
        program,
        max_trace_size,
        stdin,
        proof_stream,
        subproof_verifier,
        deferred_proof_verification_enabled,
        hook_registry,
        max_cycles,
    );
    while minimal.try_execute_chunk()?.is_some() {}
    Ok((minimal.public_values_stream().to_vec(), minimal.execution_report()))
}

/// Runs `program` to completion with no stdin/hooks/subproofs/cycle limit, returning its final
/// register file -- a minimal single-program helper for tests that just need to check computed
/// register values (instruction-suite-style tests), without the full `execute_fast` argument
/// list.
///
/// # Errors
///
/// Propagates any [`ExecutionError`] from executing an instruction.
pub fn execute_fast_registers(
    program: Arc<Program>,
) -> Result<[u32; crate::register::NUM_REGISTERS], ExecutionError> {
    let mut minimal = MinimalExecutor::new(program, u64::MAX / 2);
    while minimal.try_execute_chunk()?.is_some() {}
    Ok(minimal.registers())
}

/// Traces one already-spliced shard into a real [`ExecutionRecord`] -- the new pipeline's
/// equivalent of the legacy `Executor`-based `trace_checkpoint`.
///
/// # Errors
///
/// Propagates any [`ExecutionError`] from executing an instruction.
pub fn trace_shard(
    program: Arc<Program>,
    spliced: &SplicedMinimalTrace,
    max_syscall_cycles: u32,
) -> Result<ExecutionRecord, ExecutionError> {
    let mut record = ExecutionRecord::new(program.clone());
    let mut tracing = TracingVM::new(spliced, program, max_syscall_cycles, &mut record);
    tracing.execute()?;
    Ok(record)
}

/// Owns a `MinimalExecutor` (kept crate-private) across repeated [`next_shard`] calls, plus
/// whatever [`SplicedMinimalTrace`]s a chunk fetch already produced beyond the one returned so
/// far (`pending`) and, if the in-progress shard's current chunk ran out before completing it, the
/// [`SplicingCarry`] needed to resume it against the next chunk. A `SplicingVM` itself is never
/// stored here -- it borrows its chunk, and that chunk (a `next_shard`-local variable) doesn't
/// live across calls -- only the lifetime-free data `next_shard` needs to reconstruct one does.
pub struct ShardDriver<'a> {
    minimal: MinimalExecutor<'a>,
    pending: VecDeque<(SplicedMinimalTrace, bool)>,
    carry: Option<SplicingCarry>,
}

impl<'a> ShardDriver<'a> {
    /// Convenience constructor for callers that never feed stdin -- equivalent to
    /// `Self::new_with_stdin(program, max_trace_size, Arc::from([]))`.
    #[must_use]
    pub fn new(program: Arc<Program>, max_trace_size: u64) -> Self {
        Self::new_with_stdin(program, max_trace_size, Arc::from([]))
    }

    #[must_use]
    pub fn new_with_stdin(program: Arc<Program>, max_trace_size: u64, stdin: Arc<[Vec<u8>]>) -> Self {
        Self {
            minimal: MinimalExecutor::new_with_stdin(program, max_trace_size, stdin),
            pending: VecDeque::new(),
            carry: None,
        }
    }

    /// The full constructor -- see `MinimalExecutor::new_with_context`'s doc comment. `max_cycles`
    /// stays `None`: enforcing a cycle limit against the real proving pipeline (as opposed to
    /// `execute_fast`'s dry run) isn't implemented yet.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_context(
        program: Arc<Program>,
        max_trace_size: u64,
        stdin: Arc<[Vec<u8>]>,
        proof_stream: Vec<(ZKMReduceProof<ZkmGlobalContext, ZkmPcsProofInner>, MachineVerifyingKey<ZkmGlobalContext>)>,
        subproof_verifier: Option<&'a dyn SubproofVerifier>,
        deferred_proof_verification_enabled: bool,
        hook_registry: Option<HookRegistry<'a>>,
    ) -> Self {
        Self {
            minimal: MinimalExecutor::new_with_context(
                program,
                max_trace_size,
                stdin,
                proof_stream,
                subproof_verifier,
                deferred_proof_verification_enabled,
                hook_registry,
                None,
            ),
            pending: VecDeque::new(),
            carry: None,
        }
    }

    /// The global, never-reset clk `next_shard`'s most recent call left off at -- the new
    /// pipeline's equivalent of the legacy `Executor`'s `state.global_clk`, sent alongside each
    /// checkpoint.
    #[must_use]
    pub fn clk(&self) -> u64 {
        self.minimal.clk()
    }

    /// The `max_syscall_cycles` this run's `bump_clk_high_if_need` uses -- [`trace_shard`]'s
    /// `TracingVM` must be constructed with the same value (see
    /// `MinimalExecutor::max_syscall_cycles`'s doc comment).
    #[must_use]
    pub fn max_syscall_cycles(&self) -> u32 {
        self.minimal.max_syscall_cycles()
    }

    #[must_use]
    pub fn public_values_stream(&self) -> &[u8] {
        self.minimal.public_values_stream()
    }

    /// Only meaningful once the whole run has finished -- call only after [`next_shard`] has
    /// returned `done == true` (see `MinimalExecutor::global_memory_events`'s doc comment).
    #[must_use]
    pub fn global_memory_events(
        &self,
    ) -> (Vec<MemoryInitializeFinalizeEvent>, Vec<MemoryInitializeFinalizeEvent>) {
        self.minimal.global_memory_events()
    }

    /// A snapshot of opcode/syscall dispatch counts accumulated so far, for summary logging.
    #[must_use]
    pub fn execution_report(&self) -> ExecutionReport {
        self.minimal.execution_report()
    }
}

/// Drives `driver` through exactly one shard's worth of chunks (however many that takes),
/// returning the spliced shard and whether the whole run is now done. Callers loop this until it
/// returns `done == true`, threading only `&mut ShardDriver` across calls -- the new pipeline's
/// equivalent of `prove.rs`'s existing `execute_state(false)`-in-a-loop shape, just yielding a
/// [`SplicedMinimalTrace`] in place of a serialized `ExecutionState` checkpoint.
///
/// A `MinimalExecutor` chunk and a shard are cut independently (see `SplicingVM`'s module doc):
/// one chunk commonly contains more than one shard's worth of data whenever `element_threshold`/
/// `height_threshold` (a trace-area/row-count budget) trips well before `max_trace_size` (a raw
/// cycle-count buffer-size cutoff) does -- the common case for precompile-heavy code, where each
/// syscall's trace cost dwarfs its cycle cost. Refilling `driver.pending` below walks a chunk to
/// completion via `SplicingVM::execute`/`splice`, exactly like `run_full_pipeline`'s reference
/// loop, collecting every shard boundary hit along the way instead of returning after the first
/// one -- `splice()` already resets the `SplicingVM`'s own bookkeeping to start the next shard
/// from exactly where the one just spliced off ended, so continuing the loop (rather than
/// discarding that reset, ready-to-continue state by returning) is what makes multiple shards per
/// chunk correct.
///
/// # Errors
///
/// Propagates any [`ExecutionError`] from executing an instruction.
///
/// # Panics
///
/// If called again after a previous call already returned `done == true`.
pub fn next_shard(
    driver: &mut ShardDriver<'_>,
    program: Arc<Program>,
    element_threshold: u64,
    height_threshold: u64,
) -> Result<(SplicedMinimalTrace, bool), ExecutionError> {
    while driver.pending.is_empty() {
        let max_syscall_cycles = driver.minimal.max_syscall_cycles();
        let chunk = driver
            .minimal
            .try_execute_chunk()?
            .expect("next_shard must not be called again after a previous call returned done == true");
        let mut splicing = match driver.carry.take() {
            Some(carry) => SplicingVM::resume(carry, &chunk, program.clone(), max_syscall_cycles),
            None => SplicingVM::new(
                &chunk,
                program.clone(),
                max_syscall_cycles,
                element_threshold,
                height_threshold,
            ),
        };
        loop {
            match splicing.execute()? {
                SplicingStatus::ShardBoundary => {
                    driver.pending.push_back((splicing.splice(&chunk), false));
                }
                SplicingStatus::Done => {
                    driver.pending.push_back((splicing.splice(&chunk), true));
                    break;
                }
                SplicingStatus::TraceEnd => {
                    // This chunk is exhausted with a shard still in progress. If it already
                    // yielded at least one complete shard, stop here and defer fetching another
                    // chunk to the next `next_shard` call (via the stored carry) instead of
                    // eagerly pulling in more than one chunk's worth of data per call.
                    driver.carry = Some(splicing.into_carry(&chunk));
                    break;
                }
            }
        }
    }
    Ok(driver.pending.pop_front().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hook::HookRegistry,
        programs::tests::{fibonacci_program, hook_fp_inverse_program},
    };

    /// `execute_fast` (the `ZKMProver::execute` dry-run path) with the default hook registry
    /// active: `hook_fp_inverse_program` writes an `fp_inverse` request to `FD_FP_INV`, and the
    /// real built-in hook (not a test stub) must compute the correct modular inverse and splice it
    /// back for `HINT_READ` to pick up -- exercises `MinimalExecutor::hook_dispatch` end to end,
    /// including the `WRITE`-then-`HINT_READ` sequencing `write_fd`'s hook branch relies on.
    #[test]
    fn execute_fast_invokes_default_hooks() {
        let (public_values, report) = execute_fast(
            Arc::new(hook_fp_inverse_program()),
            u64::MAX / 2,
            Arc::from([]),
            Vec::new(),
            None,
            true,
            Some(HookRegistry::default()),
            None,
        )
        .unwrap();

        // 3 * 5 = 15 = 2*7 + 1, so 5 is the modular inverse of 3 mod 7.
        assert_eq!(public_values, vec![5, 0, 0, 0], "wrong fp_inverse hook result");
        assert!(report.total_instruction_count() > 0);
    }

    /// Same program, but with `hook_registry: None` -- mirrors legacy's own behavior for a `WRITE`
    /// to an fd with no hook registered at all (a silent no-op, not an error): `HINT_LEN` then
    /// sees an empty `stdin` and errors with `InvalidSyscallArgs`, since nothing ever spliced a
    /// result in for it to read.
    #[test]
    fn execute_fast_without_hooks_never_invokes_them() {
        let result = execute_fast(
            Arc::new(hook_fp_inverse_program()),
            u64::MAX / 2,
            Arc::from([]),
            Vec::new(),
            None,
            true,
            None,
            None,
        );
        assert!(matches!(result, Err(ExecutionError::InvalidSyscallArgs())));
    }

    /// `max_cycles` mirrors `Executor::max_cycles`: exceeding it errors with
    /// `ExecutionError::ExceededCycleLimit` instead of running to completion.
    #[test]
    fn execute_fast_respects_max_cycles() {
        let result = execute_fast(
            Arc::new(fibonacci_program()),
            u64::MAX / 2,
            Arc::from([]),
            Vec::new(),
            None,
            true,
            None,
            Some(1),
        );
        assert!(matches!(result, Err(ExecutionError::ExceededCycleLimit(1))));
    }
}
