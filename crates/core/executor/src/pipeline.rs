//! Drives the whole `MinimalExecutor` -> `SplicingVM` -> `TracingVM` pipeline for one program
//! run, producing one [`ExecutionRecord`] per shard. This is the new pipeline's single-threaded,
//! whole-run equivalent of the legacy `Executor`'s two-pass Checkpoint-then-Trace flow (driven by
//! `prove.rs`'s `checkpoint_generator_handle`/`trace_checkpoint` across a channel and a worker
//! pool) -- a stepping stone toward wiring `prove.rs` itself, and independently testable against
//! `golden.rs` without any of that concurrency.

use std::sync::Arc;

use crate::{
    events::MemoryInitializeFinalizeEvent,
    hook::HookRegistry,
    minimal::MinimalExecutor,
    splicing::{SplicingStatus, SplicingVM},
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

/// Owns a `MinimalExecutor` (kept crate-private) across repeated [`next_shard`] calls, so a shard
/// spanning more than one `MinimalExecutor` chunk never needs a self-referential `SplicingVM` to
/// survive across calls -- each `next_shard` call loops through as many chunks as that one shard
/// needs, entirely within its own stack frame, and only `ShardDriver` itself (opaque to callers)
/// needs to persist between calls.
pub struct ShardDriver<'a>(MinimalExecutor<'a>);

impl<'a> ShardDriver<'a> {
    /// Convenience constructor for callers that never feed stdin -- equivalent to
    /// `Self::new_with_stdin(program, max_trace_size, Arc::from([]))`.
    #[must_use]
    pub fn new(program: Arc<Program>, max_trace_size: u64) -> Self {
        Self::new_with_stdin(program, max_trace_size, Arc::from([]))
    }

    #[must_use]
    pub fn new_with_stdin(program: Arc<Program>, max_trace_size: u64, stdin: Arc<[Vec<u8>]>) -> Self {
        Self(MinimalExecutor::new_with_stdin(program, max_trace_size, stdin))
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
        Self(MinimalExecutor::new_with_context(
            program,
            max_trace_size,
            stdin,
            proof_stream,
            subproof_verifier,
            deferred_proof_verification_enabled,
            hook_registry,
            None,
        ))
    }

    /// The global, never-reset clk `next_shard`'s most recent call left off at -- the new
    /// pipeline's equivalent of the legacy `Executor`'s `state.global_clk`, sent alongside each
    /// checkpoint.
    #[must_use]
    pub fn clk(&self) -> u64 {
        self.0.clk()
    }

    /// The `max_syscall_cycles` this run's `bump_clk_high_if_need` uses -- [`trace_shard`]'s
    /// `TracingVM` must be constructed with the same value (see
    /// `MinimalExecutor::max_syscall_cycles`'s doc comment).
    #[must_use]
    pub fn max_syscall_cycles(&self) -> u32 {
        self.0.max_syscall_cycles()
    }

    #[must_use]
    pub fn public_values_stream(&self) -> &[u8] {
        self.0.public_values_stream()
    }

    /// Only meaningful once the whole run has finished -- call only after [`next_shard`] has
    /// returned `done == true` (see `MinimalExecutor::global_memory_events`'s doc comment).
    #[must_use]
    pub fn global_memory_events(
        &self,
    ) -> (Vec<MemoryInitializeFinalizeEvent>, Vec<MemoryInitializeFinalizeEvent>) {
        self.0.global_memory_events()
    }

    /// A snapshot of opcode/syscall dispatch counts accumulated so far, for summary logging.
    #[must_use]
    pub fn execution_report(&self) -> ExecutionReport {
        self.0.execution_report()
    }
}

/// Drives `driver` through exactly one shard's worth of chunks (however many that takes),
/// returning the spliced shard and whether the whole run is now done. Callers loop this until it
/// returns `done == true`, threading only `&mut ShardDriver` across calls -- the new pipeline's
/// equivalent of `prove.rs`'s existing `execute_state(false)`-in-a-loop shape, just yielding a
/// [`SplicedMinimalTrace`] in place of a serialized `ExecutionState` checkpoint.
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
    let minimal = &mut driver.0;
    let max_syscall_cycles = minimal.max_syscall_cycles();
    let mut chunk = minimal
        .try_execute_chunk()?
        .expect("next_shard must not be called again after a previous call returned done == true");
    let mut splicing = SplicingVM::new(
        &chunk,
        program.clone(),
        max_syscall_cycles,
        element_threshold,
        height_threshold,
    );
    loop {
        match splicing.execute()? {
            SplicingStatus::ShardBoundary => return Ok((splicing.splice(&chunk), false)),
            SplicingStatus::Done => return Ok((splicing.splice(&chunk), true)),
            SplicingStatus::TraceEnd => {
                let carry = splicing.into_carry(&chunk);
                chunk = minimal
                    .try_execute_chunk()?
                    .expect("TraceEnd means the program hasn't halted, so another chunk follows");
                splicing = SplicingVM::resume(carry, &chunk, program.clone(), max_syscall_cycles);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        golden::run_golden,
        hook::HookRegistry,
        programs::tests::{
            ed_decompress_program, fibonacci_program, halt_only_program, hello_world_program,
            hook_fp_inverse_program, secp256r1_add_program, sha_compress_program, simple_program,
        },
        Program,
    };
    use std::collections::BTreeMap;
    use zkm_hypercube::record::MachineRecord;

    /// Fields excluded from the golden count comparison below -- `local_memory_access_events` for
    /// the same reason `tracing.rs`'s own golden harness excludes it (`golden.rs` drives the
    /// legacy `Executor` via its single-pass `run()`, which never populates it regardless of
    /// correctness; see that module's `DEFERRED_FIELDS` doc comment), and `byte_lookups`/
    /// `global_lookup_events`, byproducts of machine-crate `.populate()` calls, not executor-crate
    /// work.
    const DEFERRED_FIELDS: &[&str] = &["local_memory_access_events", "byte_lookups"];

    fn assert_matches_golden(program: impl Fn() -> Program, name: &str) {
        let golden = run_golden(program());
        let records = run_full_pipeline(
            Arc::new(program()),
            u64::MAX / 2,
            Arc::from([]),
            u64::MAX / 2,
            u64::MAX / 2,
        )
        .unwrap();

        assert_eq!(records.len(), 1, "{name}: generous thresholds should keep this to a single shard");
        let last = records.last().unwrap();

        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for record in &records {
            for (chip, count) in record.stats() {
                *counts.entry(chip).or_insert(0) += count;
            }
        }
        let mut golden_counts = golden.event_counts;
        let golden_init_count =
            golden_counts.get("global_memory_initialize_events").copied().unwrap_or(0);
        let golden_finalize_count =
            golden_counts.get("global_memory_finalize_events").copied().unwrap_or(0);
        for field in DEFERRED_FIELDS {
            golden_counts.remove(*field);
            counts.remove(*field);
        }
        assert_eq!(counts, golden_counts, "{name}: event counts mismatch");

        assert_eq!(
            last.global_memory_initialize_events.len(),
            golden_init_count,
            "{name}: global_memory_initialize_events count mismatch"
        );
        assert_eq!(
            last.global_memory_finalize_events.len(),
            golden_finalize_count,
            "{name}: global_memory_finalize_events count mismatch"
        );
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
    fn matches_golden_sha_compress_real_elf() {
        assert_matches_golden(sha_compress_program, "sha_compress_program");
    }

    #[test]
    fn matches_golden_secp256r1_add_real_elf() {
        assert_matches_golden(secp256r1_add_program, "secp256r1_add_program");
    }

    /// Same as `assert_matches_golden` but with tight shard thresholds forcing multiple shards
    /// *and* a tiny `max_trace_size` forcing multiple `MinimalExecutor` chunks -- exercising the
    /// full multi-shard, multi-chunk-per-shard driver loop together, not just each in isolation.
    #[test]
    fn matches_golden_fibonacci_real_elf_many_shards_and_chunks() {
        let golden = run_golden(fibonacci_program());
        let records =
            run_full_pipeline(Arc::new(fibonacci_program()), 64, Arc::from([]), u64::MAX / 2, 40).unwrap();

        assert!(records.len() > 1, "expected a tight height_threshold to force multiple shards");

        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for record in &records {
            for (chip, count) in record.stats() {
                *counts.entry(chip).or_insert(0) += count;
            }
        }
        let mut golden_counts = golden.event_counts;
        let golden_init_count =
            golden_counts.get("global_memory_initialize_events").copied().unwrap_or(0);
        let golden_finalize_count =
            golden_counts.get("global_memory_finalize_events").copied().unwrap_or(0);
        for field in DEFERRED_FIELDS {
            golden_counts.remove(*field);
            counts.remove(*field);
        }
        assert_eq!(counts, golden_counts, "fibonacci_program (many shards/chunks): event counts mismatch");

        let last = records.last().unwrap();
        assert_eq!(
            last.global_memory_initialize_events.len(),
            golden_init_count,
            "fibonacci_program (many shards/chunks): global_memory_initialize_events count mismatch"
        );
        assert_eq!(
            last.global_memory_finalize_events.len(),
            golden_finalize_count,
            "fibonacci_program (many shards/chunks): global_memory_finalize_events count mismatch"
        );
    }

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
