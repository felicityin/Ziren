//! Drives the whole `MinimalExecutor` -> `SplicingVM` -> `TracingVM` pipeline for one program
//! run, producing one [`ExecutionRecord`] per shard. This is the new pipeline's single-threaded,
//! whole-run equivalent of the legacy `Executor`'s two-pass Checkpoint-then-Trace flow (driven by
//! `prove.rs`'s `checkpoint_generator_handle`/`trace_checkpoint` across a channel and a worker
//! pool) -- a stepping stone toward wiring `prove.rs` itself, and independently testable against
//! `golden.rs` without any of that concurrency.

use std::sync::Arc;

use crate::{
    minimal::MinimalExecutor,
    splicing::{SplicingStatus, SplicingVM},
    trace::MinimalTrace,
    tracing::TracingVM,
    ExecutionError, ExecutionRecord, Program,
};

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
    element_threshold: u64,
    height_threshold: u64,
) -> Result<Vec<ExecutionRecord>, ExecutionError> {
    let mut minimal = MinimalExecutor::new(program.clone(), max_trace_size);
    let max_syscall_cycles = minimal.max_syscall_cycles();

    let mut chunk = minimal
        .try_execute_chunk()?
        .expect("the first try_execute_chunk call always retires at least one instruction");
    let mut splicing =
        SplicingVM::new(&chunk, program.clone(), max_syscall_cycles, element_threshold, height_threshold);
    let mut records = Vec::new();
    loop {
        match splicing.execute()? {
            SplicingStatus::ShardBoundary => {
                let spliced = splicing.splice(&chunk);
                records.push(trace_shard(&spliced, program.clone(), max_syscall_cycles)?);
            }
            SplicingStatus::Done => {
                let spliced = splicing.splice(&chunk);
                records.push(trace_shard(&spliced, program.clone(), max_syscall_cycles)?);
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

fn trace_shard<T: MinimalTrace>(
    spliced: &T,
    program: Arc<Program>,
    max_syscall_cycles: u32,
) -> Result<ExecutionRecord, ExecutionError> {
    let mut record = ExecutionRecord::new(program.clone());
    let mut tracing = TracingVM::new(spliced, program, max_syscall_cycles, &mut record);
    tracing.execute()?;
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        golden::run_golden,
        programs::tests::{
            ed_decompress_program, fibonacci_program, halt_only_program, hello_world_program,
            secp256r1_add_program, sha_compress_program, simple_program,
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
        let records =
            run_full_pipeline(Arc::new(program()), u64::MAX / 2, u64::MAX / 2, u64::MAX / 2).unwrap();

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
        let records = run_full_pipeline(Arc::new(fibonacci_program()), 64, u64::MAX / 2, 40).unwrap();

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
}
