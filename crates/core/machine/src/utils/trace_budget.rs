use std::sync::OnceLock;

use zkm_hypercube::prover::{ProverPermit, ProverSemaphore};
use zkm_stark::total_system_memory_bytes;

/// Size, in bytes, of one weighted permit in the process-wide trace-memory budget semaphore.
/// Coarse enough that a typical shard's estimated size doesn't need an unwieldy number of
/// permits, fine enough that small shards don't all round up to the same handful of permits.
const TRACE_BUDGET_UNIT_BYTES: u64 = 64 * 1024 * 1024;

/// Fraction of total system RAM the trace-memory budget defaults to, absent
/// `ZKM_TRACE_MEMORY_BUDGET_MB`. Conservative: this budget only gates padded main-trace data
/// (`crates/hypercube/src/prover/trace.rs`'s `Traces`), not the executor's own memory
/// (checkpoints, in-flight `ExecutionRecord`s) or anything else resident in the process.
const DEFAULT_TRACE_BUDGET_RAM_FRACTION: u64 = 2;

struct TraceMemoryBudget {
    semaphore: ProverSemaphore,
    total_permits: u32,
}

static TRACE_MEMORY_BUDGET: OnceLock<TraceMemoryBudget> = OnceLock::new();

fn budget() -> &'static TraceMemoryBudget {
    TRACE_MEMORY_BUDGET.get_or_init(|| {
        let budget_bytes = std::env::var("ZKM_TRACE_MEMORY_BUDGET_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or_else(|| total_system_memory_bytes() / DEFAULT_TRACE_BUDGET_RAM_FRACTION);
        let total_permits = (budget_bytes / TRACE_BUDGET_UNIT_BYTES).max(1) as u32;
        TraceMemoryBudget { semaphore: ProverSemaphore::new(total_permits as usize), total_permits }
    })
}

/// Acquire a byte-weighted permit from the process-wide trace-memory budget, admitting a shard
/// whose estimated materialized main-trace size is `estimated_bytes`
/// (`zkm_core_executor::cost::estimate_record_trace_bytes`). This is what bounds how many
/// shards' padded main traces can be resident at once across every concurrent caller of
/// `prove_with_context` in this process, independent of `trace_gen_workers`/`prove_workers`
/// thread counts.
///
/// The requested permit count is clamped to the budget's total, so a single
/// oversized/precompile-heavy shard can never deadlock the pipeline by requesting more permits
/// than exist -- it consumes the whole budget by itself instead, serializing behind (or ahead
/// of) any other concurrent shard.
pub async fn acquire_trace_budget(estimated_bytes: u64) -> ProverPermit {
    let b = budget();
    let n = estimated_bytes.div_ceil(TRACE_BUDGET_UNIT_BYTES).clamp(1, u64::from(b.total_permits)) as u32;
    b.semaphore.clone().acquire_many(n).await.expect("trace memory budget semaphore closed")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap().block_on(fut)
    }

    #[test]
    fn oversized_estimate_clamps_instead_of_deadlocking() {
        // However large the real budget is, an absurdly large estimate must still be admitted
        // (clamped to the budget's total permits) rather than hang forever waiting for more
        // permits than the semaphore was ever constructed with.
        let permit = block_on(acquire_trace_budget(u64::MAX));
        drop(permit);
    }

    #[test]
    fn small_estimates_can_run_concurrently() {
        let a = block_on(acquire_trace_budget(TRACE_BUDGET_UNIT_BYTES));
        let b = block_on(acquire_trace_budget(TRACE_BUDGET_UNIT_BYTES));
        drop(a);
        drop(b);
    }
}
