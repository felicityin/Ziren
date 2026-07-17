use std::env;

use serde::{Deserialize, Serialize};
use sysinfo::System;

/// The core machine's PCS/FRI row-count parameter (`ShardVerifier::from_basefold_parameters`'s
/// `max_log_row_count`), fixed independently of `shard_size` (the executor's cycle-count
/// ceiling, `ZKMCoreOpts::shard_size`). `shard_size` can be raised well past
/// `1 << CORE_MAX_LOG_ROW_COUNT` (the executor's own per-chip/`clk` guards,
/// `crates/core/executor/src/executor.rs`, keep any individual chip's real row count under this
/// ceiling regardless); `max_log_row_count` cannot, because of KoalaBear's two-adicity: the
/// basefold verifier (`slop/crates/basefold/src/verifier.rs`) requires `log_stacking_height +
/// log_blowup <= KoalaBear::TWO_ADICITY (24)`, where `log_stacking_height =
/// max_log_row_count - 1` (`stacking_height_for`) and `log_blowup = 2` (`DEFAULT_LOG_BLOWUP`) --
/// so `(22 - 1) + 2 = 23 <= 24` holds with 1 bit of headroom; `max_log_row_count = 24` would need
/// `25 <= 24`, a guaranteed `TwoAdicityOverflow`.
pub const CORE_MAX_LOG_ROW_COUNT: usize = 22;

// A recursion shard's compress-machine ExtAlu chip can exceed 1 << 21 real rows even for the
// smallest guest program, once core's own shard_size scales up via `get_memory_opts`. Unlike
// core's own shard_size, this bound isn't memory-scaled -- it's a fixed cap sized for the
// largest actual core shard_size in play.
const RECURSION_MAX_SHARD_SIZE: usize = 1 << 22;
const MAX_SHARD_BATCH_SIZE: usize = 8;
// `trace_gen_workers` gates both the number of concurrent phase-2 trace-gen worker threads
// (`crates/core/machine/src/utils/prove.rs`'s `for _ in 0..opts.trace_gen_workers`) and how many
// shards' main traces can be held in memory / proved at once (the
// `ProverSemaphore::new(opts.trace_gen_workers.max(1))` permit pool). Kept conservative rather
// than scaling to all cores, to leave headroom for each worker's own internal rayon parallelism
// and limit how many shards' trace data are held concurrently.
const DEFAULT_TRACE_GEN_WORKERS: usize = 4;
// How many shards can be proved concurrently by the phase-2 prover (see
// `crates/core/machine/src/utils/prove.rs`'s `p2_prover_handles` loop). Kept separate from
// `trace_gen_workers` since the two workloads have different resource profiles: trace generation
// is lighter/more memory-bound, while shard proving (`commit_traces`'s FFT/Merkle-tree work) is
// heavier/more CPU-bound.
const DEFAULT_PROVE_WORKERS: usize = 1;
const DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY: usize = 128;
const DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY: usize = 1;

/// The threshold for splitting deferred events.
pub const MAX_DEFERRED_SPLIT_THRESHOLD: usize = 1 << 15;

/// The default maximum estimated trace area (in bytes, via
/// `zkm_core_executor::cost::estimate_mips_lde_size`) before a shard is stopped early.
///
/// This is a correctness bound, not just an OOM guard: the jagged PCS rejects a proof with
/// `AreaOutOfBounds` once a shard's combined preprocessed-plus-main padded cell count (row count
/// times column count, summed across every committed chip in both rounds) reaches `2^29` -- see
/// `slop_jagged::verifier::JaggedPcsVerifier::verify_trusted_evaluations`'s `log_m >= 30` check,
/// where `log_m` is `ceil(log2(total padded cell count))`. `estimate_mips_lde_size` folds
/// `preprocessed_width + main_width` per chip into one combined cell count matching that same
/// total; at `cells * 8` bytes, the ceiling is `2^32` bytes (4 GiB). The threshold here is
/// `7 * 2^29` bytes (3.5 GiB, 12.5% margin below that ceiling).
pub const DEFAULT_LDE_SIZE_THRESHOLD: u64 = 7 * (1 << 29);

/// Options to configure the Ziren prover for core and recursive proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZKMProverOpts {
    /// Options for the core prover.
    pub core_opts: ZKMCoreOpts,
    /// Options for the recursion prover.
    pub recursion_opts: ZKMCoreOpts,
}

impl Default for ZKMProverOpts {
    fn default() -> Self {
        Self { core_opts: ZKMCoreOpts::default(), recursion_opts: ZKMCoreOpts::recursion() }
    }
}

impl ZKMProverOpts {
    /// Get the default prover options.
    #[must_use]
    pub fn auto() -> Self {
        let cpu_ram_gb = System::new_all().total_memory() / (1024 * 1024 * 1024);
        ZKMProverOpts::cpu(cpu_ram_gb as usize)
    }

    /// Get the memory options (shard size, shard batch size, and divisor) for a prover on CPU based
    /// on the amount of CPU memory.
    #[must_use]
    fn get_memory_opts(cpu_ram_gb: usize) -> (usize, usize, usize) {
        match cpu_ram_gb {
            0..33 => (19, 1, 3),
            33..49 => (20, 1, 2),
            49..65 => (21, 1, 3),
            65..81 => (21, 3, 1),
            // `shard_size` can reach `1 << CORE_MAX_LOG_ROW_COUNT` safely: the executor's own
            // per-chip height ceiling and `clk`-overflow guard
            // (`crates/core/executor/src/executor.rs`) keep every chip's real row count under
            // that same bound regardless.
            81.. => (22, 4, 1),
        }
    }

    /// Get the default prover options for a prover on CPU based on the amount of CPU memory.
    ///
    /// We use a soft heuristic based on our understanding of the memory usage in the GPU prover.
    #[must_use]
    pub fn cpu(cpu_ram_gb: usize) -> Self {
        let (log2_shard_size, shard_batch_size, log2_divisor) = Self::get_memory_opts(cpu_ram_gb);

        let mut opts = ZKMProverOpts::default();
        opts.core_opts.shard_size = 1 << log2_shard_size;
        opts.core_opts.shard_batch_size = shard_batch_size;

        opts.core_opts.records_and_traces_channel_capacity = 1;
        // Was 1: see DEFAULT_TRACE_GEN_WORKERS's doc comment -- this explicit override was
        // stomping the (also-1) default down to fully sequential shard proving regardless of
        // available cores.
        opts.core_opts.trace_gen_workers = DEFAULT_TRACE_GEN_WORKERS;
        opts.core_opts.prove_workers = DEFAULT_PROVE_WORKERS;

        let divisor = 1 << log2_divisor;
        opts.core_opts.split_opts.deferred /= divisor;
        opts.core_opts.split_opts.keccak /= divisor;
        opts.core_opts.split_opts.sha_extend /= divisor;
        opts.core_opts.split_opts.sha_compress /= divisor;
        opts.core_opts.split_opts.memory /= divisor;

        opts.recursion_opts.shard_batch_size = 2;
        opts.recursion_opts.records_and_traces_channel_capacity = 1;
        opts.recursion_opts.trace_gen_workers = 1;
        opts.recursion_opts.prove_workers = 1;

        opts
    }

    /// Get the default prover options for a prover on GPU given the amount of CPU and GPU memory.
    #[must_use]
    pub fn gpu(_cpu_ram_gb: usize, gpu_ram_gb: usize) -> Self {
        let mut opts = ZKMProverOpts::default();

        // Set the core options.
        if 24 <= gpu_ram_gb {
            //let log2_shard_size = 21;
            //opts.core_opts.shard_size = 1 << log2_shard_size;
            opts.core_opts.shard_batch_size = 1;

            //let log2_deferred_threshold = 14;
            //opts.core_opts.split_opts = SplitOpts::new(1 << log2_deferred_threshold);

            //opts.core_opts.records_and_traces_channel_capacity = 4;
            //opts.core_opts.trace_gen_workers = 4;

            //if cpu_ram_gb <= 20 {
            //    opts.core_opts.records_and_traces_channel_capacity = 1;
            //    opts.core_opts.trace_gen_workers = 2;
            //}
        } else {
            unreachable!("not enough gpu memory");
        }

        // Set the recursion options.
        opts.recursion_opts.shard_batch_size = 1;

        opts
    }
}

/// Options for the core prover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZKMCoreOpts {
    /// The size of a shard in terms of cycles.
    pub shard_size: usize,
    /// The size of a batch of shards in terms of cycles.
    pub shard_batch_size: usize,
    /// Options for splitting deferred events.
    pub split_opts: SplitOpts,
    /// Whether to reconstruct the commitments.
    pub reconstruct_commitments: bool,
    /// The number of workers to use for generating traces.
    pub trace_gen_workers: usize,
    /// The number of shards that can be proved concurrently by the phase-2 prover.
    pub prove_workers: usize,
    /// The capacity of the channel for checkpoints.
    pub checkpoints_channel_capacity: usize,
    /// The capacity of the channel for records and traces.
    pub records_and_traces_channel_capacity: usize,
    /// The frequency for shape checks.
    pub shape_check_frequency: u64,
    /// The maximum estimated LDE size (in bytes) before a shard is stopped early to avoid OOM.
    pub lde_size_threshold: u64,
}

impl Default for ZKMCoreOpts {
    fn default() -> Self {
        let cpu_ram_gb = System::new_all().total_memory() / (1024 * 1024 * 1024);
        let (default_log2_shard_size, default_shard_batch_size, default_log2_divisor) =
            ZKMProverOpts::get_memory_opts(cpu_ram_gb as usize);

        let mut opts = Self {
            shard_size: env::var("SHARD_SIZE").map_or_else(
                |_| 1 << default_log2_shard_size,
                |s| s.parse::<usize>().unwrap_or(1 << default_log2_shard_size),
            ),
            shard_batch_size: env::var("SHARD_BATCH_SIZE").map_or_else(
                |_| default_shard_batch_size,
                |s| s.parse::<usize>().unwrap_or(default_shard_batch_size),
            ),
            split_opts: SplitOpts::new(MAX_DEFERRED_SPLIT_THRESHOLD),
            trace_gen_workers: env::var("TRACE_GEN_WORKERS").map_or_else(
                |_| DEFAULT_TRACE_GEN_WORKERS,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_TRACE_GEN_WORKERS),
            ),
            prove_workers: env::var("PROVE_WORKERS").map_or_else(
                |_| DEFAULT_PROVE_WORKERS,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_PROVE_WORKERS),
            ),
            checkpoints_channel_capacity: env::var("CHECKPOINTS_CHANNEL_CAPACITY").map_or_else(
                |_| DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY),
            ),
            records_and_traces_channel_capacity: env::var("RECORDS_AND_TRACES_CHANNEL_CAPACITY")
                .map_or_else(
                    |_| DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY,
                    |s| s.parse::<usize>().unwrap_or(DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY),
                ),
            shape_check_frequency: env::var("SHAPE_CHECK_FREQUENCY")
                .map_or_else(|_| 16, |s| s.parse::<u64>().unwrap_or(16)),
            lde_size_threshold: env::var("LDE_SIZE_THRESHOLD").map_or_else(
                |_| DEFAULT_LDE_SIZE_THRESHOLD,
                |s| s.parse::<u64>().unwrap_or(DEFAULT_LDE_SIZE_THRESHOLD),
            ),
            reconstruct_commitments: true,
        };

        tracing::info!(
            "shard_size: {:?}, shard_batch_size: {:?}",
            opts.shard_size,
            opts.shard_batch_size,
        );

        let divisor = 1 << default_log2_divisor;
        opts.split_opts.deferred /= divisor;
        opts.split_opts.keccak /= divisor;
        opts.split_opts.sha_extend /= divisor;
        opts.split_opts.sha_compress /= divisor;
        opts.split_opts.memory /= divisor;

        opts
    }
}

impl ZKMCoreOpts {
    /// Get the default options for the recursion prover.
    #[must_use]
    pub fn recursion() -> Self {
        let mut opts = Self::max();
        opts.reconstruct_commitments = false;
        opts.shard_size = RECURSION_MAX_SHARD_SIZE;
        opts.shard_batch_size = 2;
        opts
    }

    /// Get the maximum options for the core prover.
    #[must_use]
    pub fn max() -> Self {
        let split_threshold = env::var("SPLIT_THRESHOLD")
            .map(|s| s.parse::<usize>().unwrap_or(MAX_DEFERRED_SPLIT_THRESHOLD))
            .unwrap_or(MAX_DEFERRED_SPLIT_THRESHOLD)
            .max(MAX_DEFERRED_SPLIT_THRESHOLD);

        let shard_size = env::var("SHARD_SIZE").map_or_else(
            |_| RECURSION_MAX_SHARD_SIZE,
            |s| s.parse::<usize>().unwrap_or(RECURSION_MAX_SHARD_SIZE),
        );

        Self {
            shard_size,
            shard_batch_size: env::var("SHARD_BATCH_SIZE").map_or_else(
                |_| MAX_SHARD_BATCH_SIZE,
                |s| s.parse::<usize>().unwrap_or(MAX_SHARD_BATCH_SIZE),
            ),
            split_opts: SplitOpts::new(split_threshold),
            trace_gen_workers: env::var("TRACE_GEN_WORKERS").map_or_else(
                |_| DEFAULT_TRACE_GEN_WORKERS,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_TRACE_GEN_WORKERS),
            ),
            prove_workers: env::var("PROVE_WORKERS").map_or_else(
                |_| DEFAULT_PROVE_WORKERS,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_PROVE_WORKERS),
            ),
            checkpoints_channel_capacity: env::var("CHECKPOINTS_CHANNEL_CAPACITY").map_or_else(
                |_| DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY,
                |s| s.parse::<usize>().unwrap_or(DEFAULT_CHECKPOINTS_CHANNEL_CAPACITY),
            ),
            records_and_traces_channel_capacity: env::var("RECORDS_AND_TRACES_CHANNEL_CAPACITY")
                .map_or_else(
                    |_| DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY,
                    |s| s.parse::<usize>().unwrap_or(DEFAULT_RECORDS_AND_TRACES_CHANNEL_CAPACITY),
                ),
            shape_check_frequency: env::var("SHAPE_CHECK_FREQUENCY")
                .map_or_else(|_| 16, |s| s.parse::<u64>().unwrap_or(16)),
            lde_size_threshold: env::var("LDE_SIZE_THRESHOLD").map_or_else(
                |_| DEFAULT_LDE_SIZE_THRESHOLD,
                |s| s.parse::<u64>().unwrap_or(DEFAULT_LDE_SIZE_THRESHOLD),
            ),
            reconstruct_commitments: true,
        }
    }
}

/// Options for splitting deferred events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitOpts {
    /// The threshold for default events.
    pub deferred: usize,
    /// The threshold for keccak events.
    pub keccak: usize,
    /// The threshold for sha extend events.
    pub sha_extend: usize,
    /// The threshold for sha compress events.
    pub sha_compress: usize,
    /// The threshold for memory events.
    pub memory: usize,
    /// The threshold for combining the memory init/finalize events in to the current shard in
    /// terms of cycles.
    pub combine_memory_threshold: usize,
}

impl SplitOpts {
    /// Create a new [`SplitOpts`] with the given threshold.
    #[must_use]
    pub fn new(deferred_split_threshold: usize) -> Self {
        Self {
            deferred: deferred_split_threshold,
            keccak: 8 * deferred_split_threshold / 24,
            sha_extend: 32 * deferred_split_threshold / 48,
            sha_compress: 32 * deferred_split_threshold / 80,
            memory: 64 * deferred_split_threshold,
            combine_memory_threshold: 1 << 17,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::print_stdout)]

    use super::*;

    #[test]
    fn test_opts() {
        let opts = ZKMProverOpts::cpu(8);
        println!("8: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(15);
        println!("15: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(16);
        println!("16: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(32);
        println!("32: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(36);
        println!("36: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(64);
        println!("64: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(128);
        println!("128: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(256);
        println!("256: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::cpu(512);
        println!("512: {:?}", opts.core_opts);

        let opts = ZKMProverOpts::auto();
        println!("auto: {:?}", opts.core_opts);
    }
}
