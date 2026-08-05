//! A differential-testing golden-snapshot harness.
//!
//! Runs the legacy, monolithic `Executor` end-to-end (`Trace` mode, via `Executor::run`) and
//! summarizes its output into a [`GoldenSnapshot`]. The `MinimalExecutor` -> `SplicingVM` ->
//! `TracingVM` pipeline must reproduce a `GoldenSnapshot` produced here exactly (event counts,
//! final registers, memory digest, committed public-values stream) even though shard
//! *boundaries* may differ between the two pipelines -- this harness intentionally does not
//! expose per-shard data, only whole-run totals, so it can't be misused to assert boundary
//! equality by accident.

use std::{
    collections::BTreeMap,
    hash::{DefaultHasher, Hash, Hasher},
};

use zkm_hypercube::record::MachineRecord;
use zkm_stark::ZKMCoreOpts;

use crate::{register::NUM_REGISTERS, Executor, Program};

/// A summary of a completed `Executor` run, deliberately whole-run (not per-shard) -- see the
/// module doc for why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GoldenSnapshot {
    /// Per-chip event counts (`ExecutionRecord::stats()`, keyed by field name), summed across
    /// every shard the run produced.
    pub event_counts: BTreeMap<String, usize>,
    /// Number of shards (`ExecutionRecord`s) the run produced. Not asserted equal against the new
    /// pipeline (shard boundaries may legitimately differ, see the module doc) -- kept only for
    /// diagnostics.
    pub num_shards: usize,
    /// Final register file, in register-index order (0..36: 32 GPRs, LO, HI, BRK, HEAP).
    pub final_registers: [u32; NUM_REGISTERS],
    /// Final `pc`.
    pub final_pc: u32,
    /// Final global `clk`.
    pub final_clk: u64,
    /// The public-values byte stream the program committed via `COMMIT`/`COMMIT_DEFERRED_PROOFS`.
    pub public_values_stream: Vec<u8>,
    /// Order-independent digest over every touched `(addr, value, timestamp)` triple in the final
    /// memory image (registers + page table), sorted by address first for determinism. A stand-in
    /// for comparing full memory contents without keeping the whole image around.
    pub memory_digest: u64,
}

/// Runs `program` to completion on the legacy `Executor` and summarizes the result.
///
/// # Panics
///
/// Panics if the program fails to execute (this is a test-only reference harness; a failing
/// golden-reference run is itself a bug worth stopping on immediately).
pub(crate) fn run_golden(program: Program) -> GoldenSnapshot {
    run_golden_with_stdin(program, &[])
}

/// Like [`run_golden`], but seeds the legacy `Executor`'s `input_stream` first -- for programs
/// that read via `HINT_LEN`/`HINT_READ`.
///
/// # Panics
///
/// Panics if the program fails to execute (this is a test-only reference harness; a failing
/// golden-reference run is itself a bug worth stopping on immediately).
pub(crate) fn run_golden_with_stdin(program: Program, stdin: &[Vec<u8>]) -> GoldenSnapshot {
    let mut runtime = Executor::new(program, ZKMCoreOpts::default());
    runtime.write_vecs(stdin);
    runtime.run().expect("golden-reference Executor run failed");
    snapshot(&mut runtime)
}

fn snapshot(runtime: &mut Executor) -> GoldenSnapshot {
    let mut event_counts: BTreeMap<String, usize> = BTreeMap::new();
    for record in &runtime.records {
        for (chip, count) in record.stats() {
            *event_counts.entry(chip).or_insert(0) += count;
        }
    }

    let final_registers = runtime.registers();
    let final_pc = runtime.state.pc;
    let final_clk = runtime.state.clk;
    let public_values_stream = runtime.state.public_values_stream.clone();

    // Page-table (RAM) only, deliberately excluding registers: the new pipeline never tracks
    // register *timestamps* (registers are always live-recomputed, never oracle-logged) --
    // register *values* are already covered separately by `final_registers` above. Including
    // register timestamps here would make this digest uncomparable against the new pipeline's
    // output for no real coverage gain.
    let mut touched: Vec<(u32, u32, u64)> = runtime
        .state
        .memory
        .page_table
        .clone()
        .into_iter()
        .filter(|(_, record)| record.timestamp != 0)
        .map(|(addr, record)| (addr, record.value, record.timestamp))
        .collect();
    touched.sort_unstable_by_key(|&(addr, _, _)| addr);
    let mut hasher = DefaultHasher::new();
    touched.hash(&mut hasher);
    let memory_digest = hasher.finish();

    GoldenSnapshot {
        num_shards: runtime.records.len(),
        event_counts,
        final_registers,
        final_pc,
        final_clk,
        public_values_stream,
        memory_digest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::programs::tests::{
        fibonacci_program, halt_only_program, hello_world_program, secp256r1_add_program,
        secp256r1_double_program, sha3_chain_program, simple_program, ssz_withdrawals_program,
        u256xu2048_mul_program,
    };

    /// The harness itself must be deterministic -- two independent runs of the same program must
    /// produce byte-identical snapshots. Differential tests built against this harness rely on
    /// that holding; if the *reference* Executor weren't deterministic, nothing built against it
    /// could ever be validated.
    fn assert_deterministic(program: impl Fn() -> Program) {
        let a = run_golden(program());
        let b = run_golden(program());
        assert_eq!(a, b, "golden harness produced two different snapshots for the same program");
    }

    #[test]
    fn golden_simple_program_is_deterministic_and_sane() {
        let snap = run_golden(simple_program());
        assert!(snap.num_shards >= 1);
        // `simple_program` never retires a HALT syscall, so it runs off the end of the program --
        // it still must have executed the three ADD instructions (registers 29/30/31 get real
        // values, mirroring `test_simple_program_run`'s own `RA == 42` assertion).
        assert_eq!(snap.final_registers[31], 42);
        assert_deterministic(simple_program);
    }

    #[test]
    fn golden_halt_only_program_is_deterministic_and_sane() {
        let snap = run_golden(halt_only_program());
        assert!(snap.num_shards >= 1);
        assert_deterministic(halt_only_program);
    }

    #[test]
    fn golden_fibonacci_real_elf_is_deterministic_and_sane() {
        let snap = run_golden(fibonacci_program());
        assert!(snap.num_shards >= 1);
        assert_deterministic(fibonacci_program);
    }

    #[test]
    fn golden_hello_world_real_elf_is_deterministic_and_sane() {
        let snap = run_golden(hello_world_program());
        assert!(snap.num_shards >= 1);
        assert_deterministic(hello_world_program);
    }

    #[test]
    fn golden_sha3_chain_real_elf_is_deterministic_and_sane() {
        let snap = run_golden(sha3_chain_program());
        assert!(snap.num_shards >= 1);
        assert_deterministic(sha3_chain_program);
    }

    #[test]
    fn golden_secp256r1_add_real_elf_is_deterministic_and_sane() {
        let snap = run_golden(secp256r1_add_program());
        assert!(snap.num_shards >= 1);
        assert!(
            snap.event_counts.contains_key("syscall SECP256R1_ADD"),
            "expected the SECP256R1_ADD precompile to actually fire: {:?}",
            snap.event_counts
        );
        assert_deterministic(secp256r1_add_program);
    }

    #[test]
    fn golden_secp256r1_double_real_elf_is_deterministic_and_sane() {
        let snap = run_golden(secp256r1_double_program());
        assert!(snap.num_shards >= 1);
        assert_deterministic(secp256r1_double_program);
    }

    #[test]
    fn golden_u256xu2048_mul_real_elf_is_deterministic_and_sane() {
        let snap = run_golden(u256xu2048_mul_program());
        assert!(snap.num_shards >= 1);
        assert_deterministic(u256xu2048_mul_program);
    }

    #[test]
    fn golden_ssz_withdrawals_keccak_real_elf_is_deterministic_and_sane() {
        let snap = run_golden(ssz_withdrawals_program());
        assert!(snap.num_shards >= 1);
        assert_deterministic(ssz_withdrawals_program);
    }
}
