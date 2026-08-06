mod air;
mod columns;
mod constants;
mod trace;
mod utils;

pub const KECCAK_GENERAL_RATE_U32S: usize = 36;
pub const KECCAK_STATE_U32S: usize = 50;
pub const KECCAK_GENERAL_OUTPUT_U32S: usize = 16;
pub const BITS_PER_LIMB: usize = 64 / p3_keccak_air::U64_LIMBS;

#[derive(Default)]
pub struct KeccakSpongeChip;

impl KeccakSpongeChip {
    pub const fn new() -> Self {
        Self {}
    }
}
#[cfg(test)]
pub mod sponge_tests {
    use crate::utils::{run_test, setup_logger};
    use test_artifacts::KECCAK_SPONGE_ELF;
    use zkm_core_executor::Program;
    #[test]
    fn test_keccak_sponge_program_prove() {
        setup_logger();
        let program = Program::from(KECCAK_SPONGE_ELF).unwrap();
        run_test(program).unwrap();
    }

    /// Builds a real, deferred, honestly-split KeccakSponge-only shard, mirroring the real
    /// pipeline's `record.defer()` + `deferred.split(..)` (see `prove_with_context` in
    /// `crates/core/machine/src/utils/prove.rs`). `SyscallChip`'s `Precompile`-kind `included()`
    /// requires `cpu_events`/`global_memory_{initialize,finalize}_events` to *all* be empty, so
    /// the raw CPU-and-precompile-events-together record `run_full_pipeline` also produces won't
    /// do.
    fn keccak_sponge_only_record(
        program: &zkm_core_executor::Program,
    ) -> zkm_core_executor::ExecutionRecord {
        keccak_program_records(program).1
    }

    /// Like [`keccak_sponge_only_record`], but returns the CPU-bearing record too. `run_full_pipeline`
    /// mirrors `prove_with_context`'s Stage A/B execution and trace generation, but (like the raw
    /// per-checkpoint record `Executor::run()` used to produce) does *not* itself defer/split --
    /// that's `prove_with_context`'s own additional step -- so this still does it by hand.
    fn keccak_program_records(
        program: &zkm_core_executor::Program,
    ) -> (zkm_core_executor::ExecutionRecord, zkm_core_executor::ExecutionRecord) {
        use std::sync::Arc;
        use zkm_core_executor::{run_full_pipeline, syscalls::SyscallCode};
        use zkm_stark::ZKMCoreOpts;

        let mut records = run_full_pipeline(
            Arc::new(program.clone()),
            u64::MAX / 2,
            Arc::from([]),
            u64::MAX / 2,
            u64::MAX / 2,
        )
        .unwrap();
        assert_eq!(records.len(), 1, "expected this small test program to fit in one checkpoint");
        let mut cpu_record = records.remove(0);

        // `run_full_pipeline` doesn't back-fill `execution_shard`/`is_execution_shard`/`start_pc`/
        // `next_pc`/`initial_timestamp`/`last_timestamp` into `public_values` either (that's
        // `prove_with_context`'s job, unlike legacy `Executor::execute`, which back-filled all but
        // the clk fields automatically when it flushed a record). `is_execution_shard` in
        // particular gates whether `EvalPublicValues` even emits its own end of the CPU chip's
        // `LookupKind::State` chain boundary in `eval_public_values` -- left at its `0` default,
        // `SyscallInstrs`' own real state-chain token goes unmatched.
        cpu_record.public_values.execution_shard = 1;
        cpu_record.public_values.is_execution_shard = cpu_record.contains_cpu() as u32;
        cpu_record.public_values.start_pc = cpu_record.first_instruction_pc.unwrap();
        cpu_record.public_values.next_pc = cpu_record.last_next_pc;
        let first_clk = cpu_record.first_instruction_clk.unwrap();
        cpu_record.public_values.initial_clk_high = (first_clk >> 24) as u32;
        cpu_record.public_values.initial_clk_low = (first_clk & 0xffffff) as u32;
        let last_clk_high = cpu_record.last_instruction_clk >> 24;
        cpu_record.public_values.last_clk_high = last_clk_high as u32;
        cpu_record.public_values.last_clk_low =
            (cpu_record.last_timestamp - (last_clk_high << 24)) as u32;
        let mut deferred = cpu_record.defer();
        assert!(cpu_record.contains_cpu());
        assert!(cpu_record.precompile_events.is_empty());

        let split_records = deferred.split(true, None, ZKMCoreOpts::default().split_opts);
        let keccak_record = split_records
            .into_iter()
            .find(|r| !r.get_precompile_events(SyscallCode::KECCAK_SPONGE).is_empty())
            .expect("expected a deferred, split record with KeccakSponge precompile events");
        assert!(keccak_record.cpu_events.is_empty());
        assert!(keccak_record.global_memory_initialize_events.is_empty());
        assert!(keccak_record.global_memory_finalize_events.is_empty());
        (cpu_record, keccak_record)
    }

    /// Nets every chip's send/receive multiplicities (Local scope) for a real shard's chosen
    /// cluster against its own real generated traces, returning whether they balance. `None` in
    /// `chip_names` checks every machine chip (needed for a CPU-bearing shard, whose cluster is
    /// large); `Some(names)` restricts to a specific cluster (e.g. a small precompile-only one).
    fn assert_local_interactions_balance(
        program: &zkm_core_executor::Program,
        mut record: zkm_core_executor::ExecutionRecord,
        chip_names: Option<&[&str]>,
    ) {
        use crate::mips::MipsAir;
        use p3_koala_bear::KoalaBear;
        use slop_air::BaseAir;
        use slop_multilinear::{Mle, PaddedMle};
        use std::sync::Arc;
        use zkm_hypercube::{
            air::{LookupScope, MachineAir},
            lookup::{debug_interactions_with_all_chips, LookupKind},
            prover::Traces,
            record::MachineRecord,
        };

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        machine.generate_dependencies(std::iter::once(&mut record), None).unwrap();

        let chips = match chip_names {
            Some(names) => {
                let chips = machine
                    .chips()
                    .iter()
                    .filter(|c| names.contains(&MachineAir::<KoalaBear>::name(*c).as_str()))
                    .cloned()
                    .collect::<Vec<_>>();
                assert_eq!(chips.len(), names.len(), "missing a chip by name");
                chips
            }
            None => machine.chips().to_vec(),
        };

        let max_log_row_count = 20u32;
        let mut preprocessed_named = std::collections::BTreeMap::new();
        let mut main_named = std::collections::BTreeMap::new();
        for chip in &chips {
            let name = MachineAir::<KoalaBear>::name(chip);
            let pre_mle = match chip.generate_preprocessed_trace(program) {
                Some(t) => PaddedMle::padded_with_zeros(Arc::new(Mle::from(t)), max_log_row_count),
                None => PaddedMle::zeros(0, max_log_row_count),
            };
            preprocessed_named.insert(name.clone(), pre_mle);

            let main_mle = if chip.included(&record) {
                let trace = chip.generate_trace(&record, &mut Default::default()).unwrap();
                PaddedMle::padded_with_zeros(Arc::new(Mle::from(trace)), max_log_row_count)
            } else {
                PaddedMle::zeros(chip.width(), max_log_row_count)
            };
            main_named.insert(name, main_mle);
        }
        let preprocessed_traces = Traces { named_traces: preprocessed_named };
        let traces = Traces { named_traces: main_named };
        let public_values = record.public_values::<KoalaBear>();

        assert!(
            debug_interactions_with_all_chips(
                &chips,
                &preprocessed_traces,
                &traces,
                public_values,
                LookupKind::all_kinds(),
                LookupScope::Local,
            ),
            "local-scope send/receive interactions don't balance"
        );
    }

    /// Nets every chip's send/receive multiplicities for a real, deferred, honestly-split
    /// KeccakSponge-only shard, guarding against a lookup-level regression (an AIR `send`/
    /// `receive` call with no matching counterpart) independent of whether it happens to also
    /// break end-to-end proving (see `test_keccak_sponge_program_prove`).
    #[test]
    fn test_keccak_sponge_local_interactions_balance() {
        setup_logger();
        let program = Program::from(KECCAK_SPONGE_ELF).unwrap();
        let record = keccak_sponge_only_record(&program);
        assert_local_interactions_balance(
            &program,
            record,
            Some(&["Program", "Byte", "Global", "SyscallPrecompile", "MemoryLocal", "KeccakSponge"]),
        );
    }

    /// Same as `test_keccak_sponge_local_interactions_balance`, but for the CPU-bearing shard of
    /// the same program (dispatching the KECCAK_SPONGE syscall, among ordinary CPU execution) --
    /// this is the shard that `test_keccak_sponge_program_prove` actually fails on (confirmed via
    /// temporary instrumentation on `LogupGkrVerifier::verify_logup_gkr`'s `CumulativeSumMismatch`
    /// branch: `shard_chips` was the CPU cluster, not KeccakSponge's), so unlike the sibling test
    /// above, this one is expected to catch task #50's real bug once identified.
    #[test]
    fn test_keccak_program_cpu_shard_local_interactions_balance() {
        setup_logger();
        let program = Program::from(KECCAK_SPONGE_ELF).unwrap();
        let (cpu_record, _) = keccak_program_records(&program);
        assert_local_interactions_balance(&program, cpu_record, None);
    }

    /// Evaluates `KeccakSpongeChip::eval` against each row of a real, honestly-generated trace
    /// individually (task #50's toolkit technique #2), to check for a row-local AIR constraint
    /// violation now that interaction/lookup balance is confirmed clean (see
    /// `test_keccak_sponge_local_interactions_balance`).
    #[test]
    fn test_keccak_sponge_row_constraints() {
        use crate::syscall::precompiles::keccak_sponge::{
            columns::NUM_KECCAK_SPONGE_COLS, KeccakSpongeChip,
        };
        use p3_air::Air;
        use p3_koala_bear::KoalaBear;
        use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
        use p3_matrix::Matrix;
        use zkm_hypercube::air::MachineAir;
        use zkm_hypercube::debug::DebugConstraintBuilder;
        use zkm_hypercube::record::MachineRecord;

        setup_logger();
        let program = Program::from(KECCAK_SPONGE_ELF).unwrap();
        let record = keccak_sponge_only_record(&program);

        let chip = KeccakSpongeChip::new();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&record, &mut Default::default()).unwrap();
        let public_values = MachineRecord::public_values::<KoalaBear>(&record);
        let empty_preprocessed = RowMajorMatrixView::new(&[], 0);

        let mut total_failures = 0;
        for row in 0..trace.height() {
            let row_data = trace.row_slice(row).to_vec();
            let main = RowMajorMatrixView::new(&row_data, NUM_KECCAK_SPONGE_COLS);
            let mut builder = DebugConstraintBuilder::<KoalaBear, KoalaBear> {
                preprocessed: empty_preprocessed,
                main,
                public_values: &public_values,
                failing_constraints: Vec::new(),
                num_constraints_evaluated: 0,
                phantom: std::marker::PhantomData,
            };
            chip.eval(&mut builder);
            if !builder.failing_constraints.is_empty() {
                eprintln!(
                    "row {row}: {} failing constraint(s) at indices {:?} (of {} evaluated)",
                    builder.failing_constraints.len(),
                    builder.failing_constraints,
                    builder.num_constraints_evaluated
                );
                total_failures += 1;
            }
        }
        assert_eq!(total_failures, 0, "KeccakSpongeChip has row-local constraint violations");
    }
}
