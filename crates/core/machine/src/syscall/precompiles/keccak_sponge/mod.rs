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

    /// Nets every chip's send/receive multiplicities for a real, deferred, honestly-split
    /// KeccakSponge-only shard, guarding against a lookup-level regression (an AIR `send`/
    /// `receive` call with no matching counterpart) independent of whether it happens to also
    /// break end-to-end proving (see `test_keccak_sponge_program_prove`).
    #[test]
    fn test_keccak_sponge_local_interactions_balance() {
        use crate::mips::MipsAir;
        use p3_koala_bear::KoalaBear;
        use slop_air::BaseAir;
        use slop_multilinear::{Mle, PaddedMle};
        use std::sync::Arc;
        use zkm_core_executor::{syscalls::SyscallCode, Executor};
        use zkm_hypercube::{
            air::{LookupScope, MachineAir},
            lookup::{debug_interactions_with_all_chips, LookupKind},
            prover::Traces,
            record::MachineRecord,
        };
        use zkm_stark::ZKMCoreOpts;

        setup_logger();
        let program = Program::from(KECCAK_SPONGE_ELF).unwrap();
        let mut runtime = Executor::new(program.clone(), ZKMCoreOpts::default());
        runtime.run().unwrap();

        // `SyscallChip`'s `Precompile`-kind `included()` requires `cpu_events`/
        // `global_memory_{initialize,finalize}_events` to *all* be empty -- it only activates on
        // a properly deferred-and-split shard, not the raw record `Executor::run()` produces (CPU
        // and precompile events still together). Replicate the real pipeline's
        // `record.defer()` + `deferred.split(..)` (see `prove_with_context`) to get a record
        // shaped like what actually gets proven for this precompile.
        let mut records = runtime.records;
        assert_eq!(records.len(), 1, "expected this small test program to fit in one checkpoint");
        let mut deferred = records.remove(0).defer();
        let split_records = deferred.split(true, None, ZKMCoreOpts::default().split_opts);
        let mut record = split_records
            .into_iter()
            .find(|r| !r.get_precompile_events(SyscallCode::KECCAK_SPONGE).is_empty())
            .expect("expected a deferred, split record with KeccakSponge precompile events");
        assert!(record.cpu_events.is_empty());
        assert!(record.global_memory_initialize_events.is_empty());
        assert!(record.global_memory_finalize_events.is_empty());

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        machine.generate_dependencies(std::iter::once(&mut record), None).unwrap();

        let chip_names =
            ["Program", "Byte", "Global", "SyscallPrecompile", "MemoryLocal", "KeccakSponge"];
        let chips = machine
            .chips()
            .iter()
            .filter(|c| chip_names.contains(&MachineAir::<KoalaBear>::name(*c).as_str()))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(chips.len(), chip_names.len(), "missing a chip by name");

        let max_log_row_count = 20u32;
        let mut preprocessed_named = std::collections::BTreeMap::new();
        let mut main_named = std::collections::BTreeMap::new();
        for chip in &chips {
            let name = MachineAir::<KoalaBear>::name(chip);
            let pre_mle = match chip.generate_preprocessed_trace(&program) {
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
            "KeccakSponge's local-scope send/receive interactions don't balance"
        );
    }
}
