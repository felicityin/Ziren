/// The maximum number of public values a Ziren shard proof can have.
pub const PROOF_MAX_NUM_PVS: usize = 117;

pub mod air;
pub mod chip;
pub mod config;
pub mod debug;
pub mod folder;
pub mod logup_gkr;
pub mod lookup;
pub mod machine;
pub mod prover;
pub mod record;
pub mod septic_curve;
pub mod septic_digest;
pub mod septic_extension;
pub mod shape;
pub mod shard_context;
pub mod verifier;
pub mod word;
pub mod zerocheck;

pub use chip::*;
pub use config::*;
pub use debug::*;
pub use folder::*;
pub use logup_gkr::*;
pub use machine::*;
pub use shard_context::*;
pub use verifier::*;
pub use zerocheck::*;

#[cfg(test)]
mod tests {
    use p3_field::FieldAlgebra;
    use slop_air::{Air, BaseAir};
    use slop_koala_bear::KoalaBear;
    use slop_matrix::{dense::{RowMajorMatrix, RowMajorMatrixView}, Matrix};

    use crate::air::{InstructionAirBuilder, LookupScope, MachineAir, ZKMAirBuilder};

    use super::*;

    #[test]
    fn constructs_default_jagged_pcs_verifier() {
        let _verifier = zkm_pcs_verifier(default_fri_config(), 4, 20);
    }

    #[derive(Debug)]
    struct AddAir;

    impl<F> BaseAir<F> for AddAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl<AB: ZKMAirBuilder> Air<AB> for AddAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0);
            builder.assert_zero(row[0] + row[1] - row[2]);
        }
    }

    fn eval_constraint_sumcheck(main_values: [KoalaBear; 3]) -> KoalaBear {
        let main = RowMajorMatrixView::new(main_values.as_slice(), 3);
        let preprocessed = RowMajorMatrixView::new([].as_slice(), 0);
        let mut folder = ConstraintSumcheckFolder::<KoalaBear, KoalaBear, KoalaBear> {
            preprocessed,
            main,
            powers_of_alpha: &[KoalaBear::one()],
            accumulator: KoalaBear::zero(),
            public_values: &[],
            constraint_index: 0,
        };
        AddAir.eval(&mut folder);
        folder.accumulator
    }

    #[test]
    fn constraint_sumcheck_folder_accepts_satisfying_row() {
        let a = KoalaBear::from_canonical_u32(3);
        let b = KoalaBear::from_canonical_u32(4);
        let c = KoalaBear::from_canonical_u32(7);
        assert_eq!(eval_constraint_sumcheck([a, b, c]), KoalaBear::zero());
    }

    #[test]
    fn constraint_sumcheck_folder_rejects_violating_row() {
        let a = KoalaBear::from_canonical_u32(3);
        let b = KoalaBear::from_canonical_u32(4);
        let c = KoalaBear::from_canonical_u32(9);
        assert_ne!(eval_constraint_sumcheck([a, b, c]), KoalaBear::zero());
    }

    #[test]
    fn base_air_builder_send_alu_records_a_local_lookup() {
        struct RecordingFolder<'a> {
            main: RowMajorMatrixView<'a, KoalaBear>,
            sent: Vec<crate::air::AirLookup<KoalaBear>>,
        }

        impl<'a> slop_air::AirBuilder for RecordingFolder<'a> {
            type F = KoalaBear;
            type Expr = KoalaBear;
            type Var = KoalaBear;
            type M = RowMajorMatrixView<'a, KoalaBear>;

            fn main(&self) -> Self::M {
                self.main
            }

            fn is_first_row(&self) -> Self::Expr {
                KoalaBear::zero()
            }

            fn is_last_row(&self) -> Self::Expr {
                KoalaBear::zero()
            }

            fn is_transition_window(&self, _: usize) -> Self::Expr {
                KoalaBear::zero()
            }

            fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {}
        }

        impl<'a> crate::air::MessageBuilder<crate::air::AirLookup<KoalaBear>> for RecordingFolder<'a> {
            fn send(&mut self, message: crate::air::AirLookup<KoalaBear>, _scope: LookupScope) {
                self.sent.push(message);
            }

            fn receive(&mut self, _message: crate::air::AirLookup<KoalaBear>, _scope: LookupScope) {}
        }

        let main = RowMajorMatrixView::new([].as_slice(), 0);
        let mut folder = RecordingFolder { main, sent: Vec::new() };

        folder.send_alu(
            KoalaBear::from_canonical_u32(1),
            crate::word::Word([KoalaBear::zero(); 4]),
            crate::word::Word([KoalaBear::zero(); 4]),
            crate::word::Word([KoalaBear::zero(); 4]),
            KoalaBear::one(),
        );

        assert_eq!(folder.sent.len(), 1);
        assert_eq!(folder.sent[0].kind, crate::lookup::LookupKind::Instruction);
    }

    #[derive(Debug, Default, Clone)]
    struct AddRecord;

    impl crate::record::MachineRecord for AddRecord {
        fn stats(&self) -> hashbrown::HashMap<String, usize> {
            hashbrown::HashMap::new()
        }

        fn append(&mut self, _other: &mut Self) {}

        fn public_values<F: FieldAlgebra>(&self) -> Vec<F> {
            Vec::new()
        }

        fn eval_public_values<AB: crate::air::ZKMAirBuilder>(_builder: &mut AB) {}

        fn interactions_in_public_values() -> Vec<crate::lookup::LookupKind> {
            vec![]
        }
    }

    struct AddProgram;

    impl crate::air::MachineProgram<KoalaBear> for AddProgram {
        fn pc_start(&self) -> KoalaBear {
            KoalaBear::zero()
        }

        fn initial_global_cumulative_sum(&self) -> crate::septic_digest::SepticDigest<KoalaBear> {
            crate::septic_digest::SepticDigest::zero()
        }
    }

    impl crate::air::MachineAir<KoalaBear> for AddAir {
        type Record = AddRecord;
        type Program = AddProgram;
        type Error = std::io::Error;

        fn name(&self) -> String {
            "Add".to_string()
        }

        fn generate_trace(
            &self,
            _input: &Self::Record,
            _output: &mut Self::Record,
        ) -> Result<RowMajorMatrix<KoalaBear>, Self::Error> {
            Ok(RowMajorMatrix::new(vec![KoalaBear::zero(); 3], 3))
        }

        fn included(&self, _shard: &Self::Record) -> bool {
            true
        }
    }

    fn assert_zerocheck_air<A: crate::ZerocheckAir<KoalaBear, crate::config::ZkmExtensionField>>() {}

    #[test]
    fn add_air_satisfies_zerocheck_air_bound() {
        assert_zerocheck_air::<AddAir>();
    }

    #[test]
    fn chip_extracts_no_interactions_from_add_air() {
        let chip = crate::chip::Chip::new(AddAir);
        assert_eq!(chip.sends().len(), 0);
        assert_eq!(chip.receives().len(), 0);
        assert_eq!(chip.name(), "Add");
    }

    struct ByteSendingAir;

    impl<F> BaseAir<F> for ByteSendingAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl<AB: ZKMAirBuilder> Air<AB> for ByteSendingAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0);
            builder.send_byte(AB::Expr::one(), row[0], AB::Expr::zero(), AB::Expr::zero(), AB::Expr::one());
        }
    }

    impl crate::air::MachineAir<KoalaBear> for ByteSendingAir {
        type Record = AddRecord;
        type Program = AddProgram;
        type Error = std::io::Error;

        fn name(&self) -> String {
            "ByteSending".to_string()
        }

        fn generate_trace(
            &self,
            _input: &Self::Record,
            _output: &mut Self::Record,
        ) -> Result<RowMajorMatrix<KoalaBear>, Self::Error> {
            Ok(RowMajorMatrix::new(vec![KoalaBear::zero()], 1))
        }

        fn included(&self, _shard: &Self::Record) -> bool {
            true
        }
    }

    #[test]
    fn chip_extracts_one_send_from_byte_sending_air() {
        let chip = crate::chip::Chip::new(ByteSendingAir);
        assert_eq!(chip.sends().len(), 1);
        assert_eq!(chip.receives().len(), 0);
        assert_eq!(chip.sends()[0].kind, crate::lookup::LookupKind::Byte);
    }

    #[test]
    fn machine_smallest_cluster_finds_containing_cluster() {
        use std::collections::BTreeSet;

        let chip = crate::chip::Chip::new(AddAir);
        let all_chips = vec![chip.clone()];
        let shape = crate::machine::MachineShape::all(&all_chips);
        let machine = crate::machine::Machine::new(all_chips, PROOF_MAX_NUM_PVS, shape);

        assert_eq!(machine.chips().len(), 1);

        let requested: BTreeSet<_> = std::iter::once(chip).collect();
        let cluster = machine.smallest_cluster(&requested).expect("cluster should exist");
        assert_eq!(cluster.len(), 1);
    }

    #[test]
    fn shard_verifier_constructs_from_basefold_parameters_and_matches_machine() {
        let chip = crate::chip::Chip::new(AddAir);
        let all_chips = vec![chip];
        let shape = crate::machine::MachineShape::all(&all_chips);
        let machine = crate::machine::Machine::new(all_chips, PROOF_MAX_NUM_PVS, shape);

        let verifier = crate::verifier::ShardVerifier::from_basefold_parameters(default_fri_config(), 4, 20, machine);

        assert_eq!(verifier.max_log_row_count(), 20);
        assert_eq!(verifier.machine().chips().len(), 1);
        // Constructing a challenger should not panic.
        let _challenger = verifier.challenger();
    }
}
