pub mod air;
pub mod config;
pub mod folder;
pub mod lookup;
pub mod word;

pub use config::*;
pub use folder::*;

#[cfg(test)]
mod tests {
    use slop_air::{Air, BaseAir};
    use slop_algebra::AbstractField;
    use slop_koala_bear::KoalaBear;
    use slop_matrix::{dense::RowMajorMatrixView, Matrix};

    use crate::air::{InstructionAirBuilder, LookupScope, ZKMAirBuilder};

    use super::*;

    #[test]
    fn constructs_default_jagged_pcs_verifier() {
        let _verifier = zkm_pcs_verifier(default_fri_config(), 4, 20);
    }

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
}
