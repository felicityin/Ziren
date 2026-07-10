use std::{collections::BTreeSet, ops::Deref};

use itertools::Itertools;
use p3_field::FieldAlgebra;
use slop_air::{Air, BaseAir};
use slop_matrix::dense::RowMajorMatrixView;
use slop_multilinear::{full_geq, Mle, Point};
use slop_sumcheck::PartialSumcheckProof;
use zkm_hypercube::{
    air::MachineAir, verifier::OpeningShapeError, Chip, ChipOpenedValues, LogUpEvaluations,
    ShardOpenedValues,
};
use zkm_recursion_compiler::{
    ir::Felt,
    prelude::{Builder, Ext, SymbolicExt},
};
use zkm_stark::{InnerChallenge, InnerVal};

use crate::{
    challenger::{CanObserveVariable, FieldChallengerVariable},
    sumcheck::verify_sumcheck,
    symbolic::IntoSymbolic,
    witness::{WitnessWriter, Witnessable},
    CircuitConfig,
};

pub type RecursiveVerifierConstraintFolder<'a, C> =
    zkm_hypercube::folder::GenericVerifierConstraintFolder<
        'a,
        <C as zkm_recursion_compiler::ir::Config>::F,
        <C as zkm_recursion_compiler::ir::Config>::EF,
        Felt<<C as zkm_recursion_compiler::ir::Config>::F>,
        Ext<
            <C as zkm_recursion_compiler::ir::Config>::F,
            <C as zkm_recursion_compiler::ir::Config>::EF,
        >,
        SymbolicExt<
            <C as zkm_recursion_compiler::ir::Config>::F,
            <C as zkm_recursion_compiler::ir::Config>::EF,
        >,
    >;

#[allow(clippy::type_complexity)]
pub fn eval_constraints<C: CircuitConfig, A>(
    builder: &mut Builder<C>,
    chip: &Chip<C::F, A>,
    opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    alpha: Ext<C::F, C::EF>,
    public_values: &[Felt<C::F>],
) -> Ext<C::F, C::EF>
where
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    let mut folder = RecursiveVerifierConstraintFolder::<C> {
        preprocessed: RowMajorMatrixView::new_row(&opening.preprocessed.local),
        main: RowMajorMatrixView::new_row(&opening.main.local),
        public_values,
        alpha,
        accumulator: SymbolicExt::ZERO,
        _marker: std::marker::PhantomData,
    };

    chip.eval(&mut folder);
    builder.eval(folder.accumulator)
}

/// Compute the padded row adjustment for a chip.
pub fn compute_padded_row_adjustment<C: CircuitConfig, A>(
    builder: &mut Builder<C>,
    chip: &Chip<C::F, A>,
    alpha: Ext<C::F, C::EF>,
    public_values: &[Felt<C::F>],
) -> Ext<C::F, C::EF>
where
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    let zero = builder.constant(C::EF::ZERO);
    let dummy_preprocessed_trace = vec![zero; chip.preprocessed_width()];
    let dummy_main_trace = vec![zero; chip.width()];

    let mut folder = RecursiveVerifierConstraintFolder::<C> {
        preprocessed: RowMajorMatrixView::new_row(&dummy_preprocessed_trace),
        main: RowMajorMatrixView::new_row(&dummy_main_trace),
        alpha,
        accumulator: SymbolicExt::ZERO,
        public_values,
        _marker: std::marker::PhantomData,
    };

    chip.eval(&mut folder);
    builder.eval(folder.accumulator)
}

#[allow(clippy::type_complexity)]
pub fn verify_opening_shape<C: CircuitConfig, A>(
    chip: &Chip<C::F, A>,
    opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
) -> Result<(), OpeningShapeError>
where
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    // Verify that the preprocessed width matches the expected value for the chip.
    if opening.preprocessed.local.len() != chip.preprocessed_width() {
        return Err(OpeningShapeError::PreprocessedWidthMismatch(
            chip.preprocessed_width(),
            opening.preprocessed.local.len(),
        ));
    }

    // Verify that the main width matches the expected value for the chip.
    if opening.main.local.len() != chip.width() {
        return Err(OpeningShapeError::MainWidthMismatch(chip.width(), opening.main.local.len()));
    }

    Ok(())
}

/// Verify a zerocheck proof against a shard's chip openings and the LogUp GKR evaluations that
/// feed into it.
///
/// This is a free function rather than a method on a shard verifier (unlike SP1's
/// `RecursiveShardVerifier::verify_zerocheck`) because the jagged/basefold-backed shard verifier
/// doesn't exist in this crate yet; `max_log_row_count` replaces the single field of `self`
/// (`self.pcs_verifier.max_log_row_count`) that SP1's version actually uses.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn verify_zerocheck<C, A, Bit, Challenger>(
    builder: &mut Builder<C>,
    max_log_row_count: usize,
    shard_chips: &BTreeSet<Chip<C::F, A>>,
    opened_values: &ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    gkr_evaluations: &LogUpEvaluations<Ext<C::F, C::EF>>,
    zerocheck_proof: &PartialSumcheckProof<Ext<C::F, C::EF>>,
    public_values: &[Felt<C::F>],
    challenger: &mut Challenger,
) where
    C: CircuitConfig,
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
    Challenger: FieldChallengerVariable<C, Bit> + CanObserveVariable<C, Felt<C::F>>,
{
    let zero: Ext<C::F, C::EF> = builder.constant(C::EF::ZERO);
    let one: Ext<C::F, C::EF> = builder.constant(C::EF::ONE);
    let mut rlc_eval: Ext<C::F, C::EF> = zero;

    let alpha = challenger.sample_ext(builder);
    let gkr_batch_open_challenge: SymbolicExt<C::F, C::EF> = challenger.sample_ext(builder).into();
    let lambda = challenger.sample_ext(builder);

    // Get the value of eq(zeta, sumcheck's reduced point).
    let point_symbolic = <Point<Ext<C::F, C::EF>> as IntoSymbolic<C>>::as_symbolic(
        &zerocheck_proof.point_and_eval.0,
    );

    let gkr_evaluations_point = IntoSymbolic::<C>::as_symbolic(&gkr_evaluations.point);

    let zerocheck_eq_val = Mle::full_lagrange_eval(&gkr_evaluations_point, &point_symbolic);

    let max_elements =
        shard_chips.iter().map(|chip| chip.width() + chip.preprocessed_width()).max().unwrap_or(0);

    let gkr_batch_open_challenge_powers =
        gkr_batch_open_challenge.powers().skip(1).take(max_elements).collect::<Vec<_>>();

    for (chip, openings) in shard_chips.iter().zip_eq(opened_values.chips.values()) {
        // Verify the shape of the opening arguments matches the expected values.
        verify_opening_shape::<C, A>(chip, openings).unwrap();

        let dimension = zerocheck_proof.point_and_eval.0.dimension();

        assert_eq!(dimension, max_log_row_count);

        let mut proof_point_extended = point_symbolic.clone();
        proof_point_extended.add_dimension(zero.into());
        let degree_symbolic_ext: Point<SymbolicExt<C::F, C::EF>> =
            openings.degree.iter().map(|x| SymbolicExt::from(*x)).collect::<Point<_>>();
        degree_symbolic_ext.iter().enumerate().for_each(|(i, x)| {
            builder.assert_ext_eq(*x * (*x - one), zero);
            if i >= 1 {
                builder.assert_ext_eq(*x * *degree_symbolic_ext.first().unwrap(), zero);
            }
        });
        let geq_val = full_geq(&degree_symbolic_ext, &proof_point_extended);

        let padded_row_adjustment =
            compute_padded_row_adjustment(builder, chip, alpha, public_values);

        let constraint_eval =
            eval_constraints::<C, A>(builder, chip, openings, alpha, public_values)
                - padded_row_adjustment * geq_val;

        let openings_batch = openings
            .main
            .local
            .iter()
            .chain(openings.preprocessed.local.iter())
            .copied()
            .zip(
                gkr_batch_open_challenge_powers
                    .iter()
                    .take(openings.main.local.len() + openings.preprocessed.local.len())
                    .copied(),
            )
            .map(|(opening, power)| opening * power)
            .sum::<SymbolicExt<C::F, C::EF>>();

        rlc_eval =
            builder.eval(rlc_eval * lambda + zerocheck_eq_val * (constraint_eval + openings_batch));
    }

    builder.assert_ext_eq(rlc_eval, zerocheck_proof.point_and_eval.1);

    let zerocheck_sum_modifications_from_gkr = gkr_evaluations
        .chip_openings
        .values()
        .map(|chip_evaluation| {
            chip_evaluation
                .main_trace_evaluations
                .deref()
                .iter()
                .copied()
                .chain(
                    chip_evaluation
                        .preprocessed_trace_evaluations
                        .as_ref()
                        .iter()
                        .flat_map(|&evals| evals.deref().iter().copied()),
                )
                .zip(gkr_batch_open_challenge_powers.iter().copied())
                .map(|(opening, power)| opening * power)
                .sum::<SymbolicExt<C::F, C::EF>>()
        })
        .collect::<Vec<_>>();

    let zerocheck_sum_modification: SymbolicExt<C::F, C::EF> = zerocheck_sum_modifications_from_gkr
        .iter()
        .fold(zero.into(), |acc, modification| lambda * acc + *modification);

    // Verify that the rlc claim is zero.
    builder.assert_ext_eq(zerocheck_proof.claimed_sum, zerocheck_sum_modification);

    // Verify the zerocheck proof.
    verify_sumcheck(builder, challenger, zerocheck_proof);

    // Observe the openings
    let len_felt: Felt<_> = builder.constant(C::F::from_canonical_usize(shard_chips.len()));
    challenger.observe(builder, len_felt);
    for opening in opened_values.chips.values() {
        challenger.observe_variable_length_extension_slice(builder, &opening.preprocessed.local);
        challenger.observe_variable_length_extension_slice(builder, &opening.main.local);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for ShardOpenedValues<InnerVal, InnerChallenge>
{
    type WitnessVariable = ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let chips = self.chips.read(builder);
        Self::WitnessVariable { chips }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.chips.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for ChipOpenedValues<InnerVal, InnerChallenge>
{
    type WitnessVariable = ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let preprocessed = self.preprocessed.read(builder);
        let main = self.main.read(builder);
        let degree = self.degree.read(builder);
        Self::WitnessVariable { preprocessed, main, degree }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.preprocessed.write(witness);
        self.main.write(witness);
        self.degree.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for zkm_hypercube::AirOpenedValues<InnerChallenge>
{
    type WitnessVariable = zkm_hypercube::AirOpenedValues<Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let local = self.local.read(builder);
        Self::WitnessVariable { local }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.local.write(witness);
    }
}
