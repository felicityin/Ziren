use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use p3_field::FieldAlgebra;
use slop_air::Air;
use slop_challenger::{GrindingChallenger, IopCtx};
use slop_commit::Rounds;
use slop_multilinear::{Evaluations, MleEval};
use slop_sumcheck::PartialSumcheckProof;
use zkm_hypercube::{
    air::MachineAir, septic_curve::SepticCurve, septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    verifier::{MachineVerifyingKey, ShardProof},
    LogupGkrProof, Machine, ShardOpenedValues,
};
use zkm_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Felt, SymbolicFelt},
    prelude::Ext,
};
use zkm_stark::{InnerChallenge, InnerVal};

use crate::{
    basefold::RecursiveBasefoldProof,
    challenger::{CanObserveVariable, FieldChallengerVariable},
    hash::FieldHasherVariable,
    jagged::{
        JaggedPcsProofVariable, RecursiveJaggedPcsVerifier, RecursiveMachineJaggedPcsVerifier,
    },
    logup_gkr::RecursiveLogUpGkrVerifier,
    witness::{WitnessWriter, Witnessable},
    zerocheck::{verify_zerocheck, RecursiveVerifierConstraintFolder},
    CircuitConfig,
};

pub use crate::logup_gkr::RecursiveVerifierPublicValuesConstraintFolder;

#[allow(clippy::type_complexity)]
pub struct ShardProofVariable<C: CircuitConfig, HV: FieldHasherVariable<C>> {
    /// The commitments to main traces.
    pub main_commitment: HV::DigestVariable,
    /// The values of the traces at the final random point.
    pub opened_values: ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    /// The zerocheck IOP proof.
    pub zerocheck_proof: PartialSumcheckProof<Ext<C::F, C::EF>>,
    /// The public values.
    pub public_values: Vec<Felt<C::F>>,
    /// The `LogUp` + GKR IOP proof.
    pub logup_gkr_proof: LogupGkrProof<Felt<C::F>, Ext<C::F, C::EF>>,
    /// The evaluation proof.
    pub evaluation_proof:
        JaggedPcsProofVariable<C::F, C::EF, RecursiveBasefoldProof<C, HV>, HV::DigestVariable>,
}

impl<C: CircuitConfig, HV: FieldHasherVariable<C>> ShardProofVariable<C, HV> {
    pub fn contains_cpu(&self) -> bool {
        self.opened_values.chips.contains_key("Cpu")
    }

    pub fn contains_memory_init(&self) -> bool {
        self.opened_values.chips.contains_key("MemoryGlobalInit")
    }

    pub fn contains_memory_finalize(&self) -> bool {
        self.opened_values.chips.contains_key("MemoryGlobalFinalize")
    }
}

pub struct MachineVerifyingKeyVariable<C: CircuitConfig, HV: FieldHasherVariable<C>> {
    pub pc_start: Felt<C::F>,
    /// The starting global digest of the program, after incorporating the initial memory.
    pub initial_global_cumulative_sum: SepticDigest<Felt<C::F>>,
    /// The preprocessed commitments.
    pub preprocessed_commit: HV::DigestVariable,
}

impl<C: CircuitConfig, HV: FieldHasherVariable<C>> MachineVerifyingKeyVariable<C, HV> {
    /// Hash the verifying key + prep domains into a single digest.
    /// poseidon2(commit[0..8] || pc_start || `initial_global_cumulative_sum`)
    pub fn hash(&self, builder: &mut Builder<C>) -> HV::DigestVariable
    where
        HV::DigestVariable: IntoIterator<Item = Felt<C::F>>,
    {
        let mut inputs = Vec::new();
        inputs.extend(self.preprocessed_commit);
        inputs.push(self.pc_start);
        inputs.extend(self.initial_global_cumulative_sum.0.x.0);
        inputs.extend(self.initial_global_cumulative_sum.0.y.0);

        HV::hash(builder, &inputs)
    }

    /// Observe the verifying key into a challenger, priming it before shard verification.
    ///
    /// Must match `zkm_hypercube::verifier::config::MachineVerifyingKey::observe_into` exactly
    /// (same fields, same order) so the in-circuit Fiat-Shamir transcript matches the native
    /// prover's.
    pub fn observe_into<Challenger>(&self, builder: &mut Builder<C>, challenger: &mut Challenger)
    where
        Challenger: CanObserveVariable<C, HV::DigestVariable> + CanObserveVariable<C, Felt<C::F>>,
    {
        challenger.observe(builder, self.preprocessed_commit);
        challenger.observe(builder, self.pc_start);
        challenger.observe_slice(builder, self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(builder, self.initial_global_cumulative_sum.0.y.0);
    }
}

/// A verifier for shard proofs.
pub struct RecursiveShardVerifier<
    C: CircuitConfig,
    HV: FieldHasherVariable<C>,
    Challenger,
    A: MachineAir<C::F>,
> {
    /// The machine.
    pub machine: Machine<C::F, A>,
    /// The jagged pcs verifier.
    pub pcs_verifier: RecursiveJaggedPcsVerifier<C, HV, Challenger>,
    pub _phantom: PhantomData<(C, HV, Challenger, A)>,
}

impl<
        C: CircuitConfig<F = p3_koala_bear::KoalaBear>,
        HV: FieldHasherVariable<C>,
        Challenger,
        A: MachineAir<C::F>,
    > RecursiveShardVerifier<C, HV, Challenger, A>
where
    Challenger: FieldChallengerVariable<C, C::Bit> + CanObserveVariable<C, HV::DigestVariable>,
{
    pub fn verify_shard(
        &self,
        builder: &mut Builder<C>,
        vk: &MachineVerifyingKeyVariable<C, HV>,
        proof: &ShardProofVariable<C, HV>,
        challenger: &mut Challenger,
    ) where
        A: for<'b> Air<RecursiveVerifierConstraintFolder<'b, C>>,
    {
        let ShardProofVariable {
            main_commitment,
            opened_values,
            evaluation_proof,
            zerocheck_proof,
            public_values,
            logup_gkr_proof,
        } = proof;

        // Convert height bits to felts.
        let heights = opened_values
            .chips
            .iter()
            .map(|(name, x)| (name.clone(), x.degree.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut height_felts_map: BTreeMap<String, Felt<C::F>> = BTreeMap::new();
        let two = SymbolicFelt::from_canonical_u32(2);
        for (name, height) in &heights {
            let mut acc = SymbolicFelt::ZERO;
            // Assert max height to avoid overflow during prefix-sum-checks.
            assert!(height.len() == self.pcs_verifier.max_log_row_count + 1);
            height.iter().for_each(|x| {
                acc = *x + two * acc;
            });
            height_felts_map.insert(name.clone(), builder.eval(acc));
        }

        // Observe the public values.
        challenger.observe_slice(builder, public_values.to_vec());

        for value in public_values[self.machine.num_pv_elts()..].iter() {
            builder.assert_felt_eq(*value, C::F::ZERO);
        }

        // Observe the main commitment.
        challenger.observe(builder, *main_commitment);
        let num_chips: Felt<C::F> = builder.eval(C::F::from_canonical_usize(heights.len()));
        // Observe the number of chips.
        challenger.observe(builder, num_chips);

        for (name, height) in height_felts_map.iter() {
            challenger.observe(builder, *height);
            let mut inputs: Vec<Felt<C::F>> = vec![];
            inputs.push(builder.eval(C::F::from_canonical_usize(name.len())));
            for byte in name.as_bytes() {
                inputs.push(builder.eval(C::F::from_canonical_u8(*byte)));
            }
            challenger.observe_slice(builder, inputs);
        }

        let shard_chips = self
            .machine
            .chips()
            .iter()
            .filter(|chip| heights.contains_key(&chip.name()))
            .cloned()
            .collect::<BTreeSet<_>>();

        let degrees = opened_values.chips.values().map(|x| x.degree.clone()).collect::<Vec<_>>();

        let max_log_row_count = self.pcs_verifier.max_log_row_count;

        // Verify the `LogUp` GKR proof.
        builder.cycle_tracker_v2_enter("verify-logup-gkr".to_string());
        RecursiveLogUpGkrVerifier::<C, A>::verify_logup_gkr(
            builder,
            &shard_chips,
            &degrees,
            max_log_row_count,
            logup_gkr_proof,
            public_values,
            challenger,
        );
        builder.cycle_tracker_v2_exit();

        // Verify the zerocheck proof.
        builder.cycle_tracker_v2_enter("verify-zerocheck".to_string());
        verify_zerocheck(
            builder,
            max_log_row_count,
            &shard_chips,
            opened_values,
            &logup_gkr_proof.logup_evaluations,
            zerocheck_proof,
            public_values,
            challenger,
        );
        builder.cycle_tracker_v2_exit();

        // Verify the opening proof.
        let (preprocessed_openings_for_proof, main_openings_for_proof): (Vec<_>, Vec<_>) = proof
            .opened_values
            .chips
            .values()
            .map(|opening| (opening.preprocessed.clone(), opening.main.clone()))
            .unzip();

        let preprocessed_openings = preprocessed_openings_for_proof
            .iter()
            .map(|x| x.local.iter().as_slice())
            .collect::<Vec<_>>();

        let main_openings = main_openings_for_proof
            .iter()
            .map(|x| x.local.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        let filtered_preprocessed_openings = preprocessed_openings
            .clone()
            .into_iter()
            .filter(|x| !x.is_empty())
            .map(|x| x.iter().copied().collect::<MleEval<_>>())
            .collect::<Evaluations<_>>();

        let preprocessed_column_count = filtered_preprocessed_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .collect::<Vec<_>>();

        let added_columns: Vec<usize> =
            proof.evaluation_proof.column_counts.iter().map(|cc| cc[cc.len() - 2] + 1).collect();

        let unfiltered_preprocessed_column_count = preprocessed_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .chain(std::iter::once(added_columns[0] - 1))
            .collect::<Vec<_>>();

        let main_column_count =
            main_openings.iter().map(|table_openings| table_openings.len()).collect::<Vec<_>>();

        let unfiltered_main_column_count = main_openings
            .iter()
            .map(|table_openings| table_openings.len())
            .chain(std::iter::once(added_columns[1] - 1))
            .collect::<Vec<_>>();

        let (commitments, column_counts, unfiltered_column_counts, openings) = (
            vec![vk.preprocessed_commit, *main_commitment],
            vec![preprocessed_column_count, main_column_count.clone()],
            vec![unfiltered_preprocessed_column_count, unfiltered_main_column_count],
            Rounds { rounds: vec![filtered_preprocessed_openings, main_openings] },
        );

        let machine_jagged_verifier =
            RecursiveMachineJaggedPcsVerifier::new(&self.pcs_verifier, column_counts.clone());

        let openings = openings
            .into_iter()
            .map(|round| {
                round
                    .into_iter()
                    .flat_map(std::iter::IntoIterator::into_iter)
                    .collect::<MleEval<_>>()
            })
            .collect::<Vec<_>>();

        builder.cycle_tracker_v2_enter("jagged-verifier".to_string());
        let prefix_sum_felts = machine_jagged_verifier.verify_trusted_evaluations(
            builder,
            &commitments,
            zerocheck_proof.point_and_eval.0.clone(),
            &openings,
            evaluation_proof,
            challenger,
        );
        builder.cycle_tracker_v2_exit();

        let row_count_felt: Felt<_> =
            builder.constant(C::F::from_canonical_u32(1 << self.pcs_verifier.max_log_row_count));

        let params: Vec<Vec<Felt<C::F>>> = unfiltered_column_counts
            .iter()
            .map(|round| {
                round
                    .iter()
                    .copied()
                    .zip(height_felts_map.values().copied().chain(std::iter::once(row_count_felt)))
                    .flat_map(|(column_count, height)| {
                        std::iter::repeat_n(height, column_count).collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let preprocessed_count = params[0].len();
        let params = params.into_iter().flatten().collect::<Vec<_>>();

        builder.cycle_tracker_v2_enter("jagged - prefix-sum-checks".to_string());
        let mut param_index = 0;
        // The prefix_sum_felts coming from the C::prefix_sum_checks call excludes what is the last
        // element, namely the total area, in the Rust verifier. We add that check in manually
        // below. That is why the Rust verifier `skip_indices` has two elements, while this
        // one has one.
        let skip_indices = [preprocessed_count];

        prefix_sum_felts
            .iter()
            .zip(prefix_sum_felts.iter().skip(1))
            .enumerate()
            .filter(|(i, _)| !skip_indices.contains(i))
            .for_each(|(_, (x, y))| {
                let sum = *x + params[param_index];
                builder.assert_felt_eq(sum, *y);
                param_index += 1;
            });

        builder.assert_felt_eq(prefix_sum_felts[0], C::F::ZERO);

        // Check that the preprocessed prefix sum is the correct multiple of `stacking_height`.
        builder.assert_felt_eq(
            prefix_sum_felts[skip_indices[0] + 1],
            C::F::from_canonical_usize(
                (1 << self.pcs_verifier.stacked_pcs_verifier.log_stacking_height)
                    * evaluation_proof.pcs_proof.batch_evaluations.rounds[0].num_polynomials(),
            ),
        );

        let preprocessed_padding_col_height =
            builder.eval(prefix_sum_felts[skip_indices[0] + 1] - prefix_sum_felts[skip_indices[0]]);
        let preprocessed_padding_col_bit_decomp = C::num2bits(
            builder,
            preprocessed_padding_col_height,
            self.pcs_verifier.max_log_row_count + 1,
        );

        // We want to constrain the padding column to be in the range [0, 2^{max_log_row_count}].
        // The above constraints ensure that the padding column is in the range [0,
        // 2^{max_log_row_count+1}). The following constraints exclude the range
        // (2^{max_log_row_count}, 2^{max_log_row_count+1}), namely by ensuring that if the
        // the `max_log_row_count`-th bit is 1, then the less significant bits must be zero.
        //
        // NOTE: Strictly speaking, this is not necessary, since the jagged polynomial will
        // force a zero evaluation in case any column height is greater than
        // `2^{max_log_row_count}`, but we add this constraint for extra security, since it
        // does not have a significant performance impact.
        let max_bit = preprocessed_padding_col_bit_decomp[self.pcs_verifier.max_log_row_count];
        let max_bit = C::bits2num(builder, vec![max_bit]);
        let zero: Felt<_> = builder.constant(C::F::ZERO);
        for bit in
            preprocessed_padding_col_bit_decomp.iter().take(self.pcs_verifier.max_log_row_count)
        {
            let bit_felt = C::bits2num(builder, vec![*bit]);
            builder.assert_felt_eq(max_bit * bit_felt, zero);
        }
        let num_cols = prefix_sum_felts.len();

        // Repeat the process above for the main trace padding column.
        let main_padding_col_height =
            builder.eval(prefix_sum_felts[num_cols - 1] - prefix_sum_felts[num_cols - 2]);

        let main_padding_col_bit_decomp =
            C::num2bits(builder, main_padding_col_height, zkm_recursion_core::NUM_BITS);

        let max_bit = main_padding_col_bit_decomp[self.pcs_verifier.max_log_row_count];
        let max_bit = C::bits2num(builder, vec![max_bit]);
        for bit in main_padding_col_bit_decomp.iter().skip(self.pcs_verifier.max_log_row_count + 1)
        {
            C::assert_bit_zero(builder, *bit);
        }
        for bit in main_padding_col_bit_decomp.iter().take(self.pcs_verifier.max_log_row_count) {
            let bit_felt = C::bits2num(builder, vec![*bit]);
            builder.assert_felt_eq(max_bit * bit_felt, zero);
        }

        // Compute the total area from the shape of the stacked PCS proof.
        let total_area_felt: Felt<_> = builder.constant(C::F::from_canonical_usize(
            (1 << self.pcs_verifier.stacked_pcs_verifier.log_stacking_height)
                * proof
                    .evaluation_proof
                    .pcs_proof
                    .batch_evaluations
                    .iter()
                    .map(|evaluations| evaluations.num_polynomials())
                    .sum::<usize>(),
        ));

        // Convert the final prefix sum to a symbolic felt.
        let mut acc = SymbolicFelt::ZERO;
        // Assert max height to avoid overflow during prefix-sum-checks.
        proof.evaluation_proof.params.col_prefix_sums.iter().last().unwrap().iter().for_each(|x| {
            acc = *x + two * acc;
        });

        // Check equality between the two above-computed values.
        builder.assert_felt_eq(acc, total_area_felt);

        builder.cycle_tracker_v2_exit();
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C>
    for SepticDigest<InnerVal>
{
    type WitnessVariable = SepticDigest<Felt<C::F>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let x = self.0.x.0.read(builder);
        let y = self.0.y.0.read(builder);
        SepticDigest(SepticCurve { x: SepticExtension(x), y: SepticExtension(y) })
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.0.x.0.write(witness);
        self.0.y.0.write(witness);
    }
}

impl<C, GC> Witnessable<C> for MachineVerifyingKey<GC>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge>,
    GC: IopCtx<F = C::F, EF = C::EF> + FieldHasherVariable<C>,
    <GC as IopCtx>::Digest:
        Witnessable<C, WitnessVariable = <GC as FieldHasherVariable<C>>::DigestVariable>,
{
    type WitnessVariable = MachineVerifyingKeyVariable<C, GC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let pc_start = self.pc_start.read(builder);
        let initial_global_cumulative_sum = self.initial_global_cumulative_sum.read(builder);
        let preprocessed_commit = self.preprocessed_commit.read(builder);
        MachineVerifyingKeyVariable { pc_start, initial_global_cumulative_sum, preprocessed_commit }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.pc_start.write(witness);
        self.initial_global_cumulative_sum.write(witness);
        self.preprocessed_commit.write(witness);
    }
}

impl<C, GC, Proof> Witnessable<C> for ShardProof<GC, Proof>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge>,
    GC: IopCtx<F = C::F, EF = C::EF> + FieldHasherVariable<C>,
    Proof: Witnessable<
        C,
        WitnessVariable = crate::basefold::stacked::RecursiveStackedPcsProof<
            RecursiveBasefoldProof<C, GC>,
            InnerVal,
            InnerChallenge,
        >,
    >,
    <GC as IopCtx>::Digest:
        Witnessable<C, WitnessVariable = <GC as FieldHasherVariable<C>>::DigestVariable>,
    <GC::Challenger as GrindingChallenger>::Witness: Witnessable<C, WitnessVariable = Felt<C::F>>,
{
    type WitnessVariable = ShardProofVariable<C, GC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let main_commitment = self.main_commitment.read(builder);
        let opened_values = self.opened_values.read(builder);
        let zerocheck_proof = self.zerocheck_proof.read(builder);
        let public_values = self.public_values.read(builder);
        let logup_gkr_proof = self.logup_gkr_proof.read(builder);
        let evaluation_proof = self.evaluation_proof.read(builder);
        ShardProofVariable {
            main_commitment,
            opened_values,
            zerocheck_proof,
            public_values,
            logup_gkr_proof,
            evaluation_proof,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.main_commitment.write(witness);
        self.opened_values.write(witness);
        self.zerocheck_proof.write(witness);
        self.public_values.write(witness);
        self.logup_gkr_proof.write(witness);
        self.evaluation_proof.write(witness);
    }
}
