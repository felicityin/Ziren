use std::{collections::BTreeSet, iter::once};

use p3_air::BaseAir;
use slop_algebra::FieldAlgebra;
use slop_basefold::FriConfig;
use slop_multilinear::Point;
use slop_stacked::StackedBasefoldProof;
use zkm_hypercube::{
    air::MachineAir, config::ZkmGlobalContext, septic_digest::SepticDigest,
    verifier::MachineVerifyingKey, AirOpenedValues, Chip, ChipOpenedValues, ShardOpenedValues,
    ShardProof, PROOF_MAX_NUM_PVS,
};
use zkm_stark::{InnerChallenge, InnerVal};

use crate::dummy::{
    jagged::dummy_pcs_proof, logup_gkr::dummy_gkr_proof, sumcheck::dummy_sumcheck_proof,
};

pub fn dummy_vk() -> MachineVerifyingKey<ZkmGlobalContext> {
    MachineVerifyingKey {
        pc_start: InnerVal::zero(),
        initial_global_cumulative_sum: SepticDigest::zero(),
        preprocessed_commit: [InnerVal::zero(); 8],
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dummy_shard_proof<A: MachineAir<InnerVal>>(
    shard_chips: BTreeSet<Chip<InnerVal, A>>,
    max_log_row_count: usize,
    fri_config: FriConfig<InnerVal>,
    log_stacking_height: usize,
    log_stacking_height_multiples: &[usize],
    added_cols: &[usize],
) -> ShardProof<ZkmGlobalContext, StackedBasefoldProof<ZkmGlobalContext>> {
    let fri_queries = fri_config.num_queries;
    let log_blowup = fri_config.log_blowup;

    let evaluation_proof = dummy_pcs_proof(
        fri_queries,
        max_log_row_count,
        log_stacking_height_multiples,
        log_stacking_height,
        log_blowup,
        once(shard_chips.iter().map(MachineAir::preprocessed_width).filter(|x| *x > 0).collect())
            .chain(once(shard_chips.iter().map(|chip| chip.width()).collect::<Vec<_>>()))
            .zip(added_cols.iter().copied())
            .collect(),
    );

    let logup_gkr_proof =
        dummy_gkr_proof::<_, InnerChallenge, _>(&shard_chips, max_log_row_count);

    let zerocheck_proof = dummy_sumcheck_proof::<InnerChallenge>(max_log_row_count, 4);

    ShardProof {
        public_values: vec![InnerVal::zero(); PROOF_MAX_NUM_PVS],
        main_commitment: [InnerVal::zero(); 8],
        logup_gkr_proof,
        zerocheck_proof,
        opened_values: ShardOpenedValues {
            chips: shard_chips
                .iter()
                .map(|chip| {
                    (
                        chip.name().to_string(),
                        ChipOpenedValues {
                            preprocessed: AirOpenedValues {
                                local: vec![InnerChallenge::zero(); chip.preprocessed_width()],
                            },
                            main: AirOpenedValues {
                                local: vec![InnerChallenge::zero(); chip.width()],
                            },
                            degree: Point::from_usize(0, max_log_row_count + 1),
                        },
                    )
                })
                .collect(),
        },
        evaluation_proof,
    }
}

#[cfg(test)]
mod tests {
    use zkm_core_machine::mips::MipsAir;
    use zkm_hypercube::config::default_fri_config;

    use super::*;

    #[test]
    fn dummy_shard_proof_matches_machine_shape() {
        let machine = MipsAir::<InnerVal>::hypercube_machine();
        let shard_chips: BTreeSet<_> = machine.chips().iter().cloned().collect();
        let num_chips = shard_chips.len();

        let vk = dummy_vk();
        assert_eq!(vk.preprocessed_commit, [InnerVal::zero(); 8]);

        let proof = dummy_shard_proof(
            shard_chips,
            10,
            default_fri_config(),
            10,
            &[1, 1],
            &[1, 1],
        );

        assert_eq!(proof.opened_values.chips.len(), num_chips);
        assert_eq!(proof.public_values.len(), PROOF_MAX_NUM_PVS);
        assert_eq!(proof.evaluation_proof.max_log_row_count, 10);
    }
}
