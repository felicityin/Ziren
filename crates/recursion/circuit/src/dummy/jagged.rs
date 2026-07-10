use p3_util::log2_ceil_usize;
use slop_algebra::FieldAlgebra;
use slop_alloc::CpuBackend;
use slop_basefold::BasefoldProof;
use slop_commit::Rounds;
use slop_jagged::{JaggedPcsProof, JaggedSumcheckEvalProof};
use slop_merkle_tree::{MerkleTreeOpeningAndProof, MerkleTreeTcsProof};
use slop_multilinear::MleEval;
use slop_stacked::StackedBasefoldProof;
use slop_tensor::Tensor;
use zkm_hypercube::config::{ZkmGlobalContext, DIGEST_SIZE, NUM_ZKM_COMMITMENTS};
use zkm_stark::{InnerChallenge, InnerVal};

use super::sumcheck::dummy_sumcheck_proof;

pub fn dummy_hash() -> [InnerVal; DIGEST_SIZE] {
    [InnerVal::zero(); DIGEST_SIZE]
}

pub fn dummy_query_proof(
    log_max_height: usize,
    log_blowup: usize,
    num_queries: usize,
) -> Vec<MerkleTreeOpeningAndProof<ZkmGlobalContext>> {
    (0..log_max_height)
        .map(|i| {
            let openings = Tensor::<InnerVal, _>::zeros_in([num_queries, 4 * 2], CpuBackend);
            let proof = Tensor::<[InnerVal; DIGEST_SIZE], _>::zeros_in(
                [num_queries, log_max_height - i + log_blowup - 1],
                CpuBackend,
            );

            MerkleTreeOpeningAndProof {
                values: openings,
                proof: MerkleTreeTcsProof {
                    paths: proof,
                    merkle_root: dummy_hash(),
                    log_tensor_height: log_max_height - i + log_blowup - 1,
                    width: 4 * 2,
                },
            }
        })
        .collect::<Vec<_>>()
}

/// Make a dummy PCS proof for a given proof shape. Used to generate vkey information for fixed
/// proof shapes.
///
/// The parameter `batch_shapes` contains (width, height) data for each matrix in each batch.
pub fn dummy_pcs_proof(
    fri_queries: usize,
    max_log_row_count: usize,
    log_stacking_height_multiples: &[usize],
    log_stacking_height: usize,
    log_blowup: usize,
    column_counts_and_added_cols: Rounds<(Vec<usize>, usize)>,
) -> JaggedPcsProof<ZkmGlobalContext, StackedBasefoldProof<ZkmGlobalContext>> {
    let (column_counts, added_cols): (Rounds<Vec<usize>>, Vec<usize>) =
        column_counts_and_added_cols.into_iter().unzip();
    let max_pcs_log_height = log_stacking_height;
    let dummy_component_polys = log_stacking_height_multiples.iter().map(|&x| {
        let proof = Tensor::<[InnerVal; DIGEST_SIZE], _>::zeros_in(
            [fri_queries, max_pcs_log_height + log_blowup],
            CpuBackend,
        );
        MerkleTreeOpeningAndProof::<ZkmGlobalContext> {
            values: Tensor::<InnerVal, _>::zeros_in([fri_queries, x], CpuBackend),
            proof: MerkleTreeTcsProof {
                paths: proof,
                merkle_root: dummy_hash(),
                log_tensor_height: max_pcs_log_height + log_blowup,
                width: x,
            },
        }
    });
    let basefold_proof = BasefoldProof::<ZkmGlobalContext> {
        univariate_messages: vec![[InnerChallenge::zero(); 2]; max_pcs_log_height],
        fri_commitments: vec![dummy_hash(); max_pcs_log_height],
        final_poly: InnerChallenge::zero(),
        pow_witness: InnerVal::zero(),
        batch_grinding_witness: InnerVal::zero(),
        component_polynomials_query_openings_and_proofs: dummy_component_polys.collect(),
        query_phase_openings_and_proofs: dummy_query_proof(
            max_pcs_log_height,
            log_blowup,
            fri_queries,
        ),
    };

    let batch_evaluations: Rounds<MleEval<InnerChallenge, CpuBackend>> = Rounds {
        rounds: log_stacking_height_multiples
            .iter()
            .map(|&x| vec![InnerChallenge::zero(); x].into())
            .collect(),
    };

    let stacked_proof = StackedBasefoldProof { basefold_proof, batch_evaluations };

    let total_trace = log2_ceil_usize(
        log_stacking_height_multiples.iter().sum::<usize>() * (1 << log_stacking_height),
    );
    let total_num_variables = total_trace;

    let partial_sumcheck_proof = dummy_sumcheck_proof(total_trace, 2);

    let eval_sumcheck_proof = dummy_sumcheck_proof(2 * (total_num_variables + 1), 2);

    let jagged_eval_proof = JaggedSumcheckEvalProof { partial_sumcheck_proof: eval_sumcheck_proof };

    let new_column_counts: Rounds<Vec<usize>> = column_counts
        .into_iter()
        .zip(added_cols.iter())
        .map(|(x, &added)| {
            // Commit paths always reserve at least one padding column.
            let added = added.max(1);
            x.into_iter().chain([added - 1, 1]).collect()
        })
        .collect();

    let row_counts_and_column_counts: Rounds<Vec<(usize, usize)>> = new_column_counts
        .clone()
        .into_iter()
        .map(|cc| cc.iter().map(|&c| (0, c)).collect())
        .collect();

    JaggedPcsProof {
        pcs_proof: stacked_proof,
        jagged_eval_proof,
        sumcheck_proof: partial_sumcheck_proof,
        merkle_tree_commitments: vec![dummy_hash(); NUM_ZKM_COMMITMENTS].into_iter().collect(),
        row_counts_and_column_counts,
        expected_eval: InnerChallenge::zero(),
        max_log_row_count,
        log_m: total_trace,
    }
}
