use std::collections::BTreeMap;

use p3_util::reverse_bits_len;
use serde::{Deserialize, Serialize};
use slop_challenger::{GrindingChallenger, IopCtx};
use slop_jagged::JaggedPcsProof;
use slop_matrix::dense::RowMajorMatrixView;
use slop_multilinear::{MultilinearPcsVerifier, Point};
use slop_sumcheck::PartialSumcheckProof;
use slop_symmetric::PseudoCompressionFunction;

use crate::{config::ZkmGlobalContext, LogupGkrProof, MachineVerifyingKey, ShardContext};

/// A proof for a shard.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "GC: IopCtx, GC::Challenger: Serialize, Proof: Serialize",
    deserialize = "GC: IopCtx, GC::Challenger: Deserialize<'de>, Proof: Deserialize<'de>"
))]
pub struct ShardProof<GC: IopCtx, Proof> {
    /// The public values.
    pub public_values: Vec<GC::F>,
    /// The commitment to the main trace.
    pub main_commitment: GC::Digest,
    /// The LogUp GKR IOP proof.
    pub logup_gkr_proof: LogupGkrProof<<GC::Challenger as GrindingChallenger>::Witness, GC::EF>,
    /// The zerocheck IOP proof.
    pub zerocheck_proof: PartialSumcheckProof<GC::EF>,
    /// The values of the traces at the final random point.
    pub opened_values: ShardOpenedValues<GC::F, GC::EF>,
    /// The evaluation proof.
    pub evaluation_proof: JaggedPcsProof<GC, Proof>,
}

/// The `ShardProof` type generic in `GC` and `SC`.
pub type ShardContextProof<GC, SC> = ShardProof<GC, <<SC as ShardContext<GC>>::Config as MultilinearPcsVerifier<GC>>::Proof>;

/// The values of the chips in the shard at a random point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardOpenedValues<F, EF> {
    /// For each chip with respect to the canonical ordering, the values of the chip at the random
    /// point.
    pub chips: BTreeMap<String, ChipOpenedValues<F, EF>>,
}

/// The opening values for a given chip at a random point.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize, EF: Serialize"))]
#[serde(bound(deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>"))]
pub struct ChipOpenedValues<F, EF> {
    /// The opening of the preprocessed trace.
    pub preprocessed: AirOpenedValues<EF>,
    /// The opening of the main trace.
    pub main: AirOpenedValues<EF>,
    /// The big-endian bit representation of the degree of the chip.
    pub degree: Point<F>,
}

/// The opening values for a given table section at a random point.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub struct AirOpenedValues<T> {
    /// The opening of the local trace.
    pub local: Vec<T>,
}

impl<T> AirOpenedValues<T> {
    /// Organize the opening values into a vertical pair.
    #[must_use]
    pub fn view(&self) -> RowMajorMatrixView<'_, T>
    where
        T: Clone + Send + Sync,
    {
        RowMajorMatrixView::new_row(&self.local)
    }
}

/// A Merkle tree proof for proving membership in the recursion verifying key set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof<GC: IopCtx> {
    /// The index of the leaf being proven.
    pub index: usize,
    /// The Merkle path.
    pub path: Vec<GC::Digest>,
}

#[derive(Debug)]
/// The error type for Merkle proof verification.
pub struct VcsError;

/// Verify a Merkle proof.
pub fn verify_merkle_proof<GC: IopCtx>(
    proof: &MerkleProof<GC>,
    value: GC::Digest,
    commitment: GC::Digest,
) -> Result<(), VcsError> {
    let MerkleProof { index, path } = proof;

    let mut value = value;
    let mut index = reverse_bits_len(*index, path.len());

    for sibling in path {
        let new_pair = if index.is_multiple_of(2) { [value, *sibling] } else { [*sibling, value] };
        let (_, compressor) = GC::default_hasher_and_compressor();
        value = compressor.compress(new_pair);
        index >>= 1;
    }
    if value != commitment {
        Err(VcsError)
    } else {
        Ok(())
    }
}

/// An intermediate proof which proves the execution of a hypercube verifier.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "GC: IopCtx, GC::Challenger: Serialize, Proof: Serialize",
    deserialize = "GC: IopCtx, GC::Challenger: Deserialize<'de>, Proof: Deserialize<'de>"
))]
pub struct ZKMReduceProof<GC: IopCtx, Proof> {
    /// The verifying key associated with the proof.
    pub vk: MachineVerifyingKey<GC>,
    /// The shard proof representing the shard proof.
    pub proof: ShardProof<GC, Proof>,
    /// The Merkle proof for the recursion verifying key.
    pub vk_merkle_proof: MerkleProof<ZkmGlobalContext>,
}

impl<GC: IopCtx, Proof> std::fmt::Debug for ZKMReduceProof<GC, Proof> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZKMReduceProof").finish_non_exhaustive()
    }
}

/// An intermediate proof which proves the execution of a hypercube verifier, used for the final
/// wrap layer.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "GC: IopCtx, GC::Challenger: Serialize, Proof: Serialize",
    deserialize = "GC: IopCtx, GC::Challenger: Deserialize<'de>, Proof: Deserialize<'de>"
))]
pub struct ZKMWrapProof<GC: IopCtx, Proof> {
    /// The verifying key associated with the proof.
    pub vk: MachineVerifyingKey<GC>,
    /// The shard proof within the wrap proof.
    pub proof: ShardProof<GC, Proof>,
}
