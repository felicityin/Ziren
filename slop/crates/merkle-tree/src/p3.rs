use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use serde::{Deserialize, Serialize};
use slop_algebra::Field;
use slop_baby_bear::{baby_bear_poseidon2::BabyBearDegree4Duplex, BabyBear};
use slop_bn254::{Bn254Fr, BNGC, OUTER_DIGEST_SIZE};
use slop_challenger::IopCtx;
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};

use crate::MerkleTreeTcs;

#[derive(Clone)]
pub struct FieldMerkleTreeProver<P, PW, GC: IopCtx, const DIGEST_ELEMS: usize> {
    pub(crate) tcs: Arc<MerkleTreeTcs<GC>>,
    _phantom: PhantomData<(P, PW)>,
}

pub type BnProver<F, EF> = FieldMerkleTreeProver<F, Bn254Fr, BNGC<F, EF>, OUTER_DIGEST_SIZE>;

impl<P, PW, GC: IopCtx, const DIGEST_ELEMS: usize> std::fmt::Debug
    for FieldMerkleTreeProver<P, PW, GC, DIGEST_ELEMS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FieldMerkleTreeProver")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "[W; DIGEST_ELEMS]: Serialize",
    deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>"
))]
pub struct FieldMerkleTreeDigests<W, const DIGEST_ELEMS: usize> {
    pub digest_layers: Arc<Vec<Vec<[W; DIGEST_ELEMS]>>>,
}

impl<P, PW, GC: IopCtx, const DIGEST_ELEMS: usize> FieldMerkleTreeProver<P, PW, GC, DIGEST_ELEMS> {
    #[inline]
    pub fn new(tcs: MerkleTreeTcs<GC>) -> Self {
        Self { tcs: Arc::new(tcs), _phantom: PhantomData }
    }
}

pub type Poseidon2BabyBear16Prover = FieldMerkleTreeProver<
    <BabyBear as Field>::Packing,
    <BabyBear as Field>::Packing,
    BabyBearDegree4Duplex,
    8,
>;

pub type Poseidon2KoalaBear16Prover = FieldMerkleTreeProver<
    <KoalaBear as Field>::Packing,
    <KoalaBear as Field>::Packing,
    KoalaBearDegree4Duplex,
    8,
>;

impl<GC: IopCtx, P, PW, const DIGEST_ELEMS: usize> Default
    for FieldMerkleTreeProver<P, PW, GC, DIGEST_ELEMS>
{
    fn default() -> Self {
        Self::new(MerkleTreeTcs::default())
    }
}

// Sync trait implementations are in p3sync.rs
#[path = "p3sync.rs"]
mod p3sync;
