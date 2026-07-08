use std::{collections::BTreeSet, fmt::Debug};

use derive_where::derive_where;
use slop_algebra::Field;
use slop_multilinear::MultilinearPcsVerifier;

use crate::{air::MachineAir, Chip, ShardContext};

/// The PCS opening-proof type produced by a shard context's configured PCS verifier.
pub type PcsProof<GC, SC> = <<SC as ShardContext<GC>>::Config as MultilinearPcsVerifier<GC>>::Proof;

/// The shape of the core proof. This and prover setup parameters should entirely determine the
/// verifier circuit.
#[derive_where(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreProofShape<F: Field, A: MachineAir<F>> {
    /// The chips included in the record.
    pub shard_chips: BTreeSet<Chip<F, A>>,
    /// The number of trace cells in the preprocessed traces.
    pub preprocessed_area: usize,
    /// The area of the main traces.
    pub main_area: usize,
    /// The number of columns added to the preprocessed commit to round to the nearest multiple of
    /// `stacking_height`.
    pub preprocessed_padding_cols: usize,
    /// The number of columns added to the main commit to round to the nearest multiple of
    /// `stacking_height`.
    pub main_padding_cols: usize,
}

impl<F, A> Debug for CoreProofShape<F, A>
where
    F: Field + Debug,
    A: MachineAir<F> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProofShape")
            .field("shard_chips", &self.shard_chips.iter().map(MachineAir::name).collect::<BTreeSet<_>>())
            .field("preprocessed_area", &self.preprocessed_area)
            .field("main_area", &self.main_area)
            .field("preprocessed_padding_cols", &self.preprocessed_padding_cols)
            .field("main_padding_cols", &self.main_padding_cols)
            .finish()
    }
}
