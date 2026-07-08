use slop_challenger::IopCtx;
use slop_merkle_tree::Poseidon2KoalaBear16Prover;
use slop_multilinear::MultilinearPcsVerifier;
use slop_stacked::StackedPcsProver;

use crate::{
    config::{ZkmGlobalContext, ZkmStackedPcs},
    shard_context::ShardContextImpl,
    GkrProverImpl, LogupGkrCpuTraceGenerator, ShardVerifier, ZerocheckAir,
};

use super::{DefaultTraceGenerator, ShardProver};

type SC<GC, Verifier, A> = ShardContextImpl<GC, Verifier, A>;

/// A CPU shard prover.
pub type CpuShardProver<GC, Verifier, PcsComponents, A> = ShardProver<GC, SC<GC, Verifier, A>, PcsComponents>;

/// The Merkle-tree prover used to instantiate Ziren's own (KoalaBear-based) stacked-basefold PCS.
pub type ZkmMerkleTreeProver = Poseidon2KoalaBear16Prover;

/// The concrete PCS prover components used by Ziren's own core shard prover.
pub type ZkmInnerPcsProver = StackedPcsProver<ZkmMerkleTreeProver, ZkmGlobalContext>;

/// Ziren's own CPU shard prover, instantiated with its stacked-basefold PCS over `KoalaBear`.
pub type ZkmShardProver<A> = CpuShardProver<ZkmGlobalContext, ZkmStackedPcs, ZkmInnerPcsProver, A>;

impl<GC, Verifier, A, PcsComponents> CpuShardProver<GC, Verifier, PcsComponents, A>
where
    GC: IopCtx,
    Verifier: MultilinearPcsVerifier<GC>,
    PcsComponents: slop_jagged::DefaultJaggedProver<GC, Verifier>,
    A: ZerocheckAir<GC::F, GC::EF>,
{
    /// Create a new CPU prover from a shard verifier.
    #[must_use]
    pub fn new(verifier: ShardVerifier<GC, ShardContextImpl<GC, Verifier, A>>) -> Self {
        let ShardVerifier { jagged_pcs_verifier: pcs_verifier, machine } = verifier;
        let pcs_prover = slop_jagged::JaggedProver::from_verifier(&pcs_verifier);
        let trace_generator = DefaultTraceGenerator::new(machine);
        let logup_gkr_trace_generator = LogupGkrCpuTraceGenerator::default();
        let logup_gkr_prover = GkrProverImpl::new(logup_gkr_trace_generator);

        Self::from_components(trace_generator, logup_gkr_prover, pcs_prover)
    }
}
