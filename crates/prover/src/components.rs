use p3_koala_bear::KoalaBear;
use zkm_core_machine::mips::MipsAir;
use zkm_hypercube::{
    config::ZkmGlobalContext,
    prover::{AirProver, ZkmShardProver},
    shard_context::ShardContextImpl,
};

use crate::{CompressAir, ShrinkAir};

type MipsShardContext = ShardContextImpl<
    ZkmGlobalContext,
    zkm_hypercube::config::ZkmStackedPcs,
    MipsAir<KoalaBear>,
>;
type CompressShardContext = ShardContextImpl<
    ZkmGlobalContext,
    zkm_hypercube::config::ZkmStackedPcs,
    CompressAir<KoalaBear>,
>;
type ShrinkShardContext = ShardContextImpl<
    ZkmGlobalContext,
    zkm_hypercube::config::ZkmStackedPcs,
    ShrinkAir<KoalaBear>,
>;

pub trait ZKMProverComponents: Send + Sync {
    /// The prover for making Ziren core proofs.
    type CoreProver: AirProver<ZkmGlobalContext, MipsShardContext> + Send + Sync;

    /// The prover for making Ziren recursive (compress) proofs.
    type CompressProver: AirProver<ZkmGlobalContext, CompressShardContext> + Send + Sync;

    /// The prover for shrinking compressed proofs.
    type ShrinkProver: AirProver<ZkmGlobalContext, ShrinkShardContext> + Send + Sync;
}

pub struct DefaultProverComponents;

impl ZKMProverComponents for DefaultProverComponents {
    type CoreProver = ZkmShardProver<MipsAir<KoalaBear>>;
    type CompressProver = ZkmShardProver<CompressAir<KoalaBear>>;
    type ShrinkProver = ZkmShardProver<ShrinkAir<KoalaBear>>;
}
