use p3_koala_bear::KoalaBear;
use zkm_core_machine::mips::MipsAir;
use zkm_hypercube::{
    config::{ZkmGlobalContext, ZkmOuterGlobalContext},
    prover::{AirProver, ZkmOuterShardProver, ZkmShardProver},
    shard_context::ShardContextImpl,
};

use crate::{CompressAir, ShrinkAir, WrapAir};

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
type WrapShardContext = ShardContextImpl<
    ZkmOuterGlobalContext,
    zkm_hypercube::config::ZkmOuterStackedPcs,
    WrapAir<KoalaBear>,
>;

pub trait ZKMProverComponents: Send + Sync {
    /// The prover for making Ziren core proofs.
    type CoreProver: AirProver<ZkmGlobalContext, MipsShardContext> + Send + Sync;

    /// The prover for making Ziren recursive (compress) proofs.
    type CompressProver: AirProver<ZkmGlobalContext, CompressShardContext> + Send + Sync;

    /// The prover for shrinking compressed proofs.
    type ShrinkProver: AirProver<ZkmGlobalContext, ShrinkShardContext> + Send + Sync;

    /// The prover for wrapping a shrink proof into an outer (Bn254-bridged) proof, the last STARK
    /// stage before handing off to gnark-ffi for the final PLONK/Groth16 proof.
    type WrapProver: AirProver<ZkmOuterGlobalContext, WrapShardContext> + Send + Sync;
}

pub struct DefaultProverComponents;

impl ZKMProverComponents for DefaultProverComponents {
    type CoreProver = ZkmShardProver<MipsAir<KoalaBear>>;
    type CompressProver = ZkmShardProver<CompressAir<KoalaBear>>;
    type ShrinkProver = ZkmShardProver<ShrinkAir<KoalaBear>>;
    type WrapProver = ZkmOuterShardProver<WrapAir<KoalaBear>>;
}
