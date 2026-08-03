use std::marker::PhantomData;

use p3_air::Air;
use p3_koala_bear::KoalaBear;
use serde::{Deserialize, Serialize};
use zkm_recursion_compiler::ir::{Builder, Felt};
use zkm_hypercube::{air::MachineAir, config::ZkmGlobalContext};

use crate::{
    challenger::DuplexChallengerVariable,
    hash::{FieldHasher, FieldHasherVariable},
    merkle_tree::{verify, MerkleProof, MerkleProofVariable},
    shard::RecursiveShardVerifier,
    witness::{WitnessWriter, Witnessable},
    zerocheck::RecursiveVerifierConstraintFolder,
    CircuitConfig,
};

use super::{
    PublicValuesOutputDigest, ZKMCompressVerifier, ZKMCompressWitnessValues,
    ZKMCompressWitnessVariable,
};

/// A program to verify a batch of recursive proofs and aggregate their public values.
#[derive(Debug, Clone, Copy)]
pub struct ZKMMerkleProofVerifier<C, HV> {
    _phantom: PhantomData<(C, HV)>,
}

/// Witness layout for the compress stage verifier.
pub struct ZKMMerkleProofWitnessVariable<
    C: CircuitConfig<F = KoalaBear>,
    HV: FieldHasherVariable<C>,
> {
    /// The shard proofs to verify.
    pub vk_merkle_proofs: Vec<MerkleProofVariable<C, HV>>,
    /// Hinted values to enable dummy digests.
    pub values: Vec<HV::DigestVariable>,
    /// The root of the merkle tree.
    pub root: HV::DigestVariable,
}

/// An input layout for the reduce verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "HV::Digest: Serialize"))]
#[serde(bound(deserialize = "HV::Digest: Deserialize<'de>"))]
pub struct ZKMMerkleProofWitnessValues<HV: FieldHasher<KoalaBear>> {
    pub vk_merkle_proofs: Vec<MerkleProof<KoalaBear, HV>>,
    pub values: Vec<HV::Digest>,
    pub root: HV::Digest,
}

impl<C, HV> ZKMMerkleProofVerifier<C, HV>
where
    HV: FieldHasherVariable<C>,
    C: CircuitConfig<F = KoalaBear>,
{
    /// Verify (via Merkle tree) that the vkey digests of a proof belong to a specified set (encoded
    /// the Merkle tree proofs in input).
    pub fn verify(
        builder: &mut Builder<C>,
        digests: Vec<HV::DigestVariable>,
        input: ZKMMerkleProofWitnessVariable<C, HV>,
        value_assertions: bool,
    ) {
        let ZKMMerkleProofWitnessVariable { vk_merkle_proofs, values, root } = input;
        for ((proof, value), expected_value) in
            vk_merkle_proofs.into_iter().zip(values).zip(digests)
        {
            verify(builder, proof, value, root);
            if value_assertions {
                HV::assert_digest_eq(builder, expected_value, value);
            } else {
                HV::assert_digest_eq(builder, value, value);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ZKMCompressWithVKeyVerifier<C, A> {
    _phantom: PhantomData<(C, A)>,
}

/// Witness layout for the verifier of the proof shape phase of the compress stage.
pub struct ZKMCompressWithVKeyWitnessVariable<C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>> + crate::hash::KoalaBearFeltSelect> {
    pub compress_var: ZKMCompressWitnessVariable<C>,
    pub merkle_var: ZKMMerkleProofWitnessVariable<C, ZkmGlobalContext>,
}

/// An input layout for the verifier of the proof shape phase of the compress stage.
pub struct ZKMCompressWithVKeyWitnessValues<
    GC: slop_challenger::IopCtx<F = KoalaBear> + FieldHasher<KoalaBear>,
    Proof,
> {
    pub compress_val: ZKMCompressWitnessValues<GC, Proof>,
    pub merkle_val: ZKMMerkleProofWitnessValues<GC>,
}

impl<C, A> ZKMCompressWithVKeyVerifier<C, A>
where
    C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>> + crate::hash::KoalaBearFeltSelect,
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    /// Verify the proof shape phase of the compress stage.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<C, ZkmGlobalContext, DuplexChallengerVariable<C>, A>,
        input: ZKMCompressWithVKeyWitnessVariable<C>,
        value_assertions: bool,
        kind: PublicValuesOutputDigest,
    ) {
        let values = input
            .compress_var
            .vks_and_proofs
            .iter()
            .map(|(vk, _)| vk.hash(builder))
            .collect::<Vec<_>>();
        let vk_root = input.merkle_var.root.map(|x| builder.eval(x));
        ZKMMerkleProofVerifier::verify(builder, values, input.merkle_var, value_assertions);
        ZKMCompressVerifier::verify(builder, machine, input.compress_var, vk_root, kind);
    }
}

impl<C: CircuitConfig<F = KoalaBear, EF = zkm_stark::InnerChallenge, Bit = Felt<KoalaBear>> + crate::hash::KoalaBearFeltSelect>
    Witnessable<C> for ZKMCompressWithVKeyWitnessValues<ZkmGlobalContext, slop_stacked::StackedBasefoldProof<ZkmGlobalContext>>
{
    type WitnessVariable = ZKMCompressWithVKeyWitnessVariable<C>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        ZKMCompressWithVKeyWitnessVariable {
            compress_var: self.compress_val.read(builder),
            merkle_var: self.merkle_val.read(builder),
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.compress_val.write(witness);
        self.merkle_val.write(witness);
    }
}
