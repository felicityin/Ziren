use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use serde::{Deserialize, Serialize};

use p3_air::Air;
use p3_field::FieldAlgebra;
use p3_koala_bear::KoalaBear;

use slop_challenger::IopCtx;
use zkm_primitives::consts::WORD_SIZE;
use zkm_recursion_compiler::ir::{Builder, Felt};
use zkm_hypercube::septic_curve::SepticCurve;
use zkm_hypercube::septic_digest::SepticDigest;
use zkm_hypercube::{
    air::{MachineAir, POSEIDON_NUM_WORDS},
    config::ZkmGlobalContext,
    verifier::{MachineVerifyingKey, ShardProof},
    word::Word,
};

use zkm_recursion_core::{
    air::{RecursionPublicValues, PV_DIGEST_NUM_WORDS, RECURSIVE_PROOF_NUM_PV_ELTS},
    DIGEST_SIZE,
};

use crate::{
    challenger::DuplexChallengerVariable,
    hash::{FieldHasher, FieldHasherVariable},
    machine::assert_recursion_public_values_valid,
    shard::{MachineVerifyingKeyVariable, RecursiveShardVerifier, ShardProofVariable},
    zerocheck::RecursiveVerifierConstraintFolder,
    CircuitConfig,
};
use zkm_recursion_compiler::circuit::CircuitV2Builder;

use super::{
    recursion_public_values_digest, ZKMMerkleProofVerifier, ZKMMerkleProofWitnessValues,
    ZKMMerkleProofWitnessVariable,
};

pub struct ZKMDeferredVerifier<C, A> {
    _phantom: std::marker::PhantomData<(C, A)>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "ShardProof<GC, Proof>: Serialize, <GC as FieldHasher<KoalaBear>>::Digest: Serialize"
))]
#[serde(bound(
    deserialize = "ShardProof<GC, Proof>: Deserialize<'de>, <GC as FieldHasher<KoalaBear>>::Digest: Deserialize<'de>"
))]
pub struct ZKMDeferredWitnessValues<GC: IopCtx<F = KoalaBear> + FieldHasher<KoalaBear>, Proof> {
    pub vks_and_proofs: Vec<(MachineVerifyingKey<GC>, ShardProof<GC, Proof>)>,
    pub vk_merkle_data: ZKMMerkleProofWitnessValues<GC>,
    pub start_reconstruct_deferred_digest: [GC::F; POSEIDON_NUM_WORDS],
    pub zkm_vk_digest: [GC::F; DIGEST_SIZE],
    pub committed_value_digest: [Word<GC::F>; PV_DIGEST_NUM_WORDS],
    pub deferred_proofs_digest: [GC::F; POSEIDON_NUM_WORDS],
    pub end_pc: GC::F,
    pub end_execution_shard: GC::F,
    pub init_addr: [GC::F; 4],
    pub finalize_addr: [GC::F; 4],
    pub is_complete: bool,
}

pub struct ZKMDeferredWitnessVariable<C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>> + crate::hash::KoalaBearFeltSelect> {
    pub vks_and_proofs:
        Vec<(MachineVerifyingKeyVariable<C, ZkmGlobalContext>, ShardProofVariable<C, ZkmGlobalContext>)>,
    pub vk_merkle_data: ZKMMerkleProofWitnessVariable<C, ZkmGlobalContext>,
    pub start_reconstruct_deferred_digest: [Felt<C::F>; POSEIDON_NUM_WORDS],
    pub zkm_vk_digest: [Felt<C::F>; DIGEST_SIZE],
    pub committed_value_digest: [Word<Felt<C::F>>; PV_DIGEST_NUM_WORDS],
    pub deferred_proofs_digest: [Felt<C::F>; POSEIDON_NUM_WORDS],
    pub end_pc: Felt<C::F>,
    pub end_execution_shard: Felt<C::F>,
    pub init_addr: [Felt<C::F>; 4],
    pub finalize_addr: [Felt<C::F>; 4],
    pub is_complete: Felt<C::F>,
}

impl<C, A> ZKMDeferredVerifier<C, A>
where
    C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>> + crate::hash::KoalaBearFeltSelect,
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    /// Verify a batch of deferred proofs.
    ///
    /// Each deferred proof is a recursive proof representing some computation. Namely, every such
    /// proof represents a recursively verified program.
    /// verifier:
    /// - Asserts that each of these proofs is valid as a `compress` proof.
    /// - Asserts that each of these proofs is complete by checking the `is_complete` flag in the
    ///   proof's public values.
    /// - Aggregates the proof information into the accumulated deferred digest.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<C, ZkmGlobalContext, DuplexChallengerVariable<C>, A>,
        input: ZKMDeferredWitnessVariable<C>,
        value_assertions: bool,
    ) {
        let ZKMDeferredWitnessVariable {
            vks_and_proofs,
            vk_merkle_data,
            start_reconstruct_deferred_digest,
            zkm_vk_digest,
            committed_value_digest,
            deferred_proofs_digest,
            end_pc,
            end_execution_shard,
            init_addr,
            finalize_addr,
            is_complete,
        } = input;

        // First, verify the merkle tree proofs.
        let vk_root = vk_merkle_data.root;
        let values = vks_and_proofs.iter().map(|(vk, _)| vk.hash(builder)).collect::<Vec<_>>();
        ZKMMerkleProofVerifier::verify(builder, values, vk_merkle_data, value_assertions);

        let mut deferred_public_values_stream: Vec<Felt<C::F>> =
            (0..RECURSIVE_PROOF_NUM_PV_ELTS).map(|_| builder.uninit()).collect();
        let deferred_public_values: &mut RecursionPublicValues<_> =
            deferred_public_values_stream.as_mut_slice().borrow_mut();

        // Initialize the start of deferred digests.
        deferred_public_values.start_reconstruct_deferred_digest =
            start_reconstruct_deferred_digest;

        // Initialize the consistency check variable.
        let mut reconstruct_deferred_digest: [Felt<C::F>; POSEIDON_NUM_WORDS] =
            start_reconstruct_deferred_digest;

        for (vk, shard_proof) in vks_and_proofs {
            // Initialize a challenger.
            let mut challenger = DuplexChallengerVariable::<C>::new(builder);
            // Observe the vk and start pc.
            vk.observe_into(builder, &mut challenger);

            // Note: `verify_shard` observes the full `public_values` slice itself as the first
            // step of its transcript (matching `zkm_hypercube::verifier::shard::ShardVerifier::
            // verify_shard`), so it must not be pre-observed here -- doing so would desync the
            // in-circuit Fiat-Shamir transcript from the native prover's.
            machine.verify_shard(builder, &vk, &shard_proof, &mut challenger);

            // Get the current public values.
            let current_public_values: &RecursionPublicValues<Felt<C::F>> =
                shard_proof.public_values.as_slice().borrow();
            // Assert that the `vk_root` is the same as the witnessed one.
            for (elem, expected) in current_public_values.vk_root.iter().zip(vk_root.iter()) {
                builder.assert_felt_eq(*elem, *expected);
            }
            // Assert that the public values are valid.
            assert_recursion_public_values_valid::<C, ZkmGlobalContext>(builder, current_public_values);

            // Assert that the proof is complete.
            builder.assert_felt_eq(current_public_values.is_complete, C::F::ONE);

            // Update deferred proof digest
            // poseidon2( current_digest[..8] || pv.zkm_vk_digest[..8] ||
            // pv.committed_value_digest[..32] )
            let mut inputs: [Felt<C::F>; 48] = array::from_fn(|_| builder.uninit());
            inputs[0..DIGEST_SIZE].copy_from_slice(&reconstruct_deferred_digest);

            inputs[DIGEST_SIZE..DIGEST_SIZE + DIGEST_SIZE]
                .copy_from_slice(&current_public_values.zkm_vk_digest);

            for j in 0..PV_DIGEST_NUM_WORDS {
                for k in 0..WORD_SIZE {
                    let element = current_public_values.committed_value_digest[j][k];
                    inputs[j * WORD_SIZE + k + 16] = element;
                }
            }
            reconstruct_deferred_digest = ZkmGlobalContext::hash(builder, &inputs);
        }

        // Set the public values.

        // Set initial_pc and end_pc to be the hinted values.
        deferred_public_values.start_pc = end_pc;
        deferred_public_values.next_pc = end_pc;
        // Deferred proofs don't represent real execution-time progression, so they're spliced
        // into the clk chain as a no-op at the genesis clk value.
        deferred_public_values.initial_clk_high = builder.eval(C::F::ZERO);
        deferred_public_values.initial_clk_low = builder.eval(C::F::ONE);
        deferred_public_values.last_clk_high = builder.eval(C::F::ZERO);
        deferred_public_values.last_clk_low = builder.eval(C::F::ONE);
        deferred_public_values.start_execution_shard = end_execution_shard;
        deferred_public_values.next_execution_shard = end_execution_shard;
        // Set the init and finalize address bits to be the hinted values.
        deferred_public_values.previous_init_addr = init_addr;
        deferred_public_values.last_init_addr = init_addr;
        deferred_public_values.previous_finalize_addr = finalize_addr;
        deferred_public_values.last_finalize_addr = finalize_addr;

        // Set the zkm_vk_digest to be the hinted value.
        deferred_public_values.zkm_vk_digest = zkm_vk_digest;

        // Set the committed value digest to be the hinted value.
        deferred_public_values.committed_value_digest = committed_value_digest;
        // Set the deferred proof digest to be the hinted value.
        deferred_public_values.deferred_proofs_digest = deferred_proofs_digest;

        // Set the exit code to be zero for now.
        deferred_public_values.exit_code = builder.eval(C::F::ZERO);
        // Assign the deferred proof digests.
        deferred_public_values.end_reconstruct_deferred_digest = reconstruct_deferred_digest;
        // Set the is_complete flag.
        deferred_public_values.is_complete = is_complete;
        // Set the `contains_execution_shard` flag.
        deferred_public_values.contains_execution_shard = builder.eval(C::F::ZERO);
        // Set the cumulative sum to zero.
        deferred_public_values.global_cumulative_sum =
            SepticDigest(SepticCurve::convert(SepticDigest::<C::F>::zero().0, |value| {
                builder.eval(value)
            }));
        // Set the vk root from the witness.
        deferred_public_values.vk_root = vk_root;
        // Set the digest according to the previous values.
        deferred_public_values.digest =
            recursion_public_values_digest::<C, ZkmGlobalContext>(builder, deferred_public_values);

        builder.commit_public_values_v2(*deferred_public_values);
    }
}
