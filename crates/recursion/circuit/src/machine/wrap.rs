use std::{borrow::Borrow, marker::PhantomData};

use p3_air::Air;
use p3_field::FieldAlgebra;
use p3_koala_bear::KoalaBear;
use zkm_recursion_compiler::circuit::CircuitV2Builder;
use zkm_recursion_compiler::ir::{Builder, Felt};
use zkm_hypercube::{air::MachineAir, config::ZkmGlobalContext};

use crate::{
    challenger::{CanObserveVariable, DuplexChallengerVariable},
    machine::{assert_complete, assert_root_public_values_valid, RootPublicValues},
    shard::RecursiveShardVerifier,
    zerocheck::RecursiveVerifierConstraintFolder,
    CircuitConfig,
};

use super::ZKMCompressWitnessVariable;

/// A program that recursively verifies a proof made by [super::ZKMRootVerifier].
///
/// Note: this only covers the inner (KoalaBear-bit) wrap step. The outer BN254/Groth16-Plonk SNARK
/// wrap commit path (`zkm_imm_wrap_vk_mode`/`commit_recursion_public_values_imm_wrap_vk`) is not
/// yet ported -- FieldHasherVariable<C> for ZkmGlobalContext (hash.rs) is only implemented for
/// `C::Bit = Felt<KoalaBear>`, so the new verification gadgets can't run with the outer config's
/// `Bit = Var<Bn254Fr>` yet. That's a separate, later concern (gnark-ffi/Groth16 circuit
/// regeneration), not a FRI-vs-jagged backend question.
#[derive(Debug, Clone, Copy)]
pub struct ZKMWrapVerifier<C, A> {
    _phantom: PhantomData<(C, A)>,
}

impl<C, A> ZKMWrapVerifier<C, A>
where
    C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>>,
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    /// Verify a batch of recursive proofs and aggregate their public values.
    ///
    /// The compression verifier can aggregate proofs of different kinds:
    /// - Core proofs: proofs which are recursive proof of a batch of Ziren shard proofs. The
    ///   implementation in this function assumes a fixed recursive verifier specified by
    ///   `recursive_vk`.
    /// - Deferred proofs: proofs which are recursive proof of a batch of deferred proofs. The
    ///   implementation in this function assumes a fixed deferred verification program specified by
    ///   `deferred_vk`.
    /// - Compress proofs: these are proofs which refer to a prove of this program. The key for it
    ///   is part of public values will be propagated across all levels of recursion and will be
    ///   checked against itself as in [zkm_prover::Prover] or as in [super::ZKMRootVerifier].
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<C, ZkmGlobalContext, DuplexChallengerVariable<C>, A>,
        input: ZKMCompressWitnessVariable<C>,
    ) {
        // Read input.
        let ZKMCompressWitnessVariable { vks_and_proofs, .. } = input;

        // Assert that there is only one proof, and get the verification key and proof.
        let [(vk, proof)] = vks_and_proofs.try_into().ok().unwrap();

        // Verify the shard proof.

        // Prepare a challenger.
        let mut challenger = DuplexChallengerVariable::<C>::new(builder);

        // Observe the vk and start pc.
        vk.observe_into(builder, &mut challenger);

        // Observe the main commitment and public values.
        challenger.observe_slice(
            builder,
            proof.public_values[0..machine.machine.num_pv_elts()].iter().copied(),
        );

        machine.verify_shard(builder, &vk, &proof, &mut challenger);

        // Get the public values, and assert that they are valid.
        let public_values: &RootPublicValues<Felt<C::F>> = proof.public_values.as_slice().borrow();
        assert_root_public_values_valid::<C, ZkmGlobalContext>(builder, public_values);
        builder.assert_felt_eq(public_values.inner.is_complete, C::F::ONE);
        assert_complete(builder, &public_values.inner, public_values.inner.is_complete);

        // Reflect the public values to the next level.
        builder.commit_public_values_v2(public_values.inner);
    }
}
