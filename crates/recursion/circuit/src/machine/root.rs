use std::marker::PhantomData;

use p3_air::Air;
use p3_field::FieldAlgebra;
use p3_koala_bear::KoalaBear;

use super::{
    PublicValuesOutputDigest, ZKMCompressVerifier, ZKMCompressWithVKeyVerifier,
    ZKMCompressWithVKeyWitnessVariable, ZKMCompressWitnessVariable,
};
use crate::{
    challenger::DuplexChallengerVariable, zerocheck::RecursiveVerifierConstraintFolder,
    shard::RecursiveShardVerifier, CircuitConfig,
};
use zkm_recursion_compiler::ir::{Builder, Felt};
use zkm_recursion_core::DIGEST_SIZE;
use zkm_hypercube::{air::MachineAir, config::ZkmGlobalContext};

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `ZKMCompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct ZKMCompressRootVerifier<C, A> {
    _phantom: PhantomData<(C, A)>,
}

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `ZKMCompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct ZKMCompressRootVerifierWithVKey<C, A> {
    _phantom: PhantomData<(C, A)>,
}

impl<C, A> ZKMCompressRootVerifier<C, A>
where
    C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>> + crate::hash::KoalaBearFeltSelect,
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<C, ZkmGlobalContext, DuplexChallengerVariable<C>, A>,
        input: ZKMCompressWitnessVariable<C>,
        vk_root: [Felt<C::F>; DIGEST_SIZE],
    ) {
        // Assert that the program is complete.
        builder.assert_felt_eq(input.is_complete, C::F::ONE);
        // Verify the proof, as a compress proof.
        ZKMCompressVerifier::verify(
            builder,
            machine,
            input,
            vk_root,
            PublicValuesOutputDigest::Root,
        );
    }
}

impl<C, A> ZKMCompressRootVerifierWithVKey<C, A>
where
    C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>> + crate::hash::KoalaBearFeltSelect,
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<C, ZkmGlobalContext, DuplexChallengerVariable<C>, A>,
        input: ZKMCompressWithVKeyWitnessVariable<C>,
        value_assertions: bool,
        kind: PublicValuesOutputDigest,
    ) {
        // Assert that the program is complete.
        builder.assert_felt_eq(input.compress_var.is_complete, C::F::ONE);
        // Verify the proof, as a compress proof.
        ZKMCompressWithVKeyVerifier::verify(builder, machine, input, value_assertions, kind);
    }
}
