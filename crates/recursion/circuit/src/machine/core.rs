use std::{array, borrow::Borrow, borrow::BorrowMut, marker::PhantomData};

use itertools::Itertools;
use p3_field::FieldAlgebra;
use p3_koala_bear::KoalaBear;

use serde::{Deserialize, Serialize};
use slop_air::Air;
use slop_challenger::IopCtx;
use zkm_core_machine::mips::{MipsAir, MAX_LOG_NUMBER_OF_SHARDS};

use zkm_hypercube::{
    air::{PublicValues, PV_DIGEST_NUM_WORDS},
    config::ZkmGlobalContext,
    septic_curve::SepticCurve,
    septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    verifier::{MachineVerifyingKey, ShardProof},
    word::Word,
};

use zkm_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, Felt, SymbolicFelt},
};

use zkm_recursion_core::air::{RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS};

use crate::{
    challenger::DuplexChallengerVariable,
    machine::{assert_complete, recursion_public_values_digest},
    shard::{MachineVerifyingKeyVariable, RecursiveShardVerifier, ShardProofVariable},
    zerocheck::RecursiveVerifierConstraintFolder,
    CircuitConfig,
};

pub struct ZKMRecursionWitnessVariable<C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>>> {
    pub vk: MachineVerifyingKeyVariable<C, ZkmGlobalContext>,
    pub shard_proofs: Vec<ShardProofVariable<C, ZkmGlobalContext>>,
    pub is_complete: Felt<C::F>,
    pub is_first_shard: Felt<C::F>,
    pub vk_root: [Felt<C::F>; 8],
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "ShardProof<GC, Proof>: Serialize"))]
#[serde(bound(deserialize = "ShardProof<GC, Proof>: Deserialize<'de>"))]
pub struct ZKMRecursionWitnessValues<GC: IopCtx, Proof> {
    pub vk: MachineVerifyingKey<GC>,
    pub shard_proofs: Vec<ShardProof<GC, Proof>>,
    pub is_complete: bool,
    pub is_first_shard: bool,
    pub vk_root: [GC::F; 8],
}

/// A program for recursively verifying a batch of Ziren proofs.
#[derive(Debug, Clone, Copy)]
pub struct ZKMRecursiveVerifier<C: Config> {
    _phantom: PhantomData<C>,
}

impl<C> ZKMRecursiveVerifier<C>
where
    C: CircuitConfig<F = KoalaBear, Bit = Felt<KoalaBear>>,
{
    /// Verify a batch of Ziren shard proofs and aggregate their public values.
    ///
    /// This program represents a first recursive step in the verification of a Ziren proof
    /// consisting of one or more shards. Each shard proof is verified and its public values are
    /// aggregated into a single set representing the start and end state of the program execution
    /// across all shards.
    ///
    /// # Constraints
    ///
    /// ## Verifying the STARK proofs.
    /// For each shard, the verifier asserts the correctness of the shard proof, which is composed
    /// of verifying the jagged-PCS opening proof and verifying the zerocheck/LogUp-GKR constraints.
    ///
    /// ## Aggregating the shard public values.
    /// See [ZKMProver::verify] for the verification algorithm of a complete Ziren proof. In this
    /// function, we are aggregating several shard proofs and attesting to an aggregated state which
    /// represents all the shards.
    ///
    /// ## The leaf challenger.
    /// A key difference between the recursive tree verification and the complete one in
    /// [ZKMProver::verify] is that the recursive verifier has no way of reconstructing the
    /// challenger only from a part of the shard proof. Therefore, the value of the leaf challenger
    /// is witnessed in the program and the verifier asserts correctness given this challenger.
    /// In the course of the recursive verification, the challenger is reconstructed by observing
    /// the commitments one by one, and in the final step, the challenger is asserted to be the same
    /// as the one witnessed here.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<C, ZkmGlobalContext, DuplexChallengerVariable<C>, MipsAir<C::F>>,
        input: ZKMRecursionWitnessVariable<C>,
    ) where
        MipsAir<C::F>: for<'b> Air<RecursiveVerifierConstraintFolder<'b, C>>,
    {
        // Read input.
        let ZKMRecursionWitnessVariable { vk, shard_proofs, is_complete, is_first_shard, vk_root } =
            input;

        // Initialize shard variables.
        let mut initial_shard: Felt<_> = builder.uninit();
        let mut current_shard: Felt<_> = builder.uninit();

        // Initialize execution shard variables.
        let mut initial_execution_shard: Felt<_> = builder.uninit();
        let mut current_execution_shard: Felt<_> = builder.uninit();

        // Initialize program counter variables.
        let mut start_pc: Felt<_> = builder.uninit();
        let mut current_pc: Felt<_> = builder.uninit();

        // Initialize memory initialization and finalization variables.
        let mut initial_previous_init_addr_bits: [Felt<_>; 32] =
            array::from_fn(|_| builder.uninit());
        let mut initial_previous_finalize_addr_bits: [Felt<_>; 32] =
            array::from_fn(|_| builder.uninit());
        let mut current_init_addr_bits: [Felt<_>; 32] = array::from_fn(|_| builder.uninit());
        let mut current_finalize_addr_bits: [Felt<_>; 32] = array::from_fn(|_| builder.uninit());

        // Initialize the exit code variable.
        let mut exit_code: Felt<_> = builder.uninit();

        // Initialize the public values digest.
        let mut committed_value_digest: [Word<Felt<_>>; PV_DIGEST_NUM_WORDS] =
            array::from_fn(|_| Word(array::from_fn(|_| builder.uninit())));

        // Initialize the deferred proofs digest.
        let mut deferred_proofs_digest: [Felt<_>; 8] = array::from_fn(|_| builder.uninit());

        // Initialize the cumulative sum.
        let mut global_cumulative_sums = Vec::new();

        // Assert that the number of proofs is not zero.
        assert!(!shard_proofs.is_empty());

        // Initialize a flag to denote the first (if any) CPU shard.
        let mut cpu_shard_seen = false;

        // Verify proofs.
        for (i, shard_proof) in shard_proofs.into_iter().enumerate() {
            let contains_cpu = shard_proof.contains_cpu();
            let contains_memory_init = shard_proof.contains_memory_init();
            let contains_memory_finalize = shard_proof.contains_memory_finalize();

            // Get the public values.
            let public_values: &PublicValues<Word<Felt<_>>, Felt<_>> =
                shard_proof.public_values.as_slice().borrow();

            // If this is the first proof in the batch, initialize the variables.
            if i == 0 {
                // Shard.
                initial_shard = public_values.shard;
                current_shard = public_values.shard;

                // Execution shard.
                initial_execution_shard = public_values.execution_shard;
                current_execution_shard = public_values.execution_shard;

                // Program counter.
                start_pc = public_values.start_pc;
                current_pc = public_values.start_pc;

                // Memory initialization & finalization.
                for ((bit, pub_bit), first_bit) in current_init_addr_bits
                    .iter_mut()
                    .zip(public_values.previous_init_addr_bits.iter())
                    .zip(initial_previous_init_addr_bits.iter_mut())
                {
                    *bit = *pub_bit;
                    *first_bit = *pub_bit;
                }
                for ((bit, pub_bit), first_bit) in current_finalize_addr_bits
                    .iter_mut()
                    .zip(public_values.previous_finalize_addr_bits.iter())
                    .zip(initial_previous_finalize_addr_bits.iter_mut())
                {
                    *bit = *pub_bit;
                    *first_bit = *pub_bit;
                }

                // Exit code.
                exit_code = public_values.exit_code;

                // Committed public values digests.
                for (word, first_word) in committed_value_digest
                    .iter_mut()
                    .zip_eq(public_values.committed_value_digest.iter())
                {
                    for (byte, first_byte) in word.0.iter_mut().zip_eq(first_word.0.iter()) {
                        *byte = *first_byte;
                    }
                }

                // Deferred proofs digests.
                for (digest, first_digest) in deferred_proofs_digest
                    .iter_mut()
                    .zip_eq(public_values.deferred_proofs_digest.iter())
                {
                    *digest = *first_digest;
                }

                // First shard constraints. We verify the validity of the `is_first_shard` boolean
                // flag, and make assertions for that are specific to the first shard using that
                // flag.

                // Assert that the shard is boolean.
                builder.assert_felt_eq(is_first_shard * (is_first_shard - C::F::ONE), C::F::ZERO);
                // Assert that if the flag is set to `1`, then the shard index is `1`.
                builder.assert_felt_eq(is_first_shard * (initial_shard - C::F::ONE), C::F::ZERO);
                // Assert that if the flag is set to `0`, then the shard index is not `1`.
                builder.assert_felt_ne(
                    (SymbolicFelt::ONE - is_first_shard) * initial_shard,
                    C::F::ONE,
                );

                // If it's the first shard (which is the first execution shard), then the `start_pc`
                // should be vk.pc_start.
                builder.assert_felt_eq(is_first_shard * (start_pc - vk.pc_start), C::F::ZERO);

                // If it's the first shard, we add the vk's `initial_global_cumulative_sum` to the digest.
                global_cumulative_sums.push(builder.select_global_cumulative_sum(
                    is_first_shard,
                    vk.initial_global_cumulative_sum,
                ));

                // Assert that `init_addr_bits` and `finalize_addr_bits` are zero for the first
                for bit in current_init_addr_bits.iter() {
                    builder.assert_felt_eq(is_first_shard * *bit, C::F::ZERO);
                }
                for bit in current_finalize_addr_bits.iter() {
                    builder.assert_felt_eq(is_first_shard * *bit, C::F::ZERO);
                }
            }

            // Verify the shard.
            //
            // Do not verify the cumulative sum here, since the permutation challenge is shared
            // between all shards.

            // Prepare a challenger.
            let mut challenger = DuplexChallengerVariable::<C>::new(builder);

            // Observe the vk and start pc.
            vk.observe_into(builder, &mut challenger);

            // Note: `verify_shard` observes the full `public_values` slice itself as the first
            // step of its transcript (matching `zkm_hypercube::verifier::shard::ShardVerifier::
            // verify_shard`), so it must not be pre-observed here -- doing so would desync the
            // in-circuit Fiat-Shamir transcript from the native prover's.
            machine.verify_shard(builder, &vk, &shard_proof, &mut challenger);

            // Assert that first shard has a "CPU". Equivalently, assert that if the shard does
            // not have a "CPU", then the current shard is not 1.
            if !contains_cpu {
                builder.assert_felt_ne(current_shard, C::F::ONE);
            }

            // Shard constraints.
            {
                // Assert that the shard of the proof is equal to the current shard.
                builder.assert_felt_eq(current_shard, public_values.shard);

                // Increment the current shard by one.
                current_shard = builder.eval(current_shard + C::F::ONE);
            }

            // Execution shard constraints.
            {
                // If the shard has a "CPU" chip, then the execution shard should be incremented by
                // 1.
                if contains_cpu {
                    // If this is the first time we've seen the CPU, we initialize the initial and
                    // current execution shards.
                    if !cpu_shard_seen {
                        initial_execution_shard = public_values.execution_shard;
                        current_execution_shard = initial_execution_shard;
                        cpu_shard_seen = true;
                    }

                    builder.assert_felt_eq(current_execution_shard, public_values.execution_shard);

                    current_execution_shard = builder.eval(current_execution_shard + C::F::ONE);
                }
            }

            // Program counter constraints.
            {
                // Assert that the start_pc of the proof is equal to the current pc.
                builder.assert_felt_eq(current_pc, public_values.start_pc);

                // If it's not a shard with "CPU", then assert that the start_pc equals the
                // next_pc.
                if !contains_cpu {
                    builder.assert_felt_eq(public_values.start_pc, public_values.next_pc);
                } else {
                    // If it's a shard with "CPU", then assert that the start_pc is not zero.
                    builder.assert_felt_ne(public_values.start_pc, C::F::ZERO);
                }

                // Update current_pc to be the end_pc of the current proof.
                current_pc = public_values.next_pc;
            }

            // Exit code constraints.
            {
                // Assert that the exit code is zero (success) for all proofs.
                builder.assert_felt_eq(exit_code, C::F::ZERO);
            }

            // Memory initialization & finalization constraints.
            {
                // Assert that the MemoryInitialize address bits match the current loop variable.
                for (bit, current_bit) in current_init_addr_bits
                    .iter()
                    .zip_eq(public_values.previous_init_addr_bits.iter())
                {
                    builder.assert_felt_eq(*bit, *current_bit);
                }

                // Assert that the MemoryFinalize address bits match the current loop variable.
                for (bit, current_bit) in current_finalize_addr_bits
                    .iter()
                    .zip_eq(public_values.previous_finalize_addr_bits.iter())
                {
                    builder.assert_felt_eq(*bit, *current_bit);
                }

                // Assert that if MemoryInit is not present, then the address bits are the same.
                if !contains_memory_init {
                    for (prev_bit, last_bit) in public_values
                        .previous_init_addr_bits
                        .iter()
                        .zip_eq(public_values.last_init_addr_bits.iter())
                    {
                        builder.assert_felt_eq(*prev_bit, *last_bit);
                    }
                }

                // Assert that if MemoryFinalize is not present, then the address bits are the
                // same.
                if !contains_memory_finalize {
                    for (prev_bit, last_bit) in public_values
                        .previous_finalize_addr_bits
                        .iter()
                        .zip_eq(public_values.last_finalize_addr_bits.iter())
                    {
                        builder.assert_felt_eq(*prev_bit, *last_bit);
                    }
                }

                // Update the MemoryInitialize address bits.
                for (bit, pub_bit) in
                    current_init_addr_bits.iter_mut().zip(public_values.last_init_addr_bits.iter())
                {
                    *bit = *pub_bit;
                }

                // Update the MemoryFinalize address bits.
                for (bit, pub_bit) in current_finalize_addr_bits
                    .iter_mut()
                    .zip(public_values.last_finalize_addr_bits.iter())
                {
                    *bit = *pub_bit;
                }
            }

            // Digest constraints.
            {
                // // If `committed_value_digest` is not zero, then the current value should be equal
                // to `public_values.committed_value_digest`.

                // Set flags to indicate whether `committed_value_digest` is non-zero. The flags are
                // given by the elements of the array, and they will be used as filters to constrain
                // the equality.
                let mut is_non_zero_flags = vec![];
                for word in committed_value_digest {
                    for byte in word {
                        is_non_zero_flags.push(byte);
                    }
                }

                // Using the flags, we can constrain the equality.
                for is_non_zero in is_non_zero_flags {
                    for (word_current, word_public) in
                        committed_value_digest.into_iter().zip(public_values.committed_value_digest)
                    {
                        for (byte_current, byte_public) in word_current.into_iter().zip(word_public)
                        {
                            builder.assert_felt_eq(
                                is_non_zero * (byte_current - byte_public),
                                C::F::ZERO,
                            );
                        }
                    }
                }

                // If it's not a shard with "CPU", then the committed value digest shouldn't change.
                if !contains_cpu {
                    for (word_d, pub_word_d) in committed_value_digest
                        .iter()
                        .zip(public_values.committed_value_digest.iter())
                    {
                        for (d, pub_d) in word_d.0.iter().zip(pub_word_d.0.iter()) {
                            builder.assert_felt_eq(*d, *pub_d);
                        }
                    }
                }

                // Update the committed value digest.
                for (word_d, pub_word_d) in committed_value_digest
                    .iter_mut()
                    .zip(public_values.committed_value_digest.iter())
                {
                    for (d, pub_d) in word_d.0.iter_mut().zip(pub_word_d.0.iter()) {
                        *d = *pub_d;
                    }
                }

                // Update the exit code.
                exit_code = public_values.exit_code;

                // If `deferred_proofs_digest` is not zero, then the current value should be equal
                // to `public_values.deferred_proofs_digest.

                // Set a flag to indicate whether `deferred_proofs_digest` is non-zero. The flags
                // are given by the elements of the array, and they will be used as filters to
                // constrain the equality.
                let mut is_non_zero_flags = vec![];
                for element in deferred_proofs_digest {
                    is_non_zero_flags.push(element);
                }

                // Using the flags, we can constrain the equality.
                for is_non_zero in is_non_zero_flags {
                    for (deferred_current, deferred_public) in deferred_proofs_digest
                        .iter()
                        .zip(public_values.deferred_proofs_digest.iter())
                    {
                        builder.assert_felt_eq(
                            is_non_zero * (*deferred_current - *deferred_public),
                            C::F::ZERO,
                        );
                    }
                }

                // If it's not a shard with "CPU", then the deferred proofs digest should not
                // change.
                if !contains_cpu {
                    for (d, pub_d) in deferred_proofs_digest
                        .iter()
                        .zip(public_values.deferred_proofs_digest.iter())
                    {
                        builder.assert_felt_eq(*d, *pub_d);
                    }
                }

                // Update the deferred proofs digest.
                deferred_proofs_digest.copy_from_slice(&public_values.deferred_proofs_digest);
            }

            // Verify that the number of shards is not too large, i.e. that for every shard, we
            // have shard < 2^{MAX_LOG_NUMBER_OF_SHARDS}.
            C::range_check_felt(builder, public_values.shard, MAX_LOG_NUMBER_OF_SHARDS);

            // The old FRI backend additionally asserted `log_degree_cpu() <= MAX_CPU_LOG_DEGREE`
            // here, using a plain usize the circuit-side ShardProofVariable carried as shape
            // metadata. The new backend's ChipOpenedValues::degree is itself an in-circuit witness
            // (a Point<Felt<C::F>>, constraint-checked by verify_shard's own height/max_log_row_count
            // bound), not compile-time metadata, so there's no equivalent Rust-level assert to port;
            // the row-count bound is enforced by verify_shard itself instead.

            // Add this shard's global cumulative sum (already fully aggregated over its chips by
            // the native prover, unlike the old FRI backend which required summing per-chip
            // Global-scope digests here) to the running total.
            let shard_global_cumulative_sum = SepticDigest(SepticCurve {
                x: SepticExtension(public_values.global_cumulative_sum_x),
                y: SepticExtension(public_values.global_cumulative_sum_y),
            });
            global_cumulative_sums.push(shard_global_cumulative_sum);
        }

        let global_cumulative_sum = builder.sum_digest_v2(global_cumulative_sums);

        // Assert that the last exit code is zero.
        builder.assert_felt_eq(exit_code, C::F::ZERO);

        // Write all values to the public values struct and commit to them.
        {
            // Compute the vk digest.
            let vk_digest = vk.hash(builder);

            // Collect the deferred proof digests.
            let zero: Felt<_> = builder.eval(C::F::ZERO);
            let start_deferred_digest = [zero; 8];
            let end_deferred_digest = [zero; 8];

            // Initialize the public values we will commit to.
            let mut recursion_public_values_stream = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
            let recursion_public_values: &mut RecursionPublicValues<_> =
                recursion_public_values_stream.as_mut_slice().borrow_mut();
            recursion_public_values.committed_value_digest = committed_value_digest;
            recursion_public_values.deferred_proofs_digest = deferred_proofs_digest;
            recursion_public_values.start_pc = start_pc;
            recursion_public_values.next_pc = current_pc;
            recursion_public_values.start_shard = initial_shard;
            recursion_public_values.next_shard = current_shard;
            recursion_public_values.start_execution_shard = initial_execution_shard;
            recursion_public_values.next_execution_shard = current_execution_shard;
            recursion_public_values.previous_init_addr_bits = initial_previous_init_addr_bits;
            recursion_public_values.last_init_addr_bits = current_init_addr_bits;
            recursion_public_values.previous_finalize_addr_bits =
                initial_previous_finalize_addr_bits;
            recursion_public_values.last_finalize_addr_bits = current_finalize_addr_bits;
            recursion_public_values.zkm_vk_digest = vk_digest;
            recursion_public_values.global_cumulative_sum = global_cumulative_sum;
            recursion_public_values.start_reconstruct_deferred_digest = start_deferred_digest;
            recursion_public_values.end_reconstruct_deferred_digest = end_deferred_digest;
            recursion_public_values.exit_code = exit_code;
            recursion_public_values.is_complete = is_complete;
            // Set the contains an execution shard flag.
            recursion_public_values.contains_execution_shard =
                builder.eval(C::F::from_bool(cpu_shard_seen));
            recursion_public_values.vk_root = vk_root;

            // Calculate the digest and set it in the public values.
            recursion_public_values.digest =
                recursion_public_values_digest::<C, ZkmGlobalContext>(builder, recursion_public_values);

            assert_complete(builder, recursion_public_values, is_complete);

            builder.commit_public_values_v2(*recursion_public_values);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use p3_field::FieldAlgebra;
    use p3_koala_bear::KoalaBear;

    use slop_basefold::FriConfig;
    use slop_challenger::IopCtx;
    use zkm_core_executor::{Executor, Program};
    use zkm_core_machine::mips::MipsAir;
    use zkm_hypercube::{
        config::ZkmGlobalContext,
        prover::{AirProver, ProverSemaphore, ZkmShardProver},
        ShardVerifier,
    };
    use zkm_recursion_compiler::{circuit::AsmConfig, ir::Builder};
    use zkm_stark::{InnerChallenge, InnerVal, ZKMCoreOpts};

    use crate::{
        shard::RecursiveShardVerifier,
        utils::tests::run_test_recursion,
        witness::{WitnessBlock, Witnessable},
    };

    use super::{ZKMRecursionWitnessValues, ZKMRecursiveVerifier};

    type F = InnerVal;
    type EF = InnerChallenge;
    type C = AsmConfig<F, EF>;

    /// The log2 of the number of rows each stacked-PCS column is grouped into. Must match the
    /// value the real shard proof below was produced with -- mirrors
    /// `zkm_core_machine::utils::prove::ZKM_LOG_STACKING_HEIGHT` (crate-private), kept in sync by
    /// convention.
    const CORE_LOG_STACKING_HEIGHT: u32 = 4;

    /// A FRI config sized for fast correctness testing, not real security: a single query and no
    /// grinding. `verify_shard`'s basefold FRI-query verification does one round of Poseidon2
    /// Merkle-path hashing per query per commit-phase round, which is what dominates the
    /// in-circuit row count -- at the real `default_fri_config()` (94 queries, 16 grinding bits)
    /// that pushes the verifier circuit well past the recursion machine's default shard size.
    /// This must be used consistently for both producing the real MIPS shard proof below and
    /// constructing the circuit-side verifier that checks it.
    fn test_fri_config() -> FriConfig<KoalaBear> {
        FriConfig::new(1, 1, 0)
    }

    /// Verifies a real, honestly-produced MIPS shard proof through the in-circuit
    /// `RecursiveShardVerifier`/`ZKMRecursiveVerifier` gadgets (the phase 3.2/3.3
    /// zerocheck/LogUp-GKR/jagged/basefold verifier chain), proving the resulting constraint
    /// graph is well-formed and correctly handled end to end by the DSL/IR compiler.
    ///
    /// Currently `#[ignore]`d: `FIBONACCI_ELF` executes a `HALT` syscall, which triggers a
    /// currently-unresolved native (not circuit-specific) `GkrVerificationFailed
    /// (CumulativeSumMismatch(..))` bug that reproduces on any program using a real `SYSCALL`
    /// instruction -- see the detailed writeup on `run_test_halt_only_smoke` in
    /// `crates/core/machine/src/utils/prove.rs`. That bug must be fixed first; this test is a
    /// second, independent reproduction (via `Witnessable`/circuit verification rather than
    /// `ShardVerifier::verify_shard` directly) worth re-enabling once it is.
    #[test]
    #[ignore = "blocked on a native (non-circuit) CumulativeSumMismatch bug on any program using a real SYSCALL instruction -- see run_test_halt_only_smoke in crates/core/machine/src/utils/prove.rs"]
    fn test_verify_real_core_shard_proof() {
        let program = Program::from(test_artifacts::FIBONACCI_ELF).unwrap();
        let opts = ZKMCoreOpts::default();

        let mut runtime = Executor::new(program.clone(), opts);
        runtime.run().unwrap();
        // `runtime.record` is the raw, still-accumulating working record: `Executor::execute`
        // only back-fills the real `PublicValues` (shard index, start/next pc, etc.) into
        // `runtime.records` (plural) when it flushes a shard via `bump_record`. For a small
        // single-shard program like this one, that's exactly one record.
        assert_eq!(runtime.records.len(), 1, "expected fibonacci to execute as exactly one shard");
        let mut record = runtime.records.remove(0);
        // Work around a currently-unrelated gap in `zkm_core_executor::Executor`: unlike
        // `public_values.execution_shard` (correctly back-filled from `state.current_shard`,
        // 1-indexed), `public_values.shard` itself is never assigned anywhere in the executor and
        // stays at its `Default` value of `0`. `ZKMRecursiveVerifier::verify` (this crate)
        // expects 1-indexed shards, matching `execution_shard`'s convention, so patch it here.
        // This field isn't read or constrained anywhere in trace generation, so setting it after
        // execution is safe.
        record.public_values.shard = 1;
        // `Executor::execute` never back-fills `initial_timestamp`/`last_timestamp` either
        // (unlike `start_pc`/`next_pc`, which it does set from the same events) -- mirrors the
        // `state.initial_timestamp`/`state.last_timestamp` computation in
        // `zkm_core_machine::utils::prove::prove_with_context`'s reference flow. These anchor the
        // CPU chip's `LookupKind::State` chain boundary in `eval_public_values`.
        let first_cpu_event = record.cpu_events.first().unwrap();
        let last_cpu_event = record.cpu_events.last().unwrap();
        record.public_values.initial_timestamp = first_cpu_event.clk;
        record.public_values.last_timestamp =
            last_cpu_event.clk + 5 + last_cpu_event.num_extra_cycles;
        // `Executor::run`/`execute` never calls chip-level `generate_dependencies`, so
        // cross-chip-derived public values that depend on the actual event contents --
        // `GlobalChip`'s `global_count`/`global_cumulative_sum_{x,y}` and
        // `MemoryGlobalChip`'s `global_init_count`/`global_finalize_count` -- are left at their
        // `Default` (zero) values. The real proving pipeline
        // (`zkm_core_machine::utils::prove::prove_with_context`) always runs this step before
        // proving; without it, the LogUp GKR public-values boundary check (which is anchored to
        // these fields) disagrees with the real interactions recorded in the trace.
        MipsAir::<KoalaBear>::hypercube_machine()
            .generate_dependencies(std::iter::once(&mut record), None)
            .unwrap();
        // Likewise, `previous_init_addr_bits`/`last_init_addr_bits` (and the `finalize`
        // counterparts) are normally back-filled by the deferred-event splitting machinery in
        // `ExecutionRecord::defer`/`split`, which the real proving pipeline always runs before
        // proving but which this simplified single-shard test never invokes. Since this is the
        // only (and therefore also the first and last) shard, there is no earlier/later memory
        // chain to continue from or into, so `previous_*_addr_bits` correctly stay all-zero;
        // only `last_*_addr_bits` (the address of the sorted chain's final event) need
        // computing here, mirroring `ExecutionRecord::defer`'s per-chunk bit computation.
        if let Some(last) = record.global_memory_initialize_events.iter().max_by_key(|e| e.addr) {
            record.public_values.last_init_addr_bits =
                core::array::from_fn(|i| (last.addr >> i) & 1);
        }
        if let Some(last) = record.global_memory_finalize_events.iter().max_by_key(|e| e.addr) {
            record.public_values.last_finalize_addr_bits =
                core::array::from_fn(|i| (last.addr >> i) & 1);
        }

        eprintln!(
            "DEBUG counts: cpu={} init={} finalize={} global_lookup={} precompile_kinds={}",
            record.cpu_events.len(),
            record.global_memory_initialize_events.len(),
            record.global_memory_finalize_events.len(),
            record.global_lookup_events.len(),
            record.precompile_events.len(),
        );
        eprintln!("DEBUG public_values = {:#?}", record.public_values);

        let fri_config = test_fri_config();
        let max_log_row_count = opts.shard_size.ilog2() as usize;
        let native_machine = MipsAir::<KoalaBear>::hypercube_machine();
        let shard_prover = ZkmShardProver::<MipsAir<KoalaBear>>::new(
            ShardVerifier::from_basefold_parameters(
                fri_config,
                CORE_LOG_STACKING_HEIGHT,
                max_log_row_count,
                native_machine,
            ),
        );

        let setup_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (vk, proof, _permit) = setup_rt.block_on(shard_prover.setup_and_prove_shard(
            Arc::new(program),
            record,
            None,
            ProverSemaphore::new(1),
        ));

        // Sanity-check the proof natively before feeding it into the in-circuit verifier: this
        // isolates whether a failure belongs to the circuit gadgets or to the record/proof
        // itself, since `eval_public_values` (the LogUp GKR public-values boundary check) is the
        // same generic code used by both the native and circuit-side constraint folders.
        let native_shard_verifier = ShardVerifier::from_basefold_parameters(
            fri_config,
            CORE_LOG_STACKING_HEIGHT,
            max_log_row_count,
            MipsAir::<KoalaBear>::hypercube_machine(),
        );
        let mut native_challenger = ZkmGlobalContext::default_challenger();
        vk.observe_into(&mut native_challenger);
        native_shard_verifier
            .verify_shard(&vk, &proof, &mut native_challenger)
            .expect("native shard verification should succeed for an honestly-produced proof");

        let witness_values = ZKMRecursionWitnessValues {
            vk,
            shard_proofs: vec![proof],
            is_complete: false,
            is_first_shard: true,
            vk_root: [KoalaBear::ZERO; 8],
        };

        let mut witness_stream = Vec::<WitnessBlock<C>>::new();
        Witnessable::<C>::write(&witness_values, &mut witness_stream);

        let mut builder = Builder::<C>::default();
        let input = Witnessable::<C>::read(&witness_values, &mut builder);

        let machine = RecursiveShardVerifier::from_basefold_parameters(
            fri_config,
            CORE_LOG_STACKING_HEIGHT,
            max_log_row_count,
            MipsAir::<KoalaBear>::hypercube_machine(),
        );

        ZKMRecursiveVerifier::<C>::verify(&mut builder, &machine, input);

        run_test_recursion(builder.into_operations(), witness_stream);
    }
}
