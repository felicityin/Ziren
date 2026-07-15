use std::{borrow::Borrow, path::Path, str::FromStr};

use anyhow::Result;
use num_bigint::BigUint;
use p3_field::{FieldAlgebra, PrimeField};
use p3_koala_bear::KoalaBear;
use thiserror::Error;

use slop_challenger::IopCtx;
use zkm_core_executor::subproof::SubproofVerifier;
use zkm_core_machine::{cpu::MAX_CPU_LOG_DEGREE, mips::MipsAir};
use zkm_hypercube::{
    air::{PublicValues, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    config::{default_fri_config, ZkmGlobalContext},
    verifier::{ShardVerifier, ShardVerifierConfigError},
    word::Word,
    ZkmStackedPcs,
};
use zkm_primitives::{consts::WORD_SIZE, io::ZKMPublicValues};

use zkm_recursion_core::air::RecursionPublicValues;
use zkm_recursion_gnark_ffi::{
    Groth16Bn254Proof, Groth16Bn254Prover, PlonkBn254Proof, PlonkBn254Prover,
};

use crate::{
    build::zkm_imm_wrap_vk_mode,
    components::ZKMProverComponents,
    core_max_log_row_count, recursion_max_log_row_count,
    utils::is_recursion_public_values_valid,
    CompressAir, HashableKey, ShrinkAir, ZKMCoreProofData, ZKMProver, ZKMReduceProofWrapper,
    ZKMVerifyingKey, ZKMWrapProof, CORE_LOG_STACKING_HEIGHT, RECURSION_LOG_STACKING_HEIGHT,
};

/// Errors that can occur when verifying a native (KoalaBear) STARK-level Ziren proof.
#[derive(Error, Debug)]
pub enum ZKMVerificationError {
    #[error("empty proof")]
    EmptyProof,
    #[error("first shard is missing CPU")]
    MissingCpuInFirstShard,
    #[error("cpu log degree {0} exceeds the maximum")]
    CpuLogDegreeTooLarge(usize),
    #[error("invalid public values: {0}")]
    InvalidPublicValues(&'static str),
    #[error("invalid verification key")]
    InvalidVerificationKey,
    #[error("too many shards")]
    TooManyShards,
    #[error("shard verification failed: {0}")]
    ShardVerification(String),
}

#[derive(Error, Debug)]
pub enum PlonkVerificationError {
    #[error(
        "the verifying key does not match the inner plonk bn254 proof's committed verifying key"
    )]
    InvalidVerificationKey,
    #[error(
        "the public values in the Ziren proof do not match the public values in the inner plonk bn254 proof"
    )]
    InvalidPublicValues,
}

#[derive(Error, Debug)]
pub enum Groth16VerificationError {
    #[error(
        "the verifying key does not match the inner groth16 bn254 proof's committed verifying key"
    )]
    InvalidVerificationKey,
    #[error(
        "the public values in the Ziren proof do not match the public values in the inner groth16 bn254 proof"
    )]
    InvalidPublicValues,
}

impl<C: ZKMProverComponents> ZKMProver<C> {
    /// Verify a core proof by verifying the shards, verifying lookup bus, verifying that the
    /// shards are contiguous and complete.
    pub fn verify(
        &self,
        proof: &ZKMCoreProofData,
        vk: &ZKMVerifyingKey,
    ) -> Result<(), ZKMVerificationError> {
        // The proof should not be empty.
        if proof.0.is_empty() {
            return Err(ZKMVerificationError::EmptyProof);
        }
        // First shard has a "CPU" constraint.
        let first_shard = proof.0.first().unwrap();
        if !first_shard.opened_values.chips.contains_key("Cpu") {
            return Err(ZKMVerificationError::MissingCpuInFirstShard);
        }

        // CPU log degree bound constraints.
        for shard_proof in proof.0.iter() {
            if let Some(cpu) = shard_proof.opened_values.chips.get("Cpu") {
                let log_degree_cpu = cpu.degree.dimension();
                if log_degree_cpu > MAX_CPU_LOG_DEGREE {
                    return Err(ZKMVerificationError::CpuLogDegreeTooLarge(log_degree_cpu));
                }
            }
        }

        // Shard constraints.
        //
        // Initialization: shard should start at one.
        // Transition: shard should increment by one for each shard.
        let mut current_shard = KoalaBear::ZERO;
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            current_shard += KoalaBear::ONE;
            if public_values.shard != current_shard {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "shard index should be the previous shard index + 1 and start at 1",
                ));
            }
        }

        // Execution shard constraints.
        let mut current_execution_shard = KoalaBear::ZERO;
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            if shard_proof.opened_values.chips.contains_key("Cpu") {
                current_execution_shard += KoalaBear::ONE;
                if public_values.execution_shard != current_execution_shard {
                    return Err(ZKMVerificationError::InvalidPublicValues(
                        "execution shard index should be the previous execution shard index + 1 if cpu exists and start at 1",
                    ));
                }
            }
        }

        // Program counter constraints.
        let mut prev_next_pc = KoalaBear::ZERO;
        for (i, shard_proof) in proof.0.iter().enumerate() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            let contains_cpu = shard_proof.opened_values.chips.contains_key("Cpu");
            if i == 0 && public_values.start_pc != vk.vk.pc_start {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "start_pc != vk.start_pc: program counter should start at vk.start_pc",
                ));
            } else if i != 0 && public_values.start_pc != prev_next_pc {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "start_pc != next_pc_prev: start_pc should equal next_pc_prev for all shards",
                ));
            } else if !contains_cpu && public_values.start_pc != public_values.next_pc {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "start_pc != next_pc: start_pc should equal next_pc for non-cpu shards",
                ));
            } else if contains_cpu && public_values.start_pc == KoalaBear::ZERO {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "start_pc == 0: execution should never start at halted state",
                ));
            } else if i == proof.0.len() - 1 && public_values.next_pc != KoalaBear::ZERO {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "next_pc != 0: execution should have halted",
                ));
            }
            prev_next_pc = public_values.next_pc;
        }

        // Exit code constraints.
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            if public_values.exit_code != KoalaBear::ZERO {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "exit_code != 0: exit code should be zero for all shards",
                ));
            }
        }

        // Memory initialization & finalization constraints.
        let mut last_init_addr_bits_prev = [KoalaBear::ZERO; 32];
        let mut last_finalize_addr_bits_prev = [KoalaBear::ZERO; 32];
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            let contains_memory_init =
                shard_proof.opened_values.chips.contains_key("MemoryGlobalInit");
            let contains_memory_finalize =
                shard_proof.opened_values.chips.contains_key("MemoryGlobalFinalize");
            if public_values.previous_init_addr_bits != last_init_addr_bits_prev {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "previous_init_addr_bits != last_init_addr_bits_prev",
                ));
            } else if public_values.previous_finalize_addr_bits != last_finalize_addr_bits_prev {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "last_init_addr_bits != last_finalize_addr_bits_prev",
                ));
            } else if !contains_memory_init
                && public_values.previous_init_addr_bits != public_values.last_init_addr_bits
            {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "previous_init_addr_bits != last_init_addr_bits",
                ));
            } else if !contains_memory_finalize
                && public_values.previous_finalize_addr_bits
                    != public_values.last_finalize_addr_bits
            {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "previous_finalize_addr_bits != last_finalize_addr_bits",
                ));
            }
            last_init_addr_bits_prev = public_values.last_init_addr_bits;
            last_finalize_addr_bits_prev = public_values.last_finalize_addr_bits;
        }

        // Digest constraints.
        let zero_committed_value_digest = [Word([KoalaBear::ZERO; WORD_SIZE]); PV_DIGEST_NUM_WORDS];
        let zero_deferred_proofs_digest = [KoalaBear::ZERO; POSEIDON_NUM_WORDS];
        let mut committed_value_digest_prev = zero_committed_value_digest;
        let mut deferred_proofs_digest_prev = zero_deferred_proofs_digest;
        for shard_proof in proof.0.iter() {
            let public_values: &PublicValues<Word<_>, _> =
                shard_proof.public_values.as_slice().borrow();
            let contains_cpu = shard_proof.opened_values.chips.contains_key("Cpu");
            if committed_value_digest_prev != zero_committed_value_digest
                && public_values.committed_value_digest != committed_value_digest_prev
            {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "committed_value_digest != committed_value_digest_prev",
                ));
            } else if deferred_proofs_digest_prev != zero_deferred_proofs_digest
                && public_values.deferred_proofs_digest != deferred_proofs_digest_prev
            {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "deferred_proofs_digest != deferred_proofs_digest_prev",
                ));
            } else if !contains_cpu
                && public_values.committed_value_digest != committed_value_digest_prev
            {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "committed_value_digest != committed_value_digest_prev",
                ));
            } else if !contains_cpu
                && public_values.deferred_proofs_digest != deferred_proofs_digest_prev
            {
                return Err(ZKMVerificationError::InvalidPublicValues(
                    "deferred_proofs_digest != deferred_proofs_digest_prev",
                ));
            }
            committed_value_digest_prev = public_values.committed_value_digest;
            deferred_proofs_digest_prev = public_values.deferred_proofs_digest;
        }

        // Verify that the number of shards is not too large.
        if proof.0.len() > 1 << 16 {
            return Err(ZKMVerificationError::TooManyShards);
        }

        // Verify the shard proofs.
        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            CORE_LOG_STACKING_HEIGHT,
            core_max_log_row_count(),
            MipsAir::<KoalaBear>::hypercube_machine(),
        );
        for shard_proof in proof.0.iter() {
            let mut challenger = ZkmGlobalContext::default_challenger();
            vk.vk.observe_into(&mut challenger);
            shard_verifier
                .verify_shard(&vk.vk, shard_proof, &mut challenger)
                .map_err(|e| ZKMVerificationError::ShardVerification(format!("{e:?}")))?;
        }

        Ok(())
    }

    /// Verify a compressed proof.
    pub fn verify_compressed(
        &self,
        proof: &ZKMReduceProofWrapper,
        vk: &ZKMVerifyingKey,
    ) -> Result<(), ZKMVerificationError> {
        let ZKMReduceProofWrapper { vk: compress_vk, proof: shard_proof, vk_merkle_proof } = proof;

        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            RECURSION_LOG_STACKING_HEIGHT,
            recursion_max_log_row_count(),
            CompressAir::<KoalaBear>::compress_machine(),
        );
        let mut challenger = ZkmGlobalContext::default_challenger();
        compress_vk.observe_into(&mut challenger);
        shard_verifier
            .verify_shard(compress_vk, shard_proof, &mut challenger)
            .map_err(|e| ZKMVerificationError::ShardVerification(format!("{e:?}")))?;

        // Validate public values.
        let public_values: &RecursionPublicValues<_> =
            shard_proof.public_values.as_slice().borrow();
        if !is_recursion_public_values_valid(public_values) {
            return Err(ZKMVerificationError::InvalidPublicValues(
                "recursion public values are invalid",
            ));
        }

        if public_values.vk_root != self.recursion_vk_root {
            return Err(ZKMVerificationError::InvalidPublicValues("vk_root mismatch"));
        }

        if self.vk_verification
            && !self.recursion_vk_map.contains_key(&compress_vk.hash_koalabear())
        {
            return Err(ZKMVerificationError::InvalidVerificationKey);
        }

        zkm_hypercube::verifier::verify_merkle_proof(
            vk_merkle_proof,
            compress_vk.hash_koalabear(),
            self.recursion_vk_root,
        )
        .map_err(|_| ZKMVerificationError::InvalidVerificationKey)?;

        // `is_complete` should be 1. In the reduce program, this ensures that the proof is fully
        // reduced.
        if public_values.is_complete != KoalaBear::ONE {
            return Err(ZKMVerificationError::InvalidPublicValues("is_complete is not 1"));
        }

        // Verify that the proof is for the Ziren vkey we are expecting.
        let vkey_hash = vk.hash_koalabear();
        if public_values.zkm_vk_digest != vkey_hash {
            return Err(ZKMVerificationError::InvalidPublicValues("Ziren vk hash mismatch"));
        }

        Ok(())
    }

    /// Verify a shrink proof.
    pub fn verify_shrink(
        &self,
        proof: &ZKMReduceProofWrapper,
        vk: &ZKMVerifyingKey,
    ) -> Result<(), ZKMVerificationError> {
        let ZKMReduceProofWrapper { vk: shrink_vk, proof: shard_proof, vk_merkle_proof } = proof;

        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            RECURSION_LOG_STACKING_HEIGHT,
            recursion_max_log_row_count(),
            ShrinkAir::<KoalaBear>::shrink_machine(),
        );
        let mut challenger = ZkmGlobalContext::default_challenger();
        shrink_vk.observe_into(&mut challenger);
        shard_verifier
            .verify_shard(shrink_vk, shard_proof, &mut challenger)
            .map_err(|e| ZKMVerificationError::ShardVerification(format!("{e:?}")))?;

        // Validate public values.
        let public_values: &RecursionPublicValues<_> =
            shard_proof.public_values.as_slice().borrow();
        if !is_recursion_public_values_valid(public_values) {
            return Err(ZKMVerificationError::InvalidPublicValues(
                "recursion public values are invalid",
            ));
        }

        if public_values.vk_root != self.recursion_vk_root {
            return Err(ZKMVerificationError::InvalidPublicValues("vk_root mismatch"));
        }

        if self.vk_verification && !self.recursion_vk_map.contains_key(&shrink_vk.hash_koalabear())
        {
            return Err(ZKMVerificationError::InvalidVerificationKey);
        }

        zkm_hypercube::verifier::verify_merkle_proof(
            vk_merkle_proof,
            shrink_vk.hash_koalabear(),
            self.recursion_vk_root,
        )
        .map_err(|_| ZKMVerificationError::InvalidVerificationKey)?;

        // Verify that the proof is for the Ziren vkey we are expecting.
        let vkey_hash = vk.hash_koalabear();
        if public_values.zkm_vk_digest != vkey_hash {
            return Err(ZKMVerificationError::InvalidPublicValues("Ziren vk hash mismatch"));
        }

        Ok(())
    }

    /// Verify a wrap bn254 proof.
    ///
    /// Blocked on task #57 (`ZkmOuterGlobalContext`) -- see `lib.rs`'s module doc comment.
    pub fn verify_wrap_bn254(
        &self,
        _proof: &ZKMWrapProof,
        _vk: &ZKMVerifyingKey,
    ) -> Result<(), ZKMVerificationError> {
        unimplemented!(
            "outer/Bn254 wrap verification is blocked on task #57 (ZkmOuterGlobalContext)"
        )
    }

    /// Verifies a PLONK proof using the circuit artifacts in the build directory.
    pub fn verify_plonk_bn254(
        &self,
        proof: &PlonkBn254Proof,
        vk: &ZKMVerifyingKey,
        public_values: &ZKMPublicValues,
        build_dir: &Path,
    ) -> Result<()> {
        let prover = PlonkBn254Prover::new();

        let vkey_hash = BigUint::from_str(&proof.public_inputs[0])?;
        let committed_values_digest = BigUint::from_str(&proof.public_inputs[1])?;

        prover.verify(proof, &vkey_hash, &committed_values_digest, build_dir)?;

        verify_plonk_bn254_public_inputs(vk, public_values, &proof.public_inputs)?;

        Ok(())
    }

    /// Verifies a Groth16 proof using the circuit artifacts in the build directory.
    pub fn verify_groth16_bn254(
        &self,
        proof: &Groth16Bn254Proof,
        vk: &ZKMVerifyingKey,
        public_values: &ZKMPublicValues,
        build_dir: &Path,
    ) -> Result<()> {
        let prover = Groth16Bn254Prover::new();

        let vkey_hash = BigUint::from_str(&proof.public_inputs[0])?;
        let committed_values_digest = BigUint::from_str(&proof.public_inputs[1])?;

        prover.verify(proof, &vkey_hash, &committed_values_digest, build_dir)?;

        verify_groth16_bn254_public_inputs(vk, public_values, &proof.public_inputs)?;

        Ok(())
    }
}

/// Verify the vk_hash and public_values_hash in the public inputs of the PlonkBn254Proof match the
/// expected values.
pub fn verify_plonk_bn254_public_inputs(
    vk: &ZKMVerifyingKey,
    public_values: &ZKMPublicValues,
    plonk_bn254_public_inputs: &[String],
) -> Result<()> {
    let expected_vk_hash = BigUint::from_str(&plonk_bn254_public_inputs[0])?;
    let expected_public_values_hash = BigUint::from_str(&plonk_bn254_public_inputs[1])?;

    let vk_hash = vk.hash_bn254().as_canonical_biguint();
    if vk_hash != expected_vk_hash {
        return Err(PlonkVerificationError::InvalidVerificationKey.into());
    }

    let public_values_hash = public_values.hash_bn254();
    if public_values_hash != expected_public_values_hash {
        return Err(PlonkVerificationError::InvalidPublicValues.into());
    }

    Ok(())
}

/// Verify the vk_hash and public_values_hash in the public inputs of the Groth16Bn254Proof match
/// the expected values.
pub fn verify_groth16_bn254_public_inputs(
    vk: &ZKMVerifyingKey,
    public_values: &ZKMPublicValues,
    groth16_bn254_public_inputs: &[String],
) -> Result<()> {
    let expected_vk_hash = BigUint::from_str(&groth16_bn254_public_inputs[0])?;
    let expected_public_values_hash = BigUint::from_str(&groth16_bn254_public_inputs[1])?;

    let vk_hash = groth16_vk_hash(vk)?;
    if vk_hash != expected_vk_hash {
        return Err(Groth16VerificationError::InvalidVerificationKey.into());
    }

    let public_values_hash = public_values.hash_bn254();
    if public_values_hash != expected_public_values_hash {
        return Err(Groth16VerificationError::InvalidPublicValues.into());
    }

    Ok(())
}

/// Compute the verification key hash committed into Groth16 public inputs.
fn groth16_vk_hash(vk: &ZKMVerifyingKey) -> Result<BigUint> {
    const PART_STARK_VK_BYTES: &[u8] = include_bytes!("../../verifier/bn254-vk/part_stark_vk.bin");

    let vk_hash = vk.hash_bn254();

    if zkm_imm_wrap_vk_mode() {
        let part_stark_vk: zkm_stark::PartStarkVerifyingKey<zkm_recursion_core::stark::KoalaBearPoseidon2Outer> =
            bincode::deserialize(PART_STARK_VK_BYTES)?;
        Ok(zkm_recursion_core::hash_vkey_with_part_vk(&part_stark_vk, vk_hash).as_canonical_biguint())
    } else {
        Ok(vk_hash.as_canonical_biguint())
    }
}

impl<C: ZKMProverComponents> SubproofVerifier for ZKMProver<C> {
    fn verify_deferred_proof(
        &self,
        proof: &ZKMReduceProofWrapper,
        vk: &zkm_hypercube::MachineVerifyingKey<ZkmGlobalContext>,
        vk_hash: [u32; 8],
        committed_value_digest: [u32; 8],
    ) -> Result<(), ShardVerifierConfigError<ZkmGlobalContext, ZkmStackedPcs>> {
        // Check that the vk hash matches the vk hash from the input. The trait's error type
        // (`ShardVerifierConfigError`, a protocol-level PCS/STARK error) has no domain-specific
        // variant for these checks; `InvalidPublicValues` is the closest fit -- these are all,
        // in effect, assertions that the deferred proof's public values don't match what the
        // calling program committed to.
        if vk.hash_u32() != vk_hash {
            return Err(ShardVerifierConfigError::<ZkmGlobalContext, ZkmStackedPcs>::InvalidPublicValues);
        }
        // Check that proof is valid.
        self.verify_compressed(proof, &ZKMVerifyingKey { vk: vk.clone() })
            .map_err(|_| ShardVerifierConfigError::<ZkmGlobalContext, ZkmStackedPcs>::InvalidPublicValues)?;
        // Check that the committed value digest matches the one from syscall.
        let public_values: &RecursionPublicValues<_> =
            proof.proof.public_values.as_slice().borrow();
        if public_values.vk_root != self.recursion_vk_root {
            return Err(ShardVerifierConfigError::<ZkmGlobalContext, ZkmStackedPcs>::InvalidPublicValues);
        }
        for (i, word) in public_values.committed_value_digest.iter().enumerate() {
            if *word != committed_value_digest[i].into() {
                return Err(ShardVerifierConfigError::<ZkmGlobalContext, ZkmStackedPcs>::InvalidPublicValues);
            }
        }
        Ok(())
    }
}
