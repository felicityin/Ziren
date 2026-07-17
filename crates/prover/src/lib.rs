//! An end-to-end-prover implementation for the Ziren zkVM.
//!
//! Separates the proof generation process into multiple stages:
//!
//! 1. Generate shard proofs which split up and prove the valid execution of a MIPS program.
//! 2. Compress shard proofs into a single shard proof.
//! 3. Wrap the shard proof into a SNARK-friendly field.
//! 4. Wrap the last shard proof, proven over the SNARK-friendly field, into a PLONK proof.
//!
//! The wrap stage (3-4, `wrap_bn254`/`wrap_plonk_bn254`/`wrap_groth16_bn254`/
//! `wrap_dvsnark_bn254`) is stubbed with `unimplemented!()`: it needs a Bn254-bridged `IopCtx`
//! (`ZkmOuterGlobalContext`) that doesn't exist yet (task #57, explicitly deferred by the user
//! along with VK artifact regeneration).
//!
//! This pass also drops the shape-quantization caches the old FRI-era prover used
//! (`lift_programs_lru`/`join_programs_map`, keyed by `ZKMRecursionShape`/`ZKMCompressWithVkeyShape`
//! wrapper types that no longer exist -- see `shapes.rs`) and the `CoreShapeConfig`-driven
//! preprocessed-shape fixing (deleted in task #24). Recursion/compress/shrink programs are now
//! compiled fresh on every call instead of served from a warm cache; slower, not less correct.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::new_without_default)]
#![allow(clippy::collapsible_else_if)]

pub mod build;
pub mod components;
pub mod shapes;
pub mod types;
pub mod utils;
pub mod verify;

use std::{borrow::Borrow, collections::BTreeMap, env, path::Path, sync::Arc};

use p3_field::{FieldAlgebra, PrimeField32};
use p3_koala_bear::KoalaBear;
use tracing::instrument;
use zkm_core_executor::{ExecutionError, ExecutionReport, Executor, Program, ZKMContext};
use zkm_core_machine::{
    io::ZKMStdin,
    mips::MipsAir,
    utils::{prove_with_context, ZKMCoreProverError},
};
use zkm_hypercube::{
    config::{compressed_fri_config, default_fri_config, ultra_compressed_fri_config, ZkmGlobalContext},
    prover::{AirProver, ProverSemaphore, ZkmShardProver},
    verifier::ShardVerifier,
    word::Word,
    ZKMReduceProof, DIGEST_SIZE,
};
use zkm_primitives::{hash_deferred_proof, io::ZKMPublicValues};
use zkm_recursion_circuit::{
    challenger::DuplexChallengerVariable,
    hash::FieldHasher,
    machine::{
        PublicValuesOutputDigest, ZKMCompressRootVerifierWithVKey, ZKMCompressWithVKeyVerifier,
        ZKMCompressWithVKeyWitnessValues, ZKMCompressWitnessValues, ZKMDeferredVerifier,
        ZKMDeferredWitnessValues, ZKMMerkleProofWitnessValues, ZKMRecursionWitnessValues,
        ZKMRecursiveVerifier,
    },
    merkle_tree::MerkleTree,
    shard::RecursiveShardVerifier,
    witness::Witnessable,
};
use zkm_recursion_compiler::{circuit::AsmCompiler, config::InnerConfig, ir::Builder};
use zkm_recursion_core::{
    air::RecursionPublicValues,
    machine::RecursionAir,
    shape::{RecursionShape, RecursionShapeConfig},
    stark::KoalaBearPoseidon2Outer,
    RecursionProgram, Runtime as RecursionRuntime,
};
pub use zkm_recursion_gnark_ffi::proof::{DvSnarkBn254Proof, Groth16Bn254Proof, PlonkBn254Proof};
use zkm_stark::{inner_perm, ZKMCoreOpts, ZKMProverOpts};

pub use types::*;
use utils::words_to_bytes;

use components::{DefaultProverComponents, ZKMProverComponents};

pub use zkm_core_machine::ZKM_CIRCUIT_VERSION;

/// The configuration for the core prover.
pub type CoreSC = ZkmGlobalContext;

/// The configuration for the inner (compress/shrink) prover.
pub type InnerSC = ZkmGlobalContext;

/// The configuration for the outer (Bn254 wrap) prover. Blocked on task #57
/// (`ZkmOuterGlobalContext`) -- see the module doc comment.
pub type OuterSC = KoalaBearPoseidon2Outer;

const COMPRESS_DEGREE: usize = 3;
const SHRINK_DEGREE: usize = 3;
const WRAP_DEGREE: usize = 9;

pub const REDUCE_BATCH_SIZE: usize = 2;

pub type CompressAir<F> = RecursionAir<F, COMPRESS_DEGREE>;
pub type ShrinkAir<F> = RecursionAir<F, SHRINK_DEGREE>;
pub type WrapAir<F> = RecursionAir<F, WRAP_DEGREE>;

/// The max log row count the core machine's jagged PCS is configured for. Fixed independently
/// of `ZKMCoreOpts::shard_size` (the executor's cycle-count ceiling) -- see
/// `zkm_stark::CORE_MAX_LOG_ROW_COUNT`'s doc comment for why these two are decoupled.
fn core_max_log_row_count() -> usize {
    zkm_stark::CORE_MAX_LOG_ROW_COUNT
}

/// The max log row count the recursion (compress/shrink) machines' jagged PCS is configured for.
fn recursion_max_log_row_count() -> usize {
    ZKMCoreOpts::recursion().shard_size.ilog2() as usize
}

/// Log2 of the stacked-PCS grouping height for a machine whose jagged PCS is configured for
/// `max_log_row_count`. Mirrors `zkm_core_machine::utils::prove::stacking_height_for`; kept in
/// sync with it by convention (it isn't reachable from this crate).
fn stacking_height_for(max_log_row_count: usize) -> u32 {
    (max_log_row_count as u32).saturating_sub(1)
}

/// An end-to-end prover implementation for the Ziren zkVM.
pub struct ZKMProver<C: ZKMProverComponents = DefaultProverComponents> {
    /// The machine used for proving the core step.
    pub core_prover: C::CoreProver,

    /// The machine used for proving the recursive and reduction steps.
    pub compress_prover: C::CompressProver,

    /// The machine used for proving the shrink step.
    pub shrink_prover: C::ShrinkProver,

    /// The root of the allowed recursion verification keys.
    pub recursion_vk_root: <ZkmGlobalContext as FieldHasher<KoalaBear>>::Digest,

    /// The allowed VKs and their corresponding indices.
    pub recursion_vk_map: BTreeMap<<ZkmGlobalContext as FieldHasher<KoalaBear>>::Digest, usize>,

    /// The Merkle tree for the allowed VKs.
    pub recursion_vk_tree: MerkleTree<KoalaBear, ZkmGlobalContext>,

    /// The recursion shape configuration.
    pub compress_shape_config: Option<RecursionShapeConfig<KoalaBear, CompressAir<KoalaBear>>>,

    /// Whether to verify verification keys.
    pub vk_verification: bool,
}

impl ZKMProver<DefaultProverComponents> {
    /// Initializes a new [ZKMProver].
    #[instrument(name = "initialize prover", level = "debug", skip_all)]
    pub fn new() -> Self {
        Self::uninitialized()
    }

    /// Creates a new [ZKMProver].
    pub fn uninitialized() -> Self {
        let core_prover = ZkmShardProver::<MipsAir<KoalaBear>>::new(
            ShardVerifier::from_basefold_parameters(
                default_fri_config(),
                stacking_height_for(core_max_log_row_count()),
                core_max_log_row_count(),
                MipsAir::<KoalaBear>::hypercube_machine(),
            ),
        );

        let compress_prover = ZkmShardProver::<CompressAir<KoalaBear>>::new(
            ShardVerifier::from_basefold_parameters(
                compressed_fri_config(),
                stacking_height_for(recursion_max_log_row_count()),
                recursion_max_log_row_count(),
                CompressAir::<KoalaBear>::compress_machine(),
            ),
        );

        let shrink_prover = ZkmShardProver::<ShrinkAir<KoalaBear>>::new(
            ShardVerifier::from_basefold_parameters(
                ultra_compressed_fri_config(),
                stacking_height_for(recursion_max_log_row_count()),
                recursion_max_log_row_count(),
                ShrinkAir::<KoalaBear>::shrink_machine(),
            ),
        );

        let vk_verification =
            env::var("VERIFY_VK").map(|v| v.eq_ignore_ascii_case("true")).unwrap_or(true);

        tracing::debug!("vk verification: {}", vk_verification);

        // Read the allowed VK set. Regenerate `vk_map.bin` when the Ziren circuit is updated
        // (out of scope for this migration pass -- see `shapes.rs`).
        let allowed_vk_map: BTreeMap<[KoalaBear; DIGEST_SIZE], usize> = if vk_verification {
            bincode::deserialize(include_bytes!("../vk_map.bin")).unwrap()
        } else {
            bincode::deserialize(include_bytes!("../dummy_vk_map.bin")).unwrap()
        };

        let (root, merkle_tree) = MerkleTree::commit(allowed_vk_map.keys().copied().collect());

        let compress_shape_config = env::var("FIX_RECURSION_SHAPES")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(true)
            .then_some(RecursionShapeConfig::default());

        Self {
            core_prover,
            compress_prover,
            shrink_prover,
            recursion_vk_root: root,
            recursion_vk_map: allowed_vk_map,
            recursion_vk_tree: merkle_tree,
            compress_shape_config,
            vk_verification,
        }
    }

    /// Fully initializes the programs, proving keys, and verifying keys that are normally
    /// lazily initialized. TODO: remove this.
    pub fn initialize(&mut self) {}
}

impl<C: ZKMProverComponents> ZKMProver<C> {
    /// Creates a proving key and a verifying key for a given MIPS ELF.
    #[instrument(name = "setup", level = "debug", skip_all)]
    pub fn setup(&self, elf: &[u8]) -> (ZKMProvingKey, Program, ZKMVerifyingKey) {
        let program = self.get_program(elf).unwrap();
        let setup_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (_, vk) = setup_rt
            .block_on(self.core_prover.setup(Arc::new(program.clone()), ProverSemaphore::new(1)));
        let vk = ZKMVerifyingKey { vk };
        let pk = ZKMProvingKey { elf: elf.to_vec(), vk: vk.clone() };
        (pk, program, vk)
    }

    /// Get a program for the given ELF.
    pub fn get_program(&self, elf: &[u8]) -> eyre::Result<Program> {
        Ok(Program::from(elf).unwrap())
    }

    /// Generate a proof of a Ziren program with the specified inputs.
    #[instrument(name = "execute", level = "info", skip_all)]
    pub fn execute<'a>(
        &'a self,
        elf: &[u8],
        stdin: &ZKMStdin,
        mut context: ZKMContext<'a>,
    ) -> Result<(ZKMPublicValues, ExecutionReport), ExecutionError> {
        context.subproof_verifier = Some(self);
        let program = self.get_program(elf).unwrap();
        let opts = ZKMCoreOpts::default();
        let mut runtime = Executor::with_context(program, opts, context);
        runtime.write_vecs(&stdin.buffer);
        for (proof, vkey) in stdin.proofs.iter() {
            runtime.write_proof(proof.clone(), vkey.clone());
        }
        runtime.run_fast()?;
        Ok((ZKMPublicValues::from(&runtime.state.public_values_stream), runtime.report))
    }

    /// Generate shard proofs which split up and prove the valid execution of a MIPS program with
    /// the core prover. Uses the provided context.
    #[instrument(name = "prove_core", level = "info", skip_all)]
    pub fn prove_core<'a>(
        &'a self,
        program: Program,
        stdin: &ZKMStdin,
        opts: ZKMProverOpts,
        mut context: ZKMContext<'a>,
    ) -> Result<ZKMCoreProof, ZKMCoreProverError> {
        context.subproof_verifier = Some(self);
        let (shard_proofs, public_values_stream, cycles, _vk) =
            prove_with_context(program, stdin, opts.core_opts, context)?;
        Self::check_for_high_cycles(cycles);
        let public_values = ZKMPublicValues::from(&public_values_stream);
        Ok(ZKMCoreProof {
            proof: ZKMCoreProofData(shard_proofs),
            stdin: stdin.clone(),
            public_values,
            cycles,
        })
    }

    /// Builds the circuit-side shard verifier used to construct in-circuit verification programs
    /// for shards produced by `self.core_prover`.
    fn core_circuit_verifier(
        &self,
    ) -> RecursiveShardVerifier<
        InnerConfig,
        ZkmGlobalContext,
        DuplexChallengerVariable<InnerConfig>,
        MipsAir<KoalaBear>,
    > {
        RecursiveShardVerifier::from_basefold_parameters(
            default_fri_config(),
            stacking_height_for(core_max_log_row_count()),
            core_max_log_row_count(),
            self.core_prover.machine().clone(),
        )
    }

    /// Builds the circuit-side shard verifier used to construct in-circuit verification programs
    /// for shards produced by `self.compress_prover`.
    fn compress_circuit_verifier(
        &self,
    ) -> RecursiveShardVerifier<
        InnerConfig,
        ZkmGlobalContext,
        DuplexChallengerVariable<InnerConfig>,
        CompressAir<KoalaBear>,
    > {
        RecursiveShardVerifier::from_basefold_parameters(
            compressed_fri_config(),
            stacking_height_for(recursion_max_log_row_count()),
            recursion_max_log_row_count(),
            self.compress_prover.machine().clone(),
        )
    }

    pub fn recursion_program(
        &self,
        input: &ZKMRecursionWitnessValues<CoreSC, CorePcsProof>,
    ) -> Arc<RecursionProgram<KoalaBear>> {
        let builder_span = tracing::debug_span!("build recursion program").entered();
        let mut builder = Builder::<InnerConfig>::default();

        let input_var = input.read(&mut builder);
        let machine = self.core_circuit_verifier();
        ZKMRecursiveVerifier::verify(&mut builder, &machine, input_var);
        let operations = builder.into_operations();
        builder_span.exit();

        let compiler_span = tracing::debug_span!("compile recursion program").entered();
        let mut compiler = AsmCompiler::<InnerConfig>::default();
        let mut program = compiler.compile(operations);
        if let Some(recursion_shape_config) = &self.compress_shape_config {
            recursion_shape_config.fix_shape(&mut program);
        }
        let program = Arc::new(program);
        compiler_span.exit();
        program
    }

    pub fn compress_program(
        &self,
        input: &ZKMCompressWithVKeyWitnessValues<InnerSC, CorePcsProof>,
    ) -> Arc<RecursionProgram<KoalaBear>> {
        Arc::new(compress_program_from_input::<C>(
            self.compress_shape_config.as_ref(),
            &self.compress_circuit_verifier(),
            self.vk_verification,
            input,
        ))
    }

    pub fn shrink_program(
        &self,
        shrink_shape: RecursionShape,
        input: &ZKMCompressWithVKeyWitnessValues<InnerSC, CorePcsProof>,
    ) -> Arc<RecursionProgram<KoalaBear>> {
        let builder_span = tracing::debug_span!("build shrink program").entered();
        let mut builder = Builder::<InnerConfig>::default();
        let input_var = input.read(&mut builder);
        let machine = self.compress_circuit_verifier();
        ZKMCompressRootVerifierWithVKey::verify(
            &mut builder,
            &machine,
            input_var,
            self.vk_verification,
            PublicValuesOutputDigest::Reduce,
        );
        let operations = builder.into_operations();
        builder_span.exit();

        let compiler_span = tracing::debug_span!("compile shrink program").entered();
        let mut compiler = AsmCompiler::<InnerConfig>::default();
        let mut program = compiler.compile(operations);
        *program.shape_mut() = Some(shrink_shape);
        let program = Arc::new(program);
        compiler_span.exit();
        program
    }

    pub fn deferred_program(
        &self,
        input: &ZKMDeferredWitnessValues<InnerSC, CorePcsProof>,
    ) -> Arc<RecursionProgram<KoalaBear>> {
        let operations_span =
            tracing::debug_span!("get operations for the deferred program").entered();
        let mut builder = Builder::<InnerConfig>::default();
        let input_read_span = tracing::debug_span!("Read input values").entered();
        let input_var = input.read(&mut builder);
        input_read_span.exit();
        let verify_span = tracing::debug_span!("Verify deferred program").entered();

        let machine = self.compress_circuit_verifier();
        ZKMDeferredVerifier::verify(&mut builder, &machine, input_var, self.vk_verification);
        verify_span.exit();
        let operations = builder.into_operations();
        operations_span.exit();

        let compiler_span = tracing::debug_span!("compile deferred program").entered();
        let mut compiler = AsmCompiler::<InnerConfig>::default();
        let mut program = compiler.compile(operations);
        if let Some(recursion_shape_config) = &self.compress_shape_config {
            recursion_shape_config.fix_shape(&mut program);
        }
        let program = Arc::new(program);
        compiler_span.exit();
        program
    }

    pub fn get_recursion_core_inputs(
        &self,
        vk: &zkm_hypercube::MachineVerifyingKey<CoreSC>,
        shard_proofs: &[zkm_hypercube::ShardProof<CoreSC, CorePcsProof>],
        batch_size: usize,
        is_complete: bool,
    ) -> Vec<ZKMRecursionWitnessValues<CoreSC, CorePcsProof>> {
        let mut core_inputs = Vec::new();

        for (batch_idx, batch) in shard_proofs.chunks(batch_size).enumerate() {
            let proofs = batch.to_vec();

            core_inputs.push(ZKMRecursionWitnessValues {
                vk: vk.clone(),
                shard_proofs: proofs,
                is_complete,
                is_first_shard: batch_idx == 0,
                vk_root: self.recursion_vk_root,
            });
        }

        core_inputs
    }

    pub fn get_recursion_deferred_inputs<'a>(
        &'a self,
        vk: &'a zkm_hypercube::MachineVerifyingKey<CoreSC>,
        last_proof_pv: &zkm_hypercube::air::PublicValues<Word<KoalaBear>, KoalaBear>,
        deferred_proofs: &[ZKMReduceProofWrapper],
        batch_size: usize,
    ) -> Vec<ZKMDeferredWitnessValues<InnerSC, CorePcsProof>> {
        let mut deferred_digest = [KoalaBear::ZERO; DIGEST_SIZE];
        let mut deferred_inputs = Vec::new();

        for batch in deferred_proofs.chunks(batch_size) {
            let vks_and_proofs =
                batch.iter().cloned().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();

            let input = ZKMCompressWitnessValues { vks_and_proofs, is_complete: true };
            let input_with_merkle = self.make_merkle_proofs(input);
            let ZKMCompressWithVKeyWitnessValues { compress_val, merkle_val } = input_with_merkle;

            deferred_inputs.push(ZKMDeferredWitnessValues {
                vks_and_proofs: compress_val.vks_and_proofs,
                vk_merkle_data: merkle_val,
                start_reconstruct_deferred_digest: deferred_digest,
                is_complete: false,
                zkm_vk_digest: vk.hash_koalabear(),
                end_pc: KoalaBear::ZERO,
                end_shard: last_proof_pv.shard + KoalaBear::ONE,
                end_execution_shard: last_proof_pv.execution_shard,
                init_addr_bits: last_proof_pv.last_init_addr_bits,
                finalize_addr_bits: last_proof_pv.last_finalize_addr_bits,
                committed_value_digest: last_proof_pv.committed_value_digest,
                deferred_proofs_digest: last_proof_pv.deferred_proofs_digest,
            });

            deferred_digest = Self::hash_deferred_proofs(deferred_digest, batch);
        }
        deferred_inputs
    }

    /// Generate the inputs for the first layer of recursive proofs.
    #[allow(clippy::type_complexity)]
    pub fn get_first_layer_inputs<'a>(
        &'a self,
        vk: &'a ZKMVerifyingKey,
        shard_proofs: &[zkm_hypercube::ShardProof<CoreSC, CorePcsProof>],
        deferred_proofs: &[ZKMReduceProofWrapper],
        batch_size: usize,
    ) -> Vec<ZKMCircuitWitness> {
        let is_complete = shard_proofs.len() == 1 && deferred_proofs.is_empty();
        let core_inputs =
            self.get_recursion_core_inputs(&vk.vk, shard_proofs, batch_size, is_complete);
        let last_proof_pv = shard_proofs.last().unwrap().public_values.as_slice().borrow();
        let deferred_inputs =
            self.get_recursion_deferred_inputs(&vk.vk, last_proof_pv, deferred_proofs, batch_size);

        let mut inputs = Vec::new();
        inputs.extend(core_inputs.into_iter().map(ZKMCircuitWitness::Core));
        inputs.extend(deferred_inputs.into_iter().map(ZKMCircuitWitness::Deferred));
        inputs
    }

    /// Compiles, runs, and proves a single circuit witness against the compress machine.
    fn prove_compress_witness(
        &self,
        witness: ZKMCircuitWitness,
    ) -> (zkm_hypercube::MachineVerifyingKey<InnerSC>, zkm_hypercube::ShardProof<InnerSC, CorePcsProof>)
    {
        let (program, witness_stream) = match &witness {
            ZKMCircuitWitness::Core(input) => {
                let mut witness_stream = Vec::new();
                Witnessable::<InnerConfig>::write(input, &mut witness_stream);
                (self.recursion_program(input), witness_stream)
            }
            ZKMCircuitWitness::Deferred(input) => {
                let mut witness_stream = Vec::new();
                Witnessable::<InnerConfig>::write(input, &mut witness_stream);
                (self.deferred_program(input), witness_stream)
            }
            ZKMCircuitWitness::Compress(input) => {
                let input_with_merkle = self.make_merkle_proofs(input.clone());
                let mut witness_stream = Vec::new();
                Witnessable::<InnerConfig>::write(&input_with_merkle, &mut witness_stream);
                (self.compress_program(&input_with_merkle), witness_stream)
            }
        };

        let mut runtime =
            RecursionRuntime::<KoalaBear, zkm_stark::InnerChallenge, _>::new(
                program.clone(),
                inner_perm(),
            );
        runtime.witness_stream = witness_stream.into();
        runtime.run().map_err(|e| ZKMRecursionProverError::RuntimeError(e.to_string())).unwrap();
        runtime.print_stats();
        let mut record = runtime.record;

        self.compress_prover
            .machine()
            .generate_dependencies(std::iter::once(&mut record), None)
            .unwrap();

        let async_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (vk, proof, _permit) = async_rt.block_on(self.compress_prover.setup_and_prove_shard(
            program,
            record,
            None,
            ProverSemaphore::new(1),
        ));
        (vk, proof)
    }

    /// Reduce shard proofs to a single shard proof using the recursion prover.
    ///
    /// This performs the binary-tree reduction sequentially, one witness at a time. The old
    /// FRI-era prover ran this as a multi-stage threaded pipeline (separate checkpoint/trace-gen/
    /// prove worker threads communicating over channels); that pipeline has not been ported
    /// forward in this migration pass -- it's a performance optimization, not a correctness
    /// requirement, and re-threading it against the new `AirProver` API is left as a follow-up.
    #[instrument(name = "compress", level = "info", skip_all)]
    pub fn compress(
        &self,
        vk: &ZKMVerifyingKey,
        proof: ZKMCoreProof,
        deferred_proofs: Vec<ZKMReduceProofWrapper>,
        _opts: ZKMProverOpts,
    ) -> Result<ZKMReduceProofWrapper, ZKMRecursionProverError> {
        let batch_size = REDUCE_BATCH_SIZE;
        let first_layer_batch_size = 1;

        let shard_proofs = &proof.proof.0;
        let first_layer_inputs =
            self.get_first_layer_inputs(vk, shard_proofs, &deferred_proofs, first_layer_batch_size);

        let mut layer: Vec<_> =
            first_layer_inputs.into_iter().map(|w| self.prove_compress_witness(w)).collect();

        while layer.len() > 1 {
            let num_chunks = layer.len().div_ceil(batch_size);
            let mut next_layer = Vec::with_capacity(num_chunks);
            for chunk in layer.chunks(batch_size) {
                let is_complete = num_chunks == 1;
                let vks_and_proofs = chunk.to_vec();
                let witness = ZKMCircuitWitness::Compress(ZKMCompressWitnessValues {
                    vks_and_proofs,
                    is_complete,
                });
                next_layer.push(self.prove_compress_witness(witness));
            }
            layer = next_layer;
        }

        let (vk, proof) = layer.into_iter().next().unwrap();
        let vk_merkle_proof = self.vk_merkle_proof(&vk);

        Ok(ZKMReduceProofWrapper { vk, proof, vk_merkle_proof })
    }

    /// Wrap a reduce proof into a STARK proven over the shrink machine.
    #[instrument(name = "shrink", level = "info", skip_all)]
    pub fn shrink(
        &self,
        reduced_proof: ZKMReduceProofWrapper,
        _opts: ZKMProverOpts,
    ) -> Result<ZKMReduceProofWrapper, ZKMRecursionProverError> {
        let ZKMReduceProof { vk: compressed_vk, proof: compressed_proof, .. } = reduced_proof;
        let input = ZKMCompressWitnessValues {
            vks_and_proofs: vec![(compressed_vk, compressed_proof)],
            is_complete: true,
        };

        let input_with_merkle = self.make_merkle_proofs(input);

        let program =
            self.shrink_program(ShrinkAir::<KoalaBear>::shrink_shape(), &input_with_merkle);

        let mut runtime =
            RecursionRuntime::<KoalaBear, zkm_stark::InnerChallenge, _>::new(
                program.clone(),
                inner_perm(),
            );

        let mut witness_stream = Vec::new();
        Witnessable::<InnerConfig>::write(&input_with_merkle, &mut witness_stream);
        runtime.witness_stream = witness_stream.into();
        runtime.run().map_err(|e| ZKMRecursionProverError::RuntimeError(e.to_string()))?;
        runtime.print_stats();
        tracing::debug!("Shrink program executed successfully");

        let mut record = runtime.record;
        self.shrink_prover
            .machine()
            .generate_dependencies(std::iter::once(&mut record), None)
            .unwrap();

        let async_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (shrink_vk, shrink_proof, _permit) = async_rt.block_on(
            self.shrink_prover.setup_and_prove_shard(program, record, None, ProverSemaphore::new(1)),
        );

        let vk_merkle_proof = self.vk_merkle_proof(&shrink_vk);
        Ok(ZKMReduceProofWrapper { vk: shrink_vk, proof: shrink_proof, vk_merkle_proof })
    }

    /// Wrap a reduce proof into a STARK proven over a SNARK-friendly (Bn254) field.
    ///
    /// Blocked on task #57 (`ZkmOuterGlobalContext`) -- see the module doc comment.
    #[instrument(name = "wrap_bn254", level = "info", skip_all)]
    pub fn wrap_bn254(
        &self,
        _compressed_proof: ZKMReduceProofWrapper,
        _opts: ZKMProverOpts,
    ) -> Result<ZKMWrapProof, ZKMRecursionProverError> {
        unimplemented!(
            "outer/Bn254 wrap proving is blocked on task #57 (ZkmOuterGlobalContext)"
        )
    }

    /// Wrap the STARK proven over a SNARK-friendly field into a PLONK proof.
    ///
    /// Blocked on task #57 (`ZkmOuterGlobalContext`) -- see the module doc comment.
    #[instrument(name = "wrap_plonk_bn254", level = "info", skip_all)]
    pub fn wrap_plonk_bn254(&self, _proof: ZKMWrapProof, _build_dir: &Path) -> PlonkBn254Proof {
        unimplemented!(
            "outer/Bn254 wrap proving is blocked on task #57 (ZkmOuterGlobalContext)"
        )
    }

    /// Wrap the STARK proven over a SNARK-friendly field into a Groth16 proof.
    ///
    /// Blocked on task #57 (`ZkmOuterGlobalContext`) -- see the module doc comment.
    #[instrument(name = "wrap_groth16_bn254", level = "info", skip_all)]
    pub fn wrap_groth16_bn254(&self, _proof: ZKMWrapProof, _build_dir: &Path) -> Groth16Bn254Proof {
        unimplemented!(
            "outer/Bn254 wrap proving is blocked on task #57 (ZkmOuterGlobalContext)"
        )
    }

    /// Wrap the STARK proven over a SNARK-friendly field into a DV-SNARK proof.
    ///
    /// Blocked on task #57 (`ZkmOuterGlobalContext`) -- see the module doc comment.
    #[instrument(name = "wrap_dvsnark_bn254", level = "info", skip_all)]
    pub fn wrap_dvsnark_bn254(
        &self,
        _proof: ZKMWrapProof,
        _build_dir: &Path,
        _store_dir: &Path,
    ) -> DvSnarkBn254Proof {
        unimplemented!(
            "outer/Bn254 wrap proving is blocked on task #57 (ZkmOuterGlobalContext)"
        )
    }

    /// Accumulate deferred proofs into a single digest.
    pub fn hash_deferred_proofs(
        prev_digest: [KoalaBear; DIGEST_SIZE],
        deferred_proofs: &[ZKMReduceProofWrapper],
    ) -> [KoalaBear; 8] {
        let mut digest = prev_digest;
        for proof in deferred_proofs.iter() {
            let pv: &RecursionPublicValues<KoalaBear> =
                proof.proof.public_values.as_slice().borrow();
            let committed_values_digest = words_to_bytes(&pv.committed_value_digest);
            digest = hash_deferred_proof(
                &digest,
                &pv.zkm_vk_digest,
                &committed_values_digest.try_into().unwrap(),
            );
        }
        digest
    }

    pub fn make_merkle_proofs(
        &self,
        input: ZKMCompressWitnessValues<CoreSC, CorePcsProof>,
    ) -> ZKMCompressWithVKeyWitnessValues<CoreSC, CorePcsProof> {
        let num_vks = self.recursion_vk_map.len();
        let (vk_indices, vk_digest_values): (Vec<_>, Vec<_>) = if self.vk_verification {
            input
                .vks_and_proofs
                .iter()
                .map(|(vk, _)| {
                    let vk_digest = vk.hash_koalabear();
                    let index = self.recursion_vk_map.get(&vk_digest).expect("vk not allowed");
                    (index, vk_digest)
                })
                .unzip()
        } else {
            input
                .vks_and_proofs
                .iter()
                .map(|(vk, _)| {
                    let vk_digest = vk.hash_koalabear();
                    let index = (vk_digest[0].as_canonical_u32() as usize) % num_vks;
                    (index, [KoalaBear::from_canonical_usize(index); 8])
                })
                .unzip()
        };

        let proofs = vk_indices
            .iter()
            .map(|index| {
                let (_, proof) = MerkleTree::open(&self.recursion_vk_tree, *index);
                proof
            })
            .collect();

        let merkle_val = ZKMMerkleProofWitnessValues {
            root: self.recursion_vk_root,
            values: vk_digest_values,
            vk_merkle_proofs: proofs,
        };

        ZKMCompressWithVKeyWitnessValues { compress_val: input, merkle_val }
    }

    /// Build the outer `zkm_hypercube::verifier::MerkleProof` proving that `vk` is a member of
    /// the allowed recursion VK set, for the `ZKMReduceProof::vk_merkle_proof` field.
    fn vk_merkle_proof(
        &self,
        vk: &zkm_hypercube::MachineVerifyingKey<InnerSC>,
    ) -> zkm_hypercube::verifier::MerkleProof<ZkmGlobalContext> {
        let vk_digest = vk.hash_koalabear();
        let index = if self.vk_verification {
            *self.recursion_vk_map.get(&vk_digest).expect("vk not allowed")
        } else {
            (vk_digest[0].as_canonical_u32() as usize) % self.recursion_vk_map.len()
        };
        let (_, proof) = MerkleTree::open(&self.recursion_vk_tree, index);
        zkm_hypercube::verifier::MerkleProof { index, path: proof.path }
    }

    fn check_for_high_cycles(cycles: u64) {
        if cycles > 100_000_000 {
            tracing::warn!(
                "high cycle count, consider using the prover network for proof generation"
            );
        }
    }
}

pub fn compress_program_from_input<C: ZKMProverComponents>(
    config: Option<&RecursionShapeConfig<KoalaBear, CompressAir<KoalaBear>>>,
    machine: &RecursiveShardVerifier<
        InnerConfig,
        ZkmGlobalContext,
        DuplexChallengerVariable<InnerConfig>,
        CompressAir<KoalaBear>,
    >,
    vk_verification: bool,
    input: &ZKMCompressWithVKeyWitnessValues<InnerSC, CorePcsProof>,
) -> RecursionProgram<KoalaBear> {
    let builder_span = tracing::debug_span!("build compress program").entered();
    let mut builder = Builder::<InnerConfig>::default();
    let input_var = input.read(&mut builder);
    ZKMCompressWithVKeyVerifier::verify(
        &mut builder,
        machine,
        input_var,
        vk_verification,
        PublicValuesOutputDigest::Reduce,
    );
    let operations = builder.into_operations();
    builder_span.exit();

    let compiler_span = tracing::debug_span!("compile compress program").entered();
    let mut compiler = AsmCompiler::<InnerConfig>::default();
    let mut program = compiler.compile(operations);
    if let Some(config) = config {
        config.fix_shape(&mut program);
    }
    compiler_span.exit();

    program
}

#[cfg(test)]
pub mod tests {
    use std::{
        fs::File,
        io::{Read, Write},
    };

    use super::*;

    use anyhow::Result;

    #[cfg(test)]
    use serial_test::serial;
    #[cfg(test)]
    use zkm_core_machine::utils::setup_logger;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Test {
        Core,
        Compress,
        Shrink,
        Wrap,
        CircuitTest,
        All,
    }

    pub fn test_e2e_prover<C: ZKMProverComponents>(
        prover: &ZKMProver<C>,
        elf: &[u8],
        stdin: ZKMStdin,
        opts: ZKMProverOpts,
        test_kind: Test,
    ) -> Result<()> {
        run_e2e_prover_with_options(prover, elf, stdin, opts, test_kind, true)
    }

    pub fn bench_e2e_prover<C: ZKMProverComponents>(
        prover: &ZKMProver<C>,
        elf: &[u8],
        stdin: ZKMStdin,
        opts: ZKMProverOpts,
        test_kind: Test,
    ) -> Result<()> {
        run_e2e_prover_with_options(prover, elf, stdin, opts, test_kind, false)
    }

    pub fn run_e2e_prover_with_options<C: ZKMProverComponents>(
        prover: &ZKMProver<C>,
        elf: &[u8],
        stdin: ZKMStdin,
        opts: ZKMProverOpts,
        test_kind: Test,
        verify: bool,
    ) -> Result<()> {
        tracing::info!("initializing prover");
        let context = ZKMContext::default();

        tracing::info!("setup elf");
        let (_, program, vk) = prover.setup(elf);

        tracing::info!("prove core");
        let core_proof = prover.prove_core(program, &stdin, opts, context)?;
        let public_values = core_proof.public_values.clone();

        if verify {
            tracing::info!("verify core");
            prover.verify(&core_proof.proof, &vk)?;
        }

        if test_kind == Test::Core {
            return Ok(());
        }

        tracing::info!("compress");
        let compress_span = tracing::debug_span!("compress").entered();
        let compressed_proof = prover.compress(&vk, core_proof, vec![], opts)?;
        compress_span.exit();

        if verify {
            tracing::info!("verify compressed");
            prover.verify_compressed(&compressed_proof, &vk)?;
        }

        if test_kind == Test::Compress {
            return Ok(());
        }

        tracing::info!("shrink");
        let shrink_proof = prover.shrink(compressed_proof, opts)?;

        if verify {
            tracing::info!("verify shrink");
            prover.verify_shrink(&shrink_proof, &vk)?;
        }

        if test_kind == Test::Shrink {
            return Ok(());
        }

        tracing::info!("wrap bn254");
        let wrapped_bn254_proof = prover.wrap_bn254(shrink_proof, opts)?;
        let bytes = bincode::serialize(&wrapped_bn254_proof).unwrap();

        let mut file = File::create("proof-with-pis.bin").unwrap();
        file.write_all(bytes.as_slice()).unwrap();

        let mut file = File::open("proof-with-pis.bin").unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();

        let wrapped_bn254_proof = bincode::deserialize(&bytes).unwrap();

        if verify {
            tracing::info!("verify wrap bn254");
            prover.verify_wrap_bn254(&wrapped_bn254_proof, &vk).unwrap();
        }

        let _ = vk;
        let _ = public_values;
        let _ = test_kind;
        Ok(())
    }

    pub fn test_e2e_with_deferred_proofs_prover(
        prover: &ZKMProver<DefaultProverComponents>,
        opts: ZKMProverOpts,
    ) -> Result<()> {
        let keccak_elf = test_artifacts::KECCAK_SPONGE_ELF;
        let verify_elf = test_artifacts::VERIFY_PROOF_ELF;

        tracing::info!("setup keccak elf");
        let (_, keccak_program, keccak_vk) = prover.setup(keccak_elf);

        tracing::info!("setup verify elf");
        let (_, verify_program, verify_vk) = prover.setup(verify_elf);

        tracing::info!("prove subproof 1");
        let mut stdin = ZKMStdin::new();
        stdin.write(&1usize);
        stdin.write(&vec![0u8, 0, 0]);
        let deferred_proof_1 =
            prover.prove_core(keccak_program.clone(), &stdin, opts, Default::default())?;
        let pv_1 = deferred_proof_1.public_values.as_slice().to_vec().clone();

        tracing::info!("prove subproof 2");
        let mut stdin = ZKMStdin::new();
        stdin.write(&3usize);
        stdin.write(&vec![0u8, 1, 2]);
        stdin.write(&vec![2, 3, 4]);
        stdin.write(&vec![5, 6, 7]);
        let deferred_proof_2 =
            prover.prove_core(keccak_program, &stdin, opts, Default::default())?;
        let pv_2 = deferred_proof_2.public_values.as_slice().to_vec().clone();

        tracing::info!("compress subproof 1");
        let deferred_reduce_1 = prover.compress(&keccak_vk, deferred_proof_1, vec![], opts)?;

        tracing::info!("compress subproof 2");
        let deferred_reduce_2 = prover.compress(&keccak_vk, deferred_proof_2, vec![], opts)?;

        let mut stdin = ZKMStdin::new();
        let vkey_digest = keccak_vk.hash_koalabear();
        let vkey_digest: [u32; 8] = vkey_digest
            .iter()
            .map(|n| n.as_canonical_u32())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        stdin.write(&vkey_digest);
        stdin.write(&vec![pv_1.clone(), pv_2.clone(), pv_2.clone()]);
        stdin.write_proof(deferred_reduce_1.clone(), keccak_vk.vk.clone());
        stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());
        stdin.write_proof(deferred_reduce_2.clone(), keccak_vk.vk.clone());

        tracing::info!("proving verify program (core)");
        let verify_proof = prover.prove_core(verify_program, &stdin, opts, Default::default())?;

        tracing::info!("compress verify program");
        let verify_reduce = prover.compress(
            &verify_vk,
            verify_proof,
            vec![deferred_reduce_1, deferred_reduce_2.clone(), deferred_reduce_2],
            opts,
        )?;
        let reduce_pv: &RecursionPublicValues<_> =
            verify_reduce.proof.public_values.as_slice().borrow();
        println!("deferred_hash: {:?}", reduce_pv.deferred_proofs_digest);
        println!("complete: {:?}", reduce_pv.is_complete);

        tracing::info!("verify verify program");
        prover.verify_compressed(&verify_reduce, &verify_vk)?;

        let shrink_proof = prover.shrink(verify_reduce, opts)?;

        tracing::info!("verify shrink");
        prover.verify_shrink(&shrink_proof, &verify_vk)?;

        tracing::info!("wrap bn254");
        let wrapped_bn254_proof = prover.wrap_bn254(shrink_proof, opts)?;

        tracing::info!("verify wrap bn254");
        prover.verify_wrap_bn254(&wrapped_bn254_proof, &verify_vk).unwrap();

        Ok(())
    }

    /// Tests the core -> compress -> shrink pipeline (everything migrated to the hypercube
    /// backend so far), stopping short of the outer/Bn254 wrap stage. `wrap_bn254` and friends
    /// are `unimplemented!()` pending task #57 (`ZkmOuterGlobalContext`) -- see the module doc
    /// comment -- so `Test::All`/`Test::Wrap` would panic here.
    #[test]
    #[serial]
    #[ignore]
    fn test_e2e_up_to_shrink() -> Result<()> {
        let elf = test_artifacts::HELLO_WORLD_ELF;
        setup_logger();
        let opts = ZKMProverOpts::default();
        let prover = ZKMProver::<DefaultProverComponents>::new();
        test_e2e_prover::<DefaultProverComponents>(
            &prover,
            elf,
            ZKMStdin::default(),
            opts,
            Test::Shrink,
        )
    }

    /// Compiles (but does not prove) the first-layer recursion program that verifies a single
    /// core shard, to measure real per-chip row counts for `RecursionShapeConfig`'s
    /// `allowed_shapes` tables (`crates/recursion/core/src/shape.rs`) without paying for actual
    /// STARK proving of the recursion program, which is the dominant cost of a full compress run.
    /// If the configured shape table is too small for the real heights, `recursion_program`'s
    /// call to `RecursionShapeConfig::fix_shape` panics with `"no shape found for heights:
    /// {heights:?}"`, which reports the exact real heights needed.
    #[test]
    #[serial]
    #[ignore]
    fn measure_recursion_program_heights() -> Result<()> {
        let elf = test_artifacts::HELLO_WORLD_ELF;
        setup_logger();
        let opts = ZKMProverOpts::default();
        let prover = ZKMProver::<DefaultProverComponents>::new();
        let context = ZKMContext::default();

        let (_, program, vk) = prover.setup(elf);
        let core_proof = prover.prove_core(program, &ZKMStdin::default(), opts, context)?;
        prover.verify(&core_proof.proof, &vk)?;

        let shard_proofs = &core_proof.proof.0;
        let inputs = prover.get_first_layer_inputs(&vk, shard_proofs, &[], 1);
        let input = match &inputs[0] {
            ZKMCircuitWitness::Core(input) => input,
            _ => panic!("expected a core witness for a single-shard proof"),
        };

        let _recursion_program = prover.recursion_program(input);
        println!("recursion program compiled and fit the configured shape table");
        Ok(())
    }

    /// Tests an end-to-end workflow of proving a program across the entire proof generation
    /// pipeline.
    #[test]
    #[serial]
    #[ignore]
    fn test_e2e() -> Result<()> {
        let elf = test_artifacts::FIBONACCI_ELF;
        setup_logger();
        let opts = ZKMProverOpts::default();
        let prover = ZKMProver::<DefaultProverComponents>::new();
        test_e2e_prover::<DefaultProverComponents>(
            &prover,
            elf,
            ZKMStdin::default(),
            opts,
            Test::All,
        )
    }

    /// Tests an end-to-end workflow of proving a program across the entire proof generation
    /// pipeline.
    #[test]
    #[serial]
    #[ignore]
    fn test_e2e_hello_world() -> Result<()> {
        let elf = test_artifacts::HELLO_WORLD_ELF;

        setup_logger();
        let opts = ZKMProverOpts::default();
        let prover = ZKMProver::<DefaultProverComponents>::new();
        test_e2e_prover::<DefaultProverComponents>(
            &prover,
            elf,
            ZKMStdin::default(),
            opts,
            Test::All,
        )
    }

    /// Tests an end-to-end workflow of proving a program across the entire proof generation
    /// pipeline in addition to verifying deferred proofs.
    #[test]
    #[serial]
    #[ignore]
    fn test_e2e_with_deferred_proofs() -> Result<()> {
        setup_logger();
        let prover = ZKMProver::<DefaultProverComponents>::new();
        test_e2e_with_deferred_proofs_prover(&prover, ZKMProverOpts::default())
    }
}
