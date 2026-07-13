use p3_bn254_fr::Bn254Fr;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_koala_bear::KoalaBear;

use zkm_recursion_compiler::ir::{Builder, Config, Felt, Var};
use zkm_recursion_core::DIGEST_SIZE;

use zkm_hypercube::word::Word;

/// Convert 8 KoalaBear words into a Bn254Fr field element by shifting by 31 bits each time. The last
/// word becomes the least significant bits.
#[allow(dead_code)]
pub fn koalabears_to_bn254(digest: &[KoalaBear; 8]) -> Bn254Fr {
    let mut result = Bn254Fr::ZERO;
    for word in digest.iter() {
        // Since KoalaBear prime is less than 2^31, we can shift by 31 bits each time and still be
        // within the Bn254Fr field, so we don't have to truncate the top 3 bits.
        result *= Bn254Fr::from_canonical_u64(1 << 31);
        result += Bn254Fr::from_canonical_u32(word.as_canonical_u32());
    }
    result
}

/// Convert 32 KoalaBear bytes into a Bn254Fr field element. The first byte's most significant 3 bits
/// (which would become the 3 most significant bits) are truncated.
#[allow(dead_code)]
pub fn koalabear_bytes_to_bn254(bytes: &[KoalaBear; 32]) -> Bn254Fr {
    let mut result = Bn254Fr::ZERO;
    for (i, byte) in bytes.iter().enumerate() {
        debug_assert!(byte < &KoalaBear::from_canonical_u32(256));
        if i == 0 {
            // 32 bytes is more than Bn254 prime, so we need to truncate the top 3 bits.
            result = Bn254Fr::from_canonical_u32(byte.as_canonical_u32() & 0x1f);
        } else {
            result *= Bn254Fr::from_canonical_u32(256);
            result += Bn254Fr::from_canonical_u32(byte.as_canonical_u32());
        }
    }
    result
}

#[allow(dead_code)]
pub fn felts_to_bn254_var<C: Config>(
    builder: &mut Builder<C>,
    digest: &[Felt<C::F>; DIGEST_SIZE],
) -> Var<C::N> {
    let var_2_31: Var<_> = builder.constant(C::N::from_canonical_u32(1 << 31));
    let result = builder.constant(C::N::ZERO);
    for (i, word) in digest.iter().enumerate() {
        let word_var = builder.felt2var_circuit(*word);
        if i == 0 {
            builder.assign(result, word_var);
        } else {
            builder.assign(result, result * var_2_31 + word_var);
        }
    }
    result
}

#[allow(dead_code)]
pub fn felt_bytes_to_bn254_var<C: Config>(
    builder: &mut Builder<C>,
    bytes: &[Felt<C::F>; 32],
) -> Var<C::N> {
    let var_256: Var<_> = builder.constant(C::N::from_canonical_u32(256));
    let zero_var: Var<_> = builder.constant(C::N::ZERO);
    let result = builder.constant(C::N::ZERO);
    for (i, byte) in bytes.iter().enumerate() {
        let byte_bits = builder.num2bits_f_circuit(*byte);
        if i == 0 {
            // Since 32 bytes doesn't fit into Bn254, we need to truncate the top 3 bits.
            // For first byte, zero out 3 most significant bits.
            for i in 0..3 {
                builder.assign(byte_bits[8 - i - 1], zero_var);
            }
            let byte_var = builder.bits2num_v_circuit(&byte_bits);
            builder.assign(result, byte_var);
        } else {
            let byte_var = builder.bits2num_v_circuit(&byte_bits);
            builder.assign(result, result * var_256 + byte_var);
        }
    }
    result
}

#[allow(dead_code)]
pub fn words_to_bytes<T: Copy>(words: &[Word<T>]) -> Vec<T> {
    words.iter().flat_map(|w| w.0).collect::<Vec<_>>()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use slop_challenger::IopCtx;
    use zkm_core_machine::utils::setup_logger;
    use zkm_hypercube::{
        config::{default_fri_config, ZkmGlobalContext},
        prover::{AirProver, ProverSemaphore, ZkmShardProver},
        ShardVerifier,
    };
    use zkm_recursion_compiler::{circuit::AsmCompiler, circuit::AsmConfig, ir::DslIr};

    use zkm_recursion_compiler::ir::TracedVec;
    use zkm_recursion_core::{machine::RecursionAir, Runtime};
    use zkm_stark::{koala_bear_poseidon2::KoalaBearPoseidon2, InnerChallenge, InnerVal};

    use crate::witness::WitnessBlock;

    type SC = KoalaBearPoseidon2;
    type F = InnerVal;
    type EF = InnerChallenge;

    /// The log2 of the number of rows each stacked-PCS column is grouped into. Mirrors
    /// `zkm_recursion_core::machine::tests::RECURSION_LOG_STACKING_HEIGHT`, kept in sync by
    /// convention.
    const RECURSION_LOG_STACKING_HEIGHT: u32 = 4;

    /// A simplified version of some code from `recursion/core/src/machine.rs`.
    /// Takes in a program and runs it with the given witness and generates a proof with the
    /// wide Poseidon2 recursion machine.
    pub(crate) fn run_test_recursion(
        operations: TracedVec<DslIr<AsmConfig<F, EF>>>,
        witness_stream: impl IntoIterator<Item = WitnessBlock<AsmConfig<F, EF>>>,
    ) {
        let max_log_row_count = zkm_stark::ZKMCoreOpts::recursion().shard_size.ilog2() as usize;
        run_test_recursion_with_max_log_row_count(operations, witness_stream, max_log_row_count)
    }

    /// Like [`run_test_recursion`], but with an explicit override for the recursion machine's
    /// `max_log_row_count` (i.e. its shard size), instead of the default
    /// `zkm_stark::ZKMCoreOpts::recursion().shard_size`. Needed for circuits that emit far more
    /// rows than that default accommodates, such as a full, real `verify_shard` gadget chain
    /// (zerocheck + LogUp-GKR + jagged + basefold), which the default recursion shard size was
    /// never sized for.
    pub(crate) fn run_test_recursion_with_max_log_row_count(
        operations: TracedVec<DslIr<AsmConfig<F, EF>>>,
        witness_stream: impl IntoIterator<Item = WitnessBlock<AsmConfig<F, EF>>>,
        max_log_row_count: usize,
    ) {
        setup_logger();

        let compile_span = tracing::debug_span!("compile").entered();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile(operations));
        compile_span.exit();

        let config = SC::default();

        let run_span = tracing::debug_span!("run the recursive program").entered();
        let mut runtime = Runtime::<F, EF, _>::new(program.clone(), config.perm.clone());
        runtime.witness_stream.extend(witness_stream);
        tracing::debug_span!("run").in_scope(|| runtime.run().unwrap());
        assert!(runtime.witness_stream.is_empty());
        run_span.exit();

        let record = runtime.record;
        eprintln!(
            "DIAGNOSTIC record stats: {:?}",
            zkm_hypercube::record::MachineRecord::stats(&record)
        );

        // Run with the poseidon2 wide chip.
        let proof_wide_span = tracing::debug_span!("Run test with wide machine").entered();
        let machine = RecursionAir::<F, 3>::compress_machine();

        let shard_prover = ZkmShardProver::<RecursionAir<F, 3>>::new(
            ShardVerifier::from_basefold_parameters(
                default_fri_config(),
                RECURSION_LOG_STACKING_HEIGHT,
                max_log_row_count,
                machine.clone(),
            ),
        );

        let setup_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (vk, proof, _permit) = setup_rt.block_on(shard_prover.setup_and_prove_shard(
            program,
            record,
            None,
            ProverSemaphore::new(1),
        ));

        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            RECURSION_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        let mut challenger = ZkmGlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        if let Err(e) = shard_verifier.verify_shard(&vk, &proof, &mut challenger) {
            panic!("Verification failed: {e:?}");
        }
        proof_wide_span.exit();
    }
}
