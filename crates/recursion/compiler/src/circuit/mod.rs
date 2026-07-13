mod builder;
mod compiler;
mod config;

pub use builder::*;
pub use compiler::*;
pub use config::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use p3_field::FieldAlgebra;
    use p3_koala_bear::Poseidon2InternalLayerKoalaBear;
    use slop_challenger::IopCtx;

    use zkm_hypercube::{
        config::{default_fri_config, ZkmGlobalContext},
        prover::{AirProver, ProverSemaphore, ZkmShardProver},
        ShardVerifier,
    };
    use zkm_recursion_core::{machine::RecursionAir, Runtime, RuntimeError};
    use zkm_stark::{KoalaBearPoseidon2Inner, StarkGenericConfig};

    use crate::{
        circuit::{AsmBuilder, AsmCompiler, CircuitV2Builder},
        ir::*,
    };

    const DEGREE: usize = 3;

    /// The log2 of the number of rows each stacked-PCS column is grouped into. Mirrors
    /// `zkm_recursion_core::machine::tests::RECURSION_LOG_STACKING_HEIGHT`, kept in sync by
    /// convention.
    const RECURSION_LOG_STACKING_HEIGHT: u32 = 4;

    type SC = KoalaBearPoseidon2Inner;
    type F = <SC as StarkGenericConfig>::Val;
    type EF = <SC as StarkGenericConfig>::Challenge;
    type A = RecursionAir<F, DEGREE>;

    #[test]
    fn test_io() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let felts = builder.hint_felts_v2(3);
        assert_eq!(felts.len(), 3);
        let sum: Felt<_> = builder.eval(felts[0] + felts[1]);
        builder.assert_felt_eq(sum, felts[2]);

        let exts = builder.hint_exts_v2(3);
        assert_eq!(exts.len(), 3);
        let sum: Ext<_, _> = builder.eval(exts[0] + exts[1]);
        builder.assert_ext_ne(sum, exts[2]);

        let x = builder.hint_ext_v2();
        builder.assert_ext_eq(x, exts[0] + felts[0]);

        let y = builder.hint_felt_v2();
        let zero: Felt<_> = builder.constant(F::ZERO);
        builder.assert_felt_eq(y, zero);

        let operations = builder.into_operations();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile(operations));
        let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
            program.clone(),
            SC::new().perm,
        );
        runtime.witness_stream = [
            vec![F::ONE.into(), F::ONE.into(), F::TWO.into()],
            vec![F::ZERO.into(), F::ONE.into(), F::TWO.into()],
            vec![F::ONE.into()],
            vec![F::ZERO.into()],
        ]
        .concat()
        .into();
        runtime.run().unwrap();

        let machine = A::compress_machine();
        let max_log_row_count = zkm_stark::ZKMCoreOpts::recursion().shard_size.ilog2() as usize;

        let shard_prover = ZkmShardProver::<A>::new(ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            RECURSION_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine.clone(),
        ));

        let setup_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (vk, proof, _permit) = setup_rt.block_on(shard_prover.setup_and_prove_shard(
            program,
            runtime.record,
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
        shard_verifier.verify_shard(&vk, &proof, &mut challenger).expect("should verify");

        tracing::info!("verified recursion shard proof");
    }

    #[test]
    fn test_empty_witness_stream() {
        let mut builder = AsmBuilder::<F, EF>::default();

        let felts = builder.hint_felts_v2(3);
        assert_eq!(felts.len(), 3);
        let sum: Felt<_> = builder.eval(felts[0] + felts[1]);
        builder.assert_felt_eq(sum, felts[2]);

        let exts = builder.hint_exts_v2(3);
        assert_eq!(exts.len(), 3);
        let sum: Ext<_, _> = builder.eval(exts[0] + exts[1]);
        builder.assert_ext_ne(sum, exts[2]);

        let operations = builder.into_operations();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile(operations));
        let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
            program.clone(),
            SC::new().perm,
        );
        runtime.witness_stream =
            [vec![F::ONE.into(), F::ONE.into(), F::TWO.into()]].concat().into();

        match runtime.run() {
            Err(RuntimeError::EmptyWitnessStream) => (),
            Ok(_) => panic!("should not succeed"),
            Err(x) => panic!("should not yield error variant: {x}"),
        }
    }
}
