//! A simple example showing how to aggregate proofs of multiple programs with ZKM.

use zkm_prover::{components::DefaultProverComponents, InnerSC};
use zkm_recursion_core::Runtime as RecursionRuntime;
use zkm_recursion_circuit::{
    machine::ZKMCompressWitnessValues,
    witness::Witnessable,
};
use zkm_recursion_compiler::{
    config::InnerConfig,
};
use zkm_sdk::{
    include_elf, ProverClient, ZKMProof, ZKMStdin, ZKMProver,
};
use zkm_stark::{Challenge, MachineProver, StarkGenericConfig, Val, ZKMProverOpts};

/// A program that just runs a simple computation.
const FIBONACCI_ELF: &[u8] = include_elf!("fibonacci");

fn main() {
    // Setup the logger.
    zkm_sdk::utils::setup_logger();

    // Initialize the proving client.
    let client = ProverClient::new();

    // Setup the proving and verifying keys.
    let (fibonacci_pk, _fibonacci_vk) = client.setup(FIBONACCI_ELF);

    // Generate the fibonacci proofs.
    let proof_1 = tracing::info_span!("generate fibonacci proof n=10").in_scope(|| {
        let mut stdin = ZKMStdin::new();
        stdin.write(&10);
        client.prove(&fibonacci_pk, stdin).compressed().run().expect("proving failed")
    });
    let proof_2 = tracing::info_span!("generate fibonacci proof n=20").in_scope(|| {
        let mut stdin = ZKMStdin::new();
        stdin.write(&20);
        client.prove(&fibonacci_pk, stdin).compressed().run().expect("proving failed")
    });
    println!("generate compressed proof done");

    //---------------------------------------------------

    let opts = ZKMProverOpts::default();

    let prover: ZKMProver<DefaultProverComponents> = ZKMProver::new();

    let ZKMProof::Compressed(proof1) = proof_1.proof else { panic!() };
    let ZKMProof::Compressed(proof2) = proof_2.proof else { panic!() };

    prover.compress2(
        vec![*proof1, *proof2],
        opts,
    ).unwrap();

    // let vks_and_proofs = vec![
    //     (proof1.vk, proof1.proof),
    //     (proof2.vk, proof2.proof),
    // ];
    // let input = ZKMCompressWitnessValues { vks_and_proofs, is_complete: true };

    // let (compress_program, witness_stream) = {
    //     let mut witness_stream = Vec::new();

    //     let input_with_merkle = prover.make_merkle_proofs(input);

    //     Witnessable::<InnerConfig>::write(
    //         &input_with_merkle,
    //         &mut witness_stream,
    //     );

    //     (prover.compress_program(&input_with_merkle), witness_stream)
    // };

    // // Execute the runtime.
    // let record = tracing::debug_span!("execute runtime").in_scope(|| {
    //     let mut runtime =
    //         RecursionRuntime::<Val<InnerSC>, Challenge<InnerSC>, _>::new(
    //             compress_program.clone(),
    //             prover.compress_prover.config().perm.clone(),
    //         );
    //     runtime.witness_stream = witness_stream.into();
    //     runtime
    //         .run()
    //         .unwrap_or_else(|err| panic!("Runtime execution failed: {err}"));
    //     runtime.record
    // });

    // // Generate the dependencies.
    // let mut records = vec![record];
    // tracing::debug_span!("generate dependencies").in_scope(|| {
    //     prover.compress_prover.machine().generate_dependencies(
    //         &mut records,
    //         &opts.recursion_opts,
    //         None,
    //     )
    // });

    // // Generate the traces.
    // let record = records.into_iter().next().unwrap();
    // let traces = tracing::debug_span!("generate traces")
    //     .in_scope(|| prover.compress_prover.generate_traces(&record));

    // // Get the keys.
    // let (pk, vk) = tracing::debug_span!("Setup compress program")
    //     .in_scope(|| prover.compress_prover.setup(&compress_program));

    // // Observe the proving key.
    // let mut challenger = prover.compress_prover.config().challenger();
    // tracing::debug_span!("observe proving key").in_scope(|| {
    //     pk.observe_into(&mut challenger);
    // });

    // #[cfg(feature = "debug")]
    // prover.compress_prover.debug_constraints(
    //     &prover.compress_prover.pk_to_host(&pk),
    //     vec![record.clone()],
    //     &mut challenger.clone(),
    // );

    // // Commit to the record and traces.
    // let data = tracing::debug_span!("commit")
    //     .in_scope(|| prover.compress_prover.commit(&record, traces));

    // // Generate the proof.
    // let proof = tracing::debug_span!("open").in_scope(|| {
    //     prover.compress_prover.open(&pk, data, &mut challenger).unwrap()
    // });

    // // Verify the proof.
    // #[cfg(feature = "debug")]
    // prover.compress_prover
    //     .machine()
    //     .verify(
    //         &vk,
    //         &zkm_stark::MachineProof {
    //             shard_proofs: vec![proof.clone()],
    //         },
    //         &mut prover.compress_prover.config().challenger(),
    //     )
    //     .unwrap();
}
