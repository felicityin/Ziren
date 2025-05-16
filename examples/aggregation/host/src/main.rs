//! A simple example showing how to aggregate proofs of multiple programs with ZKM.

use std::time::Instant;

use tokio::task;
use zkm_sdk::{
    include_elf, HashableKey, ProverClient, ZKMProof, ZKMProofWithPublicValues, ZKMStdin,
    ZKMVerifyingKey,
};

/// A program that aggregates the proofs of the simple program.
const AGGREGATION_ELF: &[u8] = include_elf!("aggregation");

/// A program that just runs a simple computation.
const FIBONACCI_ELF: &[u8] = include_elf!("fibonacci");

/// An input to the aggregation program.
///
/// Consists of a proof and a verification key.
struct AggregationInput {
    pub proof: ZKMProofWithPublicValues,
    pub vk: ZKMVerifyingKey,
}

#[tokio::main]
async fn main() {
    let start = Instant::now();

    // Setup the logger.
    zkm_sdk::utils::setup_logger();

    // Initialize the proving client.
    let client = ProverClient::new();
    let prove_mode = zkm_sdk::ZKMProofKind::Compressed;

    // Setup the proving and verifying keys.
    let (aggregation_pk, _) = client.setup(AGGREGATION_ELF);
    let (fibonacci_pk, fibonacci_vk) = client.setup(FIBONACCI_ELF);

    // Generate the fibonacci proofs.
    let proof_1 = task::spawn_blocking(move || {
        let client = ProverClient::new();
        let (fibonacci_pk, fibonacci_vk) = client.setup(FIBONACCI_ELF);
        let mut stdin = ZKMStdin::new();
        stdin.write(&10);
        client.prove(&fibonacci_pk, stdin).compressed().run().expect("proving failed")
    })
    .await
    .expect("proving failed 1");

    let proof_2 = task::spawn_blocking(move || {
        let client = ProverClient::new();
        let (fibonacci_pk, fibonacci_vk) = client.setup(FIBONACCI_ELF);
        let mut stdin = ZKMStdin::new();
        stdin.write(&20);
        client.prove(&fibonacci_pk, stdin).compressed().run().expect("proving failed")
    })
    .await
    .expect("proving failed 2");

    // Setup the inputs to the aggregation program.
    let input_1 = AggregationInput { proof: proof_1, vk: fibonacci_vk.clone() };
    let input_2 = AggregationInput { proof: proof_2, vk: fibonacci_vk.clone() };
    let inputs = vec![input_1, input_2];

    // Aggregate the proofs.
    // tracing::info_span!("aggregate the proofs").in_scope(|| {
        let mut stdin = ZKMStdin::new();

        // Write the verification keys.
        let vkeys = inputs.iter().map(|input| input.vk.hash_u32()).collect::<Vec<_>>();
        stdin.write::<Vec<[u32; 8]>>(&vkeys);

        // Write the public values.
        let public_values =
            inputs.iter().map(|input| input.proof.public_values.to_vec()).collect::<Vec<_>>();
        stdin.write::<Vec<Vec<u8>>>(&public_values);

        // Write the proofs.
        //
        // Note: this data will not actually be read by the aggregation program, instead it will be
        // witnessed by the prover during the recursive aggregation process inside zkMIPS itself.
        for input in inputs {
            let ZKMProof::Compressed(proof) = input.proof.proof else { panic!() };
            stdin.write_proof(*proof, input.vk.vk);
        }

        let proving_start = Instant::now();

        // Generate the plonk bn254 proof.
        task::spawn_blocking(move || {
            client.prove(&aggregation_pk, stdin).compressed().run().expect("proving failed");
        })
        .await
        .expect("proving failed");

        let proving_duration = proving_start.elapsed();
        println!("Proof successfully generated! proving duration: {:?}", proving_duration);
    // });

    let duration = start.elapsed();
    println!("total duration: {:?}", duration);
}
