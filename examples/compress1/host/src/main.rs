//! A simple example showing how to aggregate proofs of multiple programs with ZKM.

use zkm_prover::components::DefaultProverComponents;
use zkm_sdk::{
    include_elf, ProverClient, ZKMProof, ZKMStdin, ZKMProver,
};
use zkm_stark::ZKMProverOpts;

/// A program that just runs a simple computation.
const FIBONACCI_ELF: &[u8] = include_elf!("fibonacci");

fn main() {
    // Setup the logger.
    zkm_sdk::utils::setup_logger();

    // Initialize the proving client.
    let client = ProverClient::new();

    // Setup the proving and verifying keys.
    let (fibonacci_pk, fibonacci_vk) = client.setup(FIBONACCI_ELF);

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

    let ZKMProof::Compressed(proof1) = proof_1.proof else { panic!() };
    let ZKMProof::Compressed(proof2) = proof_2.proof else { panic!() };

    let opts = ZKMProverOpts::default();

    let prover: ZKMProver<DefaultProverComponents> = ZKMProver::new();

    prover.compress1(
        &fibonacci_vk,
        vec![*proof1, *proof2],
        opts,
    ).unwrap();
}
