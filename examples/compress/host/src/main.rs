//! A simple example showing how to aggregate proofs of multiple programs with ZKM.

use zkm_prover::components::DefaultProverComponents;
use zkm_sdk::{
    include_elf, HashableKey, ProverClient, ZKMProof, ZKMProofWithPublicValues, ZKMStdin,
    ZKMVerifyingKey, provers::ProofOpts, install::try_install_circuit_artifacts, ZKMProver,
};

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

    let opts = ProofOpts::default();

    let prover: ZKMProver<DefaultProverComponents> = ZKMProver::new();

    let ZKMProof::Compressed(proof) = proof_1.proof else { panic!() };

    // Generate the shrink proof.
    let compress_proof = prover.shrink(*proof, opts.zkm_prover_opts).unwrap();

    // Genenerate the wrap proof.
    let outer_proof = prover.wrap_bn254(compress_proof, opts.zkm_prover_opts).unwrap();
    println!("wrap_bn254 done");

    let groth16_bn254_artifacts = try_install_circuit_artifacts("groth16");

    let groth16_proof = prover.wrap_groth16_bn254(outer_proof, &groth16_bn254_artifacts);
    println!("wrap_groth16_bn254 done");
}
