//! A simple example showing how to aggregate proofs of multiple programs with ZKM.

use std::borrow::BorrowMut;
use std::collections::BTreeMap;

use zkm_sdk::{
    include_elf, HashableKey, ProverClient, ZKMProof, ZKMProofWithPublicValues, ZKMStdin,
    ZKMVerifyingKey,
};
use zkm_recursion_circuit::{
    // config::InnerSC,
    merkle_tree::MerkleTree,
};
use zkm_recursion_core::air::RecursionPublicValues;
use zkm_stark::DIGEST_SIZE;
use zkm_prover::InnerSC;
use p3_koala_bear::KoalaBear;

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

fn main() {
    // Setup the logger.
    zkm_sdk::utils::setup_logger();

    pub static VK_MAP: &'static [u8] = include_bytes!("../vk_map.bin");
    let allowed_vk_map: BTreeMap<[KoalaBear; DIGEST_SIZE], usize> =
        bincode::deserialize(&VK_MAP).unwrap();
    let (recursion_vk_root, _merkle_tree) =  MerkleTree::<KoalaBear, InnerSC>::commit(allowed_vk_map.keys().copied().collect());

    // Initialize the proving client.
    let client = ProverClient::new();

    // Setup the proving and verifying keys.
    let (aggregation_pk, _) = client.setup(AGGREGATION_ELF);

    // Generate the fibonacci proofs.
    let mut proof_1 = ZKMProofWithPublicValues::load("proof-with-pvs.bin").expect("loading proof failed");

    let ZKMProof::Compressed(mut proof) = proof_1.proof else { panic!() };
    let mut public_values: &mut RecursionPublicValues<_> =
        proof.proof.public_values.as_mut_slice().borrow_mut();
    public_values.vk_root = recursion_vk_root;

    proof_1 = ZKMProofWithPublicValues {
        proof: ZKMProof::Compressed(proof),
        public_values: proof_1.public_values,
        zkm_version: "v1.2.3".to_string(),
    };

    let fibonacci_vk: zkm_sdk::ZKMVerifyingKey =
        bincode::deserialize(&std::fs::read("vk.bin").unwrap()).unwrap();

    // Setup the inputs to the aggregation program.
    let input_1 = AggregationInput { proof: proof_1, vk: fibonacci_vk.clone() };
    let inputs = vec![input_1];

    // Aggregate the proofs.
    tracing::info_span!("aggregate the proofs").in_scope(|| {
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
        // witnessed by the prover during the recursive aggregation process inside Ziren itself.
        for input in inputs {
            let ZKMProof::Compressed(proof) = input.proof.proof else { panic!() };
            stdin.write_proof(*proof, input.vk.vk);
        }

        // Generate the proof.
        client.prove(&aggregation_pk, stdin).run().expect("proving failed");
    });
}
