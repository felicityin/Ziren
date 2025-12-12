use zkm_sdk::{include_elf, utils, ProverClient, ZKMProofWithPublicValues, ZKMStdin};

/// The ELF we want to execute inside the zkVM.
const ELF: &[u8] = include_elf!("fibonacci");

fn main() {
    // Setup logging.
    utils::setup_logger();

    // Create a `ProverClient` method.
    let client = ProverClient::new();

    let deserialized_proof =
        ZKMProofWithPublicValues::load("proof-with-pvs.bin").expect("loading proof failed");
    let vk: zkm_sdk::ZKMVerifyingKey =
        bincode::deserialize(&std::fs::read("vk.bin").unwrap()).unwrap();

    // Verify the deserialized proof.
    client.verify(&deserialized_proof, &vk).expect("verification failed");

    println!("successfully generated and verified proof for the program!")
}
