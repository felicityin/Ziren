use zkm_sdk::{utils, ProverClient, ZKMStdin};

pub const GOAT_ELF: &[u8] = include_bytes!("../../guest/goat");
pub const GOAT_STDIN: &[u8] = include_bytes!("../../guest/stdin");

fn main() {
    // Setup a tracer for logging.
    utils::setup_logger();

    let stdin: ZKMStdin = bincode::deserialize(GOAT_STDIN).unwrap();

    let client = ProverClient::new();
    client.execute(GOAT_ELF, stdin.clone()).run().unwrap();
    println!("execute successfully!");

     // Generate the proof for the given guest and input.
    let (pk, vk) = client.setup(GOAT_ELF);
    let proof = client.prove(&pk, stdin).run().unwrap();
    println!("generated proof");

    // Verify proof and public values
    client.verify(&proof, &vk).expect("verification failed");
    println!("verify successfully!");
}
