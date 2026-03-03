use zkm_sdk::{utils, ProverClient, ZKMStdin};

pub const RETH_ELF: &[u8] = include_bytes!("../../guest/reth");
pub const RETH_STDIN: &[u8] = include_bytes!("../../guest/stdin-24438200");

fn main() {
    // Setup a tracer for logging.
    utils::setup_logger();

    let stdin: ZKMStdin = bincode::deserialize(RETH_STDIN).unwrap();

    let client = ProverClient::new();
    client.execute(RETH_ELF, stdin).run().unwrap();

    println!("successfully!");
}
