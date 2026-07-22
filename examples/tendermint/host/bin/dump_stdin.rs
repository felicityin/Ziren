//! Dumps the same two real, CBOR-encoded light-block fixtures `examples/tendermint/host`'s own
//! `main.rs` feeds the guest via stdin, as raw `.bin` files next to the compiled guest ELF -- for
//! a `zkm-core-machine` debug test to load directly (that test can't depend on
//! `tendermint-light-client-verifier`/`serde_cbor` itself without pulling them into the machine
//! crate).

use std::io::Write;

#[path = "../src/util.rs"]
mod util;

fn main() {
    let light_block_1 = util::load_light_block(2279100).expect("Failed to load light block 1");
    let light_block_2 = util::load_light_block(2279130).expect("Failed to load light block 2");

    let encoded_1 = serde_cbor::to_vec(&light_block_1).unwrap();
    let encoded_2 = serde_cbor::to_vec(&light_block_2).unwrap();

    std::fs::File::create("../../target/tendermint_stdin_1.bin")
        .unwrap()
        .write_all(&encoded_1)
        .unwrap();
    std::fs::File::create("../../target/tendermint_stdin_2.bin")
        .unwrap()
        .write_all(&encoded_2)
        .unwrap();

    println!("wrote {} + {} bytes", encoded_1.len(), encoded_2.len());
}
