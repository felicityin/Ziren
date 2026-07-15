#![no_std]
#![no_main]
zkm_zkvm::entrypoint!(main);

use zkm_zkvm::lib::keccak256::keccak256;

pub fn main() {
    for _ in 0..25 {
        let mut state = [1u8; 100];
        keccak256(&mut state);
        //println!("{:?}", state);
    }

    // A multi-block (>136-byte rate) input, so the KeccakSpongeBlock cross-block chaining
    // interaction is actually exercised (single-block calls above leave it provably inactive).
    let multi_block_state = [2u8; 200];
    keccak256(&multi_block_state);
}
