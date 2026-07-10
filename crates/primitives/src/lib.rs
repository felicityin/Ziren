//! zkm-primitives contains types and functions that are used in both zkm-core and zkm-zkvm.
//! Because it is imported in the zkvm entrypoint, it should be kept minimal.

use lazy_static::lazy_static;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
//use p3_monty_31::{Poseidon2InternalLayerMonty31, Poseidon2ExternalLayerMonty31};

pub mod consts;
pub mod io;
pub mod types;

/// The canonical KoalaBear/α=3 Poseidon2 permutation (R_F=8, R_P=20), shared by every consumer in
/// the workspace (guest Poseidon2 precompile, recursion VM, and `zkm_hypercube::ZkmGlobalContext`'s
/// Merkle hashing/Fiat-Shamir transcript). Delegates directly to `slop_koala_bear::my_kb_16_perm`
/// so there is exactly one KoalaBear Poseidon2 round-constant table in the workspace.
pub fn poseidon2_init() -> Poseidon2KoalaBear<16> {
    slop_koala_bear::my_kb_16_perm()
}

use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};

pub fn poseidon2_hash(input: Vec<KoalaBear>) -> [KoalaBear; 8] {
    POSEIDON2_HASHER.hash_iter(input)
}

pub fn poseidon2_hasher() -> PaddingFreeSponge<Poseidon2KoalaBear<16>, 16, 8, 8> {
    let hasher = poseidon2_init();
    PaddingFreeSponge::<Poseidon2KoalaBear<16>, 16, 8, 8>::new(hasher)
}

lazy_static! {
    pub static ref POSEIDON2_HASHER: PaddingFreeSponge::<Poseidon2KoalaBear<16>, 16, 8, 8> =
        poseidon2_hasher();
}

/// Append a single deferred proof to a hash chain of deferred proofs.
pub fn hash_deferred_proof(
    prev_digest: &[KoalaBear; 8],
    vk_digest: &[KoalaBear; 8],
    pv_digest: &[KoalaBear; 32],
) -> [KoalaBear; 8] {
    let mut inputs = Vec::with_capacity(48);
    inputs.extend_from_slice(prev_digest);
    inputs.extend_from_slice(vk_digest);
    inputs.extend_from_slice(pv_digest);
    poseidon2_hash(inputs.to_vec())
}

#[cfg(test)]
mod tests {
    use p3_field::FieldAlgebra;
    use p3_poseidon2::poseidon2_round_numbers_128;
    use p3_symmetric::Permutation;

    use super::*;

    /// Ziren instantiates Poseidon2 over KoalaBear (width 16) with S-box degree α=3 (see
    /// `poseidon2_init`). The partial- (internal-) round count must match Plonky3's own 128-bit
    /// recommendation for that configuration; a lower-degree S-box needs *more* partial rounds for
    /// algebraic-attack resistance, so under-counting silently weakens the hash.
    #[test]
    fn koalabear_poseidon2_round_counts_match_plonky3_reference() {
        let (rounds_f, rounds_p) = poseidon2_round_numbers_128::<KoalaBear>(16, 3);
        assert_eq!(rounds_f, 8, "external (full) round count drifted from the Plonky3 reference");
        assert_eq!(rounds_p, 20, "internal (partial) round count drifted from the Plonky3 reference");
    }

    /// `poseidon2_init()` must be byte-for-byte the same permutation as
    /// `slop_koala_bear::my_kb_16_perm()`: it's the reference every other KoalaBear Poseidon2
    /// instantiation in the workspace (guest precompile AIR, recursion VM chips,
    /// `ZkmGlobalContext`'s Merkle hasher/challenger) is checked against, and any drift between them
    /// breaks in-circuit hash verification against out-of-circuit commitments.
    #[test]
    fn poseidon2_init_matches_slop_koala_bear_reference() {
        let input: [KoalaBear; 16] =
            std::array::from_fn(|i| KoalaBear::from_canonical_u32(i as u32 + 1));

        let expected = slop_koala_bear::my_kb_16_perm().permute(input);
        let actual = poseidon2_init().permute(input);
        assert_eq!(actual, expected);
    }
}
