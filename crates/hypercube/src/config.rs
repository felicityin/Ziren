use slop_challenger::IopCtx;
use slop_jagged::JaggedPcsVerifier;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_primitives::FriConfig;
use slop_stacked::StackedPcsVerifier;

pub type ZkmGlobalContext = KoalaBearDegree4Duplex;

pub type ZkmField = <ZkmGlobalContext as IopCtx>::F;

pub type ZkmExtensionField = <ZkmGlobalContext as IopCtx>::EF;

pub type ZkmStackedPcs = StackedPcsVerifier<ZkmGlobalContext>;

pub type ZkmPcsVerifier = JaggedPcsVerifier<ZkmGlobalContext, ZkmStackedPcs>;

pub const NUM_ZKM_COMMITMENTS: usize = 2;

/// The number of bits to grind in sampling the GKR randomness.
pub const GKR_GRINDING_BITS: usize = 12;

/// The digest size (in field elements) for Ziren's Poseidon2/KoalaBear hash.
pub const DIGEST_SIZE: usize = 8;

/// The target bits of security for FRI's query phase, under the unique-decoding-radius (UDR)
/// bound. Mirrors SP1's `SP1_TARGET_BITS_OF_SECURITY`.
pub const ZKM_TARGET_BITS_OF_SECURITY: usize = 100;

/// The number of proof-of-work grinding bits added to the Fiat-Shamir transcript before FRI
/// query sampling. Each grinding bit reduces the number of queries needed for a given target
/// security level by one bit's worth. Mirrors SP1's `SP1_PROOF_OF_WORK_BITS`.
pub const ZKM_PROOF_OF_WORK_BITS: usize = 16;

/// The log2 of FRI's blowup (rate inverse) for the core (MIPS) machine.
pub const DEFAULT_LOG_BLOWUP: usize = 1;

/// The log2 of FRI's blowup (rate inverse) for the compress/recursion machines.
pub const COMPRESSED_LOG_BLOWUP: usize = 2;

/// The log2 of FRI's blowup (rate inverse) for the shrink/wrap machines.
pub const ULTRA_COMPRESSED_LOG_BLOWUP: usize = 3;

/// The number of FRI queries needed to reach [`ZKM_TARGET_BITS_OF_SECURITY`] bits of security
/// (under the unique-decoding-radius bound) at a given blowup factor and grinding budget.
///
/// Each query independently catches a cheating prover except with probability `(1 + rho) / 2`,
/// where `rho = 2^-log_blowup` is the code rate; grinding contributes `grinding_bits` bits of
/// security "for free" via proof-of-work, reducing the number of queries needed to cover the
/// remaining `target_bits - grinding_bits`. Mirrors SP1's
/// `unique_decoding_queries_with_custom_grinding`.
#[must_use]
pub fn unique_decoding_queries_with_custom_grinding(
    log_blowup: usize,
    grinding_bits: usize,
) -> usize {
    let rate = 1.0 / (1u64 << log_blowup) as f64;
    let half_rate_plus_half = 0.5 + (rate / 2.0);
    (-((ZKM_TARGET_BITS_OF_SECURITY - grinding_bits) as f64) / half_rate_plus_half.log2()).ceil()
        as usize
}

/// [`unique_decoding_queries_with_custom_grinding`] at the default grinding budget
/// ([`ZKM_PROOF_OF_WORK_BITS`]).
#[must_use]
pub fn unique_decoding_queries(log_blowup: usize) -> usize {
    unique_decoding_queries_with_custom_grinding(log_blowup, ZKM_PROOF_OF_WORK_BITS)
}

#[must_use]
pub fn default_fri_config() -> FriConfig<<ZkmGlobalContext as IopCtx>::F> {
    let num_queries = std::env::var("FRI_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| unique_decoding_queries(DEFAULT_LOG_BLOWUP));
    FriConfig::new(DEFAULT_LOG_BLOWUP, num_queries, ZKM_PROOF_OF_WORK_BITS)
}

#[must_use]
pub fn compressed_fri_config() -> FriConfig<<ZkmGlobalContext as IopCtx>::F> {
    let num_queries = std::env::var("FRI_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| unique_decoding_queries(COMPRESSED_LOG_BLOWUP));
    FriConfig::new(COMPRESSED_LOG_BLOWUP, num_queries, ZKM_PROOF_OF_WORK_BITS)
}

#[must_use]
pub fn ultra_compressed_fri_config() -> FriConfig<<ZkmGlobalContext as IopCtx>::F> {
    let num_queries = std::env::var("FRI_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| unique_decoding_queries(ULTRA_COMPRESSED_LOG_BLOWUP));
    FriConfig::new(ULTRA_COMPRESSED_LOG_BLOWUP, num_queries, ZKM_PROOF_OF_WORK_BITS)
}

#[must_use]
pub fn zkm_pcs_verifier(
    fri_config: FriConfig<<ZkmGlobalContext as IopCtx>::F>,
    log_stacking_height: u32,
    max_log_row_count: usize,
) -> ZkmPcsVerifier {
    ZkmPcsVerifier::new_from_basefold_params(
        fri_config,
        log_stacking_height,
        max_log_row_count,
        NUM_ZKM_COMMITMENTS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_decoding_queries_matches_target_bits() {
        assert_eq!(unique_decoding_queries(DEFAULT_LOG_BLOWUP), 203);
        assert_eq!(unique_decoding_queries(COMPRESSED_LOG_BLOWUP), 124);
        assert_eq!(unique_decoding_queries(ULTRA_COMPRESSED_LOG_BLOWUP), 102);
    }

    #[test]
    fn derived_fri_configs_route_through_unique_decoding_queries() {
        // Skipped rather than asserted-clean-env: FRI_QUERIES may legitimately be set by the
        // surrounding test run (e.g. for fast iteration), in which case this isn't the right
        // test to catch a routing bug -- unique_decoding_queries_matches_target_bits already
        // covers the formula itself independent of the env override.
        if std::env::var("FRI_QUERIES").is_ok() {
            return;
        }
        assert_eq!(default_fri_config().num_queries, 203);
        assert_eq!(compressed_fri_config().num_queries, 124);
        assert_eq!(ultra_compressed_fri_config().num_queries, 102);
    }
}
