use slop_challenger::IopCtx;
use slop_jagged::JaggedPcsVerifier;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_primitives::FriConfig;
use slop_stacked::StackedPcsVerifier;

pub type ZkmGlobalContext = KoalaBearDegree4Duplex;

pub type ZkmStackedPcs = StackedPcsVerifier<ZkmGlobalContext>;

pub type ZkmPcsVerifier = JaggedPcsVerifier<ZkmGlobalContext, ZkmStackedPcs>;

pub const NUM_ZKM_COMMITMENTS: usize = 2;

#[must_use]
pub fn default_fri_config() -> FriConfig<<ZkmGlobalContext as IopCtx>::F> {
    let num_queries = std::env::var("FRI_QUERIES").ok().and_then(|v| v.parse().ok()).unwrap_or(84);
    FriConfig::new(1, num_queries, 16)
}

#[must_use]
pub fn compressed_fri_config() -> FriConfig<<ZkmGlobalContext as IopCtx>::F> {
    let num_queries = std::env::var("FRI_QUERIES").ok().and_then(|v| v.parse().ok()).unwrap_or(42);
    FriConfig::new(2, num_queries, 16)
}

#[must_use]
pub fn ultra_compressed_fri_config() -> FriConfig<<ZkmGlobalContext as IopCtx>::F> {
    let num_queries = std::env::var("FRI_QUERIES").ok().and_then(|v| v.parse().ok()).unwrap_or(28);
    FriConfig::new(3, num_queries, 16)
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
