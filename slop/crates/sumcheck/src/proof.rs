use serde::{Deserialize, Serialize};
use slop_algebra::{FieldAlgebra, UnivariatePolynomial};
use slop_multilinear::Point;

/// A sumchexckl proof that does not include the evaluation proofs.
///
/// Verifying a partial sumcheck proof is equivalent to verifying the sumcheck claim on the
/// condition of having evaluation proofs for the given componment polynomials at the given points.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PartialSumcheckProof<K> {
    pub univariate_polys: Vec<UnivariatePolynomial<K>>,
    pub claimed_sum: K,
    pub point_and_eval: (Point<K>, K),
}

impl<K: FieldAlgebra> PartialSumcheckProof<K> {
    /// Creates a dummy sumcheck proof with the given number of variables and degree.
    ///
    /// NOTE: ONLY USE THIS FOR TESTING AND MOCK PROOF CREATION.
    #[must_use]
    pub fn dummy() -> Self {
        Self {
            univariate_polys: Vec::new(),
            claimed_sum: K::zero(),
            point_and_eval: (Point::<K>::from_usize(0, 0), K::zero()),
        }
    }
}
