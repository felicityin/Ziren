use std::sync::Arc;

use slop_commit::Rounds;

use slop_algebra::Field;
use slop_multilinear::Point;

use crate::{JaggedLittlePolynomialProverParams, LongMle};

pub(crate) fn partial_jagged_multilinear<F: Field>(
    jagged_params: &JaggedLittlePolynomialProverParams,
    _row_data: Rounds<Arc<Vec<usize>>>,
    _column_data: Rounds<Arc<Vec<usize>>>,
    z_row: &Point<F>,
    z_col: &Point<F>,
    num_components: usize,
) -> LongMle<F> {
    assert_eq!(num_components, 1, "only one component is supported for now");

    let values = jagged_params.partial_jagged_little_polynomial_evaluation(z_row, z_col);
    let log_stacking_height = values.num_variables();
    LongMle::from_components(vec![values], log_stacking_height)
}
