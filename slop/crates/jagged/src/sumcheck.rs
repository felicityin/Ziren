use std::sync::Arc;

use slop_algebra::{ExtensionField, Field};
use slop_commit::{Message, Rounds};
use slop_multilinear::{Mle, Point};
use slop_stacked::interleave_multilinears_with_fixed_rate;

use crate::{
    populate::partial_jagged_multilinear, HadamardProduct, JaggedLittlePolynomialProverParams,
    LongMle,
};

pub(crate) fn jagged_sumcheck_poly<F: Field, EF: ExtensionField<F>>(
    base: Rounds<Message<Mle<F>>>,
    jagged_params: &JaggedLittlePolynomialProverParams,
    row_data: Rounds<Arc<Vec<usize>>>,
    column_data: Rounds<Arc<Vec<usize>>>,
    log_stacking_height: u32,
    z_row: &Point<EF>,
    z_col: &Point<EF>,
) -> HadamardProduct<F, EF> {
    let base = base.into_iter().flatten().collect::<Message<Mle<_, _>>>();
    let long_mle = LongMle::from_message(base, log_stacking_height);
    let jaggled_mle =
        partial_jagged_multilinear(jagged_params, row_data, column_data, z_row, z_col, 1);

    let total_num_variables = jaggled_mle.num_variables();

    let restacked_mle = LongMle::from_message(
        interleave_multilinears_with_fixed_rate(
            1,
            long_mle.components().clone(),
            total_num_variables,
        ),
        total_num_variables,
    );

    HadamardProduct { base: restacked_mle, ext: jaggled_mle }
}
