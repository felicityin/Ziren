use slop_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder};
use slop_algebra::{ExtensionField, Field};
use slop_matrix::dense::RowMajorMatrixView;

use crate::air::EmptyMessageBuilder;

/// A builder for debugging constraints.
pub struct DebugConstraintBuilder<'a, F: Field, EF: ExtensionField<F>> {
    pub preprocessed: RowMajorMatrixView<'a, F>,
    pub main: RowMajorMatrixView<'a, F>,
    pub public_values: &'a [F],
    pub failing_constraints: Vec<usize>,
    pub num_constraints_evaluated: usize,
    pub phantom: std::marker::PhantomData<EF>,
}

impl<F, EF> ExtensionBuilder for DebugConstraintBuilder<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type EF = EF;
    type VarEF = EF;
    type ExprEF = EF;

    fn assert_zero_ext<I>(&mut self, _x: I)
    where
        I: Into<Self::ExprEF>,
    {
        panic!("extension fields are not supported in the debug builder; traces are over the base field");
    }
}

impl<'a, F, EF> PermutationAirBuilder for DebugConstraintBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type MP = RowMajorMatrixView<'a, EF>;
    type RandomVar = EF;

    fn permutation(&self) -> Self::MP {
        unimplemented!()
    }

    fn permutation_randomness(&self) -> &[Self::EF] {
        unimplemented!()
    }
}

impl<F, EF> PairBuilder for DebugConstraintBuilder<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<F, EF> DebugConstraintBuilder<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    #[allow(clippy::unused_self)]
    #[inline]
    fn debug_constraint(&mut self, x: F, y: F) {
        if x != y {
            self.failing_constraints.push(self.num_constraints_evaluated);
        }
        self.num_constraints_evaluated += 1;
    }
}

impl<'a, F, EF> AirBuilder for DebugConstraintBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = RowMajorMatrixView<'a, F>;

    fn is_first_row(&self) -> Self::Expr {
        unimplemented!()
    }

    fn is_last_row(&self) -> Self::Expr {
        unimplemented!()
    }

    fn is_transition_window(&self, _size: usize) -> Self::Expr {
        unimplemented!()
    }

    fn main(&self) -> Self::M {
        self.main
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.debug_constraint(x.into(), F::zero());
    }

    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        self.debug_constraint(x.into(), F::one());
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        self.debug_constraint(x.into(), y.into());
    }

    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        if x != F::zero() && x != F::one() {
            self.failing_constraints.push(self.num_constraints_evaluated);
        }
        self.num_constraints_evaluated += 1;
    }
}

impl<F: Field, EF: ExtensionField<F>> EmptyMessageBuilder for DebugConstraintBuilder<'_, F, EF> {}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues for DebugConstraintBuilder<'_, F, EF> {
    type PublicVar = F;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}
