use std::{
    marker::PhantomData,
    ops::{Add, Mul, MulAssign, Sub},
};

use slop_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder};
use slop_algebra::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use slop_matrix::dense::RowMajorMatrixView;

use crate::air::EmptyMessageBuilder;

pub type VerifierConstraintFolder<'a, F, EF> = GenericVerifierConstraintFolder<'a, F, EF, F, EF, EF>;

pub struct GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr> {
    pub preprocessed: RowMajorMatrixView<'a, Var>,
    pub main: RowMajorMatrixView<'a, Var>,
    pub alpha: Var,
    pub accumulator: Expr,
    pub public_values: &'a [PubVar],
    pub _marker: PhantomData<(F, EF)>,
}

impl<'a, F, EF, PubVar, Var, Expr> AirBuilder for GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
    PubVar: Into<Expr> + Copy,
{
    type F = F;
    type Expr = Expr;
    type Var = Var;
    type M = RowMajorMatrixView<'a, Var>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        unimplemented!()
    }

    fn is_last_row(&self) -> Self::Expr {
        unimplemented!()
    }

    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unimplemented!()
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: Expr = x.into();
        self.accumulator *= self.alpha.into();
        self.accumulator += x;
    }
}

impl<F, EF, PubVar, Var, Expr> ExtensionBuilder for GenericVerifierConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
    PubVar: Into<Expr> + Copy,
{
    type EF = EF;
    type ExprEF = Expr;
    type VarEF = Var;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_zero(x);
    }
}

impl<'a, F, EF, PubVar, Var, Expr> PermutationAirBuilder for GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
    PubVar: Into<Expr> + Copy,
{
    type MP = RowMajorMatrixView<'a, Var>;
    type RandomVar = Var;

    fn permutation(&self) -> Self::MP {
        unimplemented!();
    }

    fn permutation_randomness(&self) -> &[Self::Var] {
        unimplemented!()
    }
}

impl<F, EF, PubVar, Var, Expr> PairBuilder for GenericVerifierConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
    PubVar: Into<Expr> + Copy,
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<F, EF, PubVar, Var, Expr> EmptyMessageBuilder for GenericVerifierConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
    PubVar: Into<Expr> + Copy,
{
}

impl<F, EF, PubVar, Var, Expr> AirBuilderWithPublicValues for GenericVerifierConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
    PubVar: Into<Expr> + Copy,
{
    type PublicVar = PubVar;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

/// A folder for the zerocheck sumcheck poly.
pub struct ConstraintSumcheckFolder<'a, F: Field, K: Field, EF> {
    pub preprocessed: RowMajorMatrixView<'a, K>,
    pub main: RowMajorMatrixView<'a, K>,
    pub powers_of_alpha: &'a [EF],
    pub accumulator: EF,
    pub public_values: &'a [F],
    pub constraint_index: usize,
}

impl<
        'a,
        F: Field,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        EF: Field + Mul<K, Output = EF>,
    > AirBuilder for ConstraintSumcheckFolder<'a, F, K, EF>
{
    type F = F;
    type Expr = K;
    type Var = K;
    type M = RowMajorMatrixView<'a, K>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        unimplemented!()
    }

    fn is_last_row(&self) -> Self::Expr {
        unimplemented!()
    }

    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unimplemented!()
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator += self.powers_of_alpha[self.constraint_index] * x.into();
        self.constraint_index += 1;
    }
}

impl<
        F: Field,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        EF: Field + Mul<K, Output = EF> + ExtensionField<F> + AbstractExtensionField<K> + From<K>,
    > ExtensionBuilder for ConstraintSumcheckFolder<'_, F, K, EF>
{
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.accumulator += self.powers_of_alpha[self.constraint_index] * x.into();
        self.constraint_index += 1;
    }
}

impl<
        'a,
        F: Field,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        EF: Field + Mul<K, Output = EF> + ExtensionField<F> + AbstractExtensionField<K>,
    > PermutationAirBuilder for ConstraintSumcheckFolder<'a, F, K, EF>
{
    type MP = RowMajorMatrixView<'a, EF>;
    type RandomVar = EF;

    fn permutation(&self) -> Self::MP {
        unimplemented!()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        unimplemented!()
    }
}

impl<
        F: Field,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        EF: Field + Mul<K, Output = EF>,
    > PairBuilder for ConstraintSumcheckFolder<'_, F, K, EF>
{
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<
        F: Field,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        EF: Field + Mul<K, Output = EF>,
    > AirBuilderWithPublicValues for ConstraintSumcheckFolder<'_, F, K, EF>
{
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<
        F: Field,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        EF: Field + Mul<K, Output = EF>,
    > EmptyMessageBuilder for ConstraintSumcheckFolder<'_, F, K, EF>
{
}
