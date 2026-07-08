use std::{
    marker::PhantomData,
    ops::{Add, Div, Mul, MulAssign, Sub},
};

use slop_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder};
use slop_algebra::{FieldExtensionAlgebra, FieldAlgebra, ExtensionField, Field};
use slop_challenger::IopCtx;
use slop_matrix::dense::RowMajorMatrixView;

use crate::air::{AirLookup, EmptyMessageBuilder, LookupScope, MessageBuilder};

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
    Expr: FieldAlgebra
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
    Expr: FieldAlgebra<F = EF>
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
    Expr: FieldAlgebra<F = EF>
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
    Expr: FieldAlgebra<F = EF>
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
    Expr: FieldAlgebra<F = EF>
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
    Expr: FieldAlgebra<F = EF>
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
        EF: Field + Mul<K, Output = EF> + ExtensionField<F> + FieldExtensionAlgebra<K> + From<K>,
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
        EF: Field + Mul<K, Output = EF> + ExtensionField<F> + FieldExtensionAlgebra<K>,
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

/// A folder for verifying an AIR's contribution to the public-values digest, computed from the
/// LogUp permutation-argument fingerprints of the lookups it sends/receives rather than by
/// re-deriving them from a jagged/basefold opening.
pub type VerifierPublicValuesConstraintFolder<'a, C> =
    GenericVerifierPublicValuesConstraintFolder<'a, <C as IopCtx>::F, <C as IopCtx>::EF, <C as IopCtx>::F, <C as IopCtx>::EF, <C as IopCtx>::EF>;

pub struct GenericVerifierPublicValuesConstraintFolder<'a, F, EF, PubVar, Var, Expr> {
    pub perm_challenges: (&'a Var, &'a [Expr]),
    pub alpha: Var,
    pub accumulator: Expr,
    pub public_values: &'a [PubVar],
    pub local_interaction_digest: Expr,
    pub _marker: PhantomData<(F, EF)>,
}

impl<'a, F, EF, PubVar, Var, Expr> AirBuilder for GenericVerifierPublicValuesConstraintFolder<'a, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: FieldAlgebra<F = EF>
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
        unimplemented!()
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

impl<F, EF, PubVar, Var, Expr> ExtensionBuilder for GenericVerifierPublicValuesConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: FieldAlgebra<F = EF>
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

impl<'a, F, EF, PubVar, Var, Expr> PermutationAirBuilder for GenericVerifierPublicValuesConstraintFolder<'a, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: FieldAlgebra<F = EF>
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
        unimplemented!()
    }

    fn permutation_randomness(&self) -> &[Self::Var] {
        unimplemented!()
    }
}

impl<F, EF, PubVar, Var, Expr> PairBuilder for GenericVerifierPublicValuesConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: FieldAlgebra<F = EF>
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
        unimplemented!()
    }
}

impl<F, EF, PubVar, Var, Expr> MessageBuilder<AirLookup<Expr>> for GenericVerifierPublicValuesConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: FieldAlgebra<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>
        + Div<Expr, Output = Expr>,
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
    fn send(&mut self, message: AirLookup<Expr>, _scope: LookupScope) {
        let mut denominator: Expr = (*self.perm_challenges.0).into();
        let mut betas = self.perm_challenges.1.iter().cloned();
        denominator += betas.next().unwrap() * F::from_canonical_usize(message.kind as usize);
        for value in message.values {
            denominator += value * betas.next().unwrap();
        }
        let digest = message.multiplicity / denominator;
        self.local_interaction_digest += digest;
    }

    fn receive(&mut self, message: AirLookup<Expr>, _scope: LookupScope) {
        let mut denominator: Expr = (*self.perm_challenges.0).into();
        let mut betas = self.perm_challenges.1.iter().cloned();
        denominator += betas.next().unwrap() * F::from_canonical_usize(message.kind as usize);
        for value in message.values {
            denominator += value * betas.next().unwrap();
        }
        let digest = message.multiplicity / denominator;
        self.local_interaction_digest -= digest;
    }
}

impl<F, EF, PubVar, Var, Expr> AirBuilderWithPublicValues for GenericVerifierPublicValuesConstraintFolder<'_, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: FieldAlgebra<F = EF>
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
