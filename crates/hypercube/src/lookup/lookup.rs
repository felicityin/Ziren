use core::fmt::Debug;
use std::ops::Mul;

use slop_air::{PairCol, VirtualPairCol};
use slop_algebra::{AbstractField, Field};
use slop_multilinear::MleEval;

use crate::air::LookupScope;

use super::LookupKind;

/// An interaction for a lookup or a permutation argument, as a linear (affine) combination of
/// the raw preprocessed/main trace columns.
#[derive(Clone)]
pub struct Lookup<F: Field> {
    pub values: Vec<VirtualPairCol<F>>,
    pub multiplicity: VirtualPairCol<F>,
    pub kind: LookupKind,
    pub scope: LookupScope,
}

impl<F: Field> Lookup<F> {
    pub const fn new(
        values: Vec<VirtualPairCol<F>>,
        multiplicity: VirtualPairCol<F>,
        kind: LookupKind,
        scope: LookupScope,
    ) -> Self {
        Self { values, multiplicity, kind, scope }
    }

    pub const fn argument_index(&self) -> usize {
        self.kind as usize
    }

    /// Evaluates the interaction's multiplicity and fingerprint against a concrete row.
    pub fn eval<Expr, Var>(
        &self,
        preprocessed: Option<&MleEval<Var>>,
        main: &MleEval<Var>,
        alpha: Expr,
        betas: &[Expr],
    ) -> (Expr, Expr)
    where
        F: Into<Expr>,
        Expr: AbstractField + Mul<F, Output = Expr>,
        Var: Into<Expr> + Copy,
    {
        let mut multiplicity_eval = self.multiplicity.constant.into();
        for (column, weight) in self.multiplicity.column_weights.iter() {
            let weight: Expr = (*weight).into();
            match column {
                PairCol::Preprocessed(i) => {
                    multiplicity_eval += preprocessed.as_ref().unwrap()[*i].into() * weight;
                }
                PairCol::Main(i) => multiplicity_eval += main[*i].into() * weight,
            }
        }

        let mut betas = betas.iter().cloned();
        let mut fingerprint_eval =
            alpha + betas.next().unwrap() * Expr::from_canonical_usize(self.argument_index());
        for (element, beta) in self.values.iter().zip(betas) {
            let evaluation = if let Some(preprocessed) = preprocessed {
                element.apply::<Expr, Var>(preprocessed, main)
            } else {
                element.apply::<Expr, Var>(&[], main)
            };
            fingerprint_eval += evaluation * beta;
        }

        (multiplicity_eval, fingerprint_eval)
    }
}

impl<F: Field> Debug for Lookup<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lookup").field("kind", &self.kind).field("scope", &self.scope).finish_non_exhaustive()
    }
}
