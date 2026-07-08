use slop_air::{AirBuilder, AirBuilderWithPublicValues, PairBuilder, PairCol, VirtualPairCol};
use slop_algebra::Field;
use slop_matrix::dense::RowMajorMatrix;
use slop_uni_stark::{Entry, SymbolicExpression, SymbolicVariable};

use crate::{
    air::{AirLookup, LookupScope, MessageBuilder},
    PROOF_MAX_NUM_PVS,
};

use super::Lookup;

/// A builder that symbolically evaluates an AIR to extract its lookup interactions as
/// affine (degree <= 1) combinations of the preprocessed/main columns.
pub struct InteractionBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    sends: Vec<Lookup<F>>,
    receives: Vec<Lookup<F>>,
    public_values: Vec<F>,
}

impl<F: Field> InteractionBuilder<F> {
    #[must_use]
    pub fn new(preprocessed_width: usize, main_width: usize) -> Self {
        let preprocessed_width = preprocessed_width.max(1);
        let prep_values = (0..preprocessed_width)
            .map(move |column| SymbolicVariable::new(Entry::Preprocessed { offset: 0 }, column))
            .collect();

        let main_values = (0..main_width)
            .map(move |column| SymbolicVariable::new(Entry::Main { offset: 0 }, column))
            .collect();

        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, main_width),
            sends: vec![],
            receives: vec![],
            public_values: vec![F::zero(); PROOF_MAX_NUM_PVS],
        }
    }

    #[must_use]
    pub fn interactions(self) -> (Vec<Lookup<F>>, Vec<Lookup<F>>) {
        (self.sends, self.receives)
    }
}

impl<F: Field> AirBuilder for InteractionBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        unimplemented!();
    }

    fn is_last_row(&self) -> Self::Expr {
        unimplemented!();
    }

    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unimplemented!();
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {}
}

impl<F: Field> PairBuilder for InteractionBuilder<F> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field> MessageBuilder<AirLookup<SymbolicExpression<F>>> for InteractionBuilder<F> {
    fn send(&mut self, message: AirLookup<SymbolicExpression<F>>, scope: LookupScope) {
        let values = message.values.into_iter().map(|v| symbolic_to_virtual_pair(&v)).collect::<Vec<_>>();
        let multiplicity = symbolic_to_virtual_pair(&message.multiplicity);
        self.sends.push(Lookup::new(values, multiplicity, message.kind, scope));
    }

    fn receive(&mut self, message: AirLookup<SymbolicExpression<F>>, scope: LookupScope) {
        let values = message.values.into_iter().map(|v| symbolic_to_virtual_pair(&v)).collect::<Vec<_>>();
        let multiplicity = symbolic_to_virtual_pair(&message.multiplicity);
        self.receives.push(Lookup::new(values, multiplicity, message.kind, scope));
    }
}

impl<F: Field> AirBuilderWithPublicValues for InteractionBuilder<F> {
    type PublicVar = F;

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field> crate::air::OperationSummaryAirBuilder for InteractionBuilder<F> {}

fn symbolic_to_virtual_pair<F: Field>(expression: &SymbolicExpression<F>) -> VirtualPairCol<F> {
    if expression.degree_multiple() > 1 {
        panic!("degree multiple is too high");
    }

    let (column_weights, constant) = eval_symbolic_to_virtual_pair(expression);

    let column_weights = column_weights.into_iter().collect();

    VirtualPairCol::new(column_weights, constant)
}

fn eval_symbolic_to_virtual_pair<F: Field>(expression: &SymbolicExpression<F>) -> (Vec<(PairCol, F)>, F) {
    match expression {
        SymbolicExpression::Constant(c) => (vec![], *c),
        SymbolicExpression::Variable(v) => match v.entry {
            Entry::Preprocessed { offset: 0 } => (vec![(PairCol::Preprocessed(v.index), F::one())], F::zero()),
            Entry::Main { offset: 0 } => (vec![(PairCol::Main(v.index), F::one())], F::zero()),
            _ => panic!("not an affine expression in current row elements {:?}", v.entry),
        },
        SymbolicExpression::Add { x, y, .. } => {
            let (v_l, c_l) = eval_symbolic_to_virtual_pair(x);
            let (v_r, c_r) = eval_symbolic_to_virtual_pair(y);
            ([v_l, v_r].concat(), c_l + c_r)
        }
        SymbolicExpression::Sub { x, y, .. } => {
            let (v_l, c_l) = eval_symbolic_to_virtual_pair(x);
            let (v_r, c_r) = eval_symbolic_to_virtual_pair(y);
            let neg_v_r = v_r.iter().map(|(c, w)| (*c, -*w)).collect();
            ([v_l, neg_v_r].concat(), c_l - c_r)
        }
        SymbolicExpression::Neg { x, .. } => {
            let (v, c) = eval_symbolic_to_virtual_pair(x);
            (v.iter().map(|(c, w)| (*c, -*w)).collect(), -c)
        }
        SymbolicExpression::Mul { x, y, .. } => {
            let (v_l, c_l) = eval_symbolic_to_virtual_pair(x);
            let (v_r, c_r) = eval_symbolic_to_virtual_pair(y);

            let mut v = vec![];
            v.extend(v_l.iter().map(|(c, w)| (*c, *w * c_r)));
            v.extend(v_r.iter().map(|(c, w)| (*c, *w * c_l)));

            if !v_l.is_empty() && !v_r.is_empty() {
                panic!("Not an affine expression")
            }

            (v, c_l * c_r)
        }
        SymbolicExpression::IsFirstRow => panic!("not an affine expression in current row elements for first row"),
        SymbolicExpression::IsLastRow => panic!("not an affine expression in current row elements for last row"),
        SymbolicExpression::IsTransition => {
            panic!("not an affine expression in current row elements for transition row")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;

    use slop_air::{Air, BaseAir};
    use slop_algebra::FieldAlgebra;
    use slop_koala_bear::KoalaBear;
    use slop_matrix::Matrix;

    use super::*;
    use crate::{air::ZKMAirBuilder, lookup::LookupKind};

    #[test]
    fn test_symbolic_to_virtual_pair_col() {
        type F = KoalaBear;

        let x = SymbolicVariable::<F>::new(Entry::Main { offset: 0 }, 0);
        let y = SymbolicVariable::<F>::new(Entry::Main { offset: 0 }, 1);
        let z = x + y;

        let (column_weights, constant) = super::eval_symbolic_to_virtual_pair(&z);
        let column_weights = column_weights.into_iter().collect::<Vec<_>>();
        let z = VirtualPairCol::new(column_weights, constant);

        let expr: F = z.apply(&[], &[F::one(), F::one()]);
        assert_eq!(expr, F::two());
    }

    struct LookupTestAir;

    const NUM_COLS: usize = 3;

    impl<F: Field> BaseAir<F> for LookupTestAir {
        fn width(&self) -> usize {
            NUM_COLS
        }
    }

    impl<AB: ZKMAirBuilder> Air<AB> for LookupTestAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0);
            let local: &[AB::Var] = (*local).borrow();

            let x = local[0];
            let y = local[1];
            let z = local[2];

            builder.send(
                AirLookup::new(vec![x.into(), y.into()], AB::F::from_canonical_u32(3).into(), LookupKind::Memory),
                LookupScope::Local,
            );
            builder.send(
                AirLookup::new(vec![x + y, z.into()], AB::F::from_canonical_u32(5).into(), LookupKind::Memory),
                LookupScope::Local,
            );

            builder.receive(AirLookup::new(vec![x.into()], y.into(), LookupKind::Byte), LookupScope::Local);
        }
    }

    #[test]
    fn test_lookup_interactions() {
        let air = LookupTestAir;

        let mut builder = InteractionBuilder::<KoalaBear>::new(0, NUM_COLS);
        air.eval(&mut builder);

        let mut main = builder.main();
        let (sends, receives) = builder.interactions();

        assert_eq!(sends.len(), 2);
        assert_eq!(receives.len(), 1);

        assert_eq!(sends[0].kind, LookupKind::Memory);
        assert_eq!(receives[0].kind, LookupKind::Byte);

        // x=1, y=2, z=3: first send multiplicity is the constant 3, evaluated against a
        // concrete row rather than compared symbolically (SymbolicExpression has no PartialEq).
        let row = [KoalaBear::from_canonical_u32(1), KoalaBear::from_canonical_u32(2), KoalaBear::from_canonical_u32(3)];
        let multiplicity = sends[0].multiplicity.apply::<KoalaBear, KoalaBear>(&[], &row);
        assert_eq!(multiplicity, KoalaBear::from_canonical_u32(3));

        // second send's values are [x+y, z] = [3, 3].
        let values: Vec<KoalaBear> =
            sends[1].values.iter().map(|v| v.apply::<KoalaBear, KoalaBear>(&[], &row)).collect();
        assert_eq!(values, vec![KoalaBear::from_canonical_u32(3), KoalaBear::from_canonical_u32(3)]);

        let _ = main.row_mut(0);
    }
}
