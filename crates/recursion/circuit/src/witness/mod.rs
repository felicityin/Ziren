use p3_bn254_fr::Bn254Fr;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};

pub use zkm_recursion_compiler::ir::Witness as OuterWitness;
use zkm_recursion_compiler::{
    config::OuterConfig,
    ir::{Builder, Config, Ext, Felt, Var},
};
use zkm_recursion_core::air::Block;

use zkm_stark::{InnerChallenge, InnerVal};

use crate::CircuitConfig;

pub trait WitnessWriter<C: CircuitConfig>: Sized {
    fn write_bit(&mut self, value: bool);

    fn write_var(&mut self, value: C::N);

    fn write_felt(&mut self, value: C::F);

    fn write_ext(&mut self, value: C::EF);
}

/// TODO change the name. For now, the name is unique to prevent confusion.
pub trait Witnessable<C: CircuitConfig> {
    type WitnessVariable;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable;

    fn write(&self, witness: &mut impl WitnessWriter<C>);
}

impl<C: CircuitConfig> Witnessable<C> for bool {
    type WitnessVariable = C::Bit;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_bit(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_bit(*self);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for &T {
    type WitnessVariable = T::WitnessVariable;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        (*self).read(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        (*self).write(witness)
    }
}

impl<C: CircuitConfig, T: Witnessable<C>, U: Witnessable<C>> Witnessable<C> for (T, U) {
    type WitnessVariable = (T::WitnessVariable, U::WitnessVariable);

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        (self.0.read(builder), self.1.read(builder))
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.0.write(witness);
        self.1.write(witness);
    }
}

impl<C: CircuitConfig<F = InnerVal>> Witnessable<C> for InnerVal {
    type WitnessVariable = Felt<InnerVal>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_felt(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_felt(*self);
    }
}

impl<C: CircuitConfig<F = InnerVal, EF = InnerChallenge>> Witnessable<C> for InnerChallenge {
    type WitnessVariable = Ext<InnerVal, InnerChallenge>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        C::read_ext(builder)
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        // vec![Block::from(self.as_base_slice())]
        witness.write_ext(*self);
    }
}

impl<C: CircuitConfig, T: Witnessable<C>, const N: usize> Witnessable<C> for [T; N] {
    type WitnessVariable = [T::WitnessVariable; N];

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect::<Vec<_>>().try_into().unwrap_or_else(
            |x: Vec<_>| {
                // Cannot just `.unwrap()` without requiring Debug bounds.
                panic!("could not coerce vec of len {} into array of len {N}", x.len())
            },
        )
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Vec<T> {
    type WitnessVariable = Vec<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect()
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for slop_commit::Rounds<T> {
    type WitnessVariable = slop_commit::Rounds<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|x| x.read(builder)).collect()
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for x in self.iter() {
            x.write(witness);
        }
    }
}

impl<C: CircuitConfig, K: Clone + Ord, V: Witnessable<C>> Witnessable<C>
    for std::collections::BTreeMap<K, V>
{
    type WitnessVariable = std::collections::BTreeMap<K, V::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        self.iter().map(|(k, v)| (k.clone(), v.read(builder))).collect()
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        for v in self.values() {
            v.write(witness);
        }
    }
}

/// A witness block used by the `InnerConfig`, backed by plain field-element storage rather than
/// the `Var`/`Felt`/`Ext` variables Bn254-targeting Gnark/Plonk circuits need.
pub type WitnessBlock<C> = Block<<C as Config>::F>;

impl<C: CircuitConfig<F = p3_koala_bear::KoalaBear, Bit = Felt<p3_koala_bear::KoalaBear>>>
    WitnessWriter<C> for Vec<WitnessBlock<C>>
{
    fn write_bit(&mut self, value: bool) {
        self.push(Block::from(C::F::from_bool(value)))
    }

    fn write_var(&mut self, _value: <C>::N) {
        unimplemented!("Cannot write Var<N> in this configuration")
    }

    fn write_felt(&mut self, value: <C>::F) {
        self.push(Block::from(value))
    }

    fn write_ext(&mut self, value: <C>::EF) {
        self.push(Block::from(value.as_base_slice()))
    }
}

impl WitnessWriter<OuterConfig> for OuterWitness<OuterConfig> {
    fn write_bit(&mut self, value: bool) {
        self.vars.push(Bn254Fr::from_bool(value));
    }

    fn write_var(&mut self, value: Bn254Fr) {
        self.vars.push(value);
    }

    fn write_felt(&mut self, value: <OuterConfig as Config>::F) {
        self.felts.push(value);
    }

    fn write_ext(&mut self, value: <OuterConfig as Config>::EF) {
        self.exts.push(value);
    }
}

impl<C: CircuitConfig<N = Bn254Fr>> Witnessable<C> for Bn254Fr {
    type WitnessVariable = Var<Bn254Fr>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        builder.witness_var()
    }
    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        witness.write_var(*self)
    }
}
