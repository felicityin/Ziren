use hashbrown::HashMap;

use p3_field::FieldAlgebra;

use crate::air::ZKMAirBuilder;

/// A record that can be proven by a machine.
pub trait MachineRecord: Default + Sized + Send + Sync + Clone {
    /// The statistics of the record.
    fn stats(&self) -> HashMap<String, usize>;

    /// Appends two records together.
    fn append(&mut self, other: &mut Self);

    /// Returns the public values of the record.
    fn public_values<F: FieldAlgebra>(&self) -> Vec<F>;

    /// Constrains the public values of the record.
    fn eval_public_values<AB: ZKMAirBuilder>(builder: &mut AB);

    /// The widest interaction (`values.len() + 1`, to also cover the `LookupKind` value itself)
    /// that `eval_public_values` sends or receives. Needed so that the shard verifier knows how
    /// much randomness to allocate for the `LogUpGkr` `beta_seed` challenge -- `eval_public_values`
    /// runs unconditionally for every shard regardless of which chips that shard's cluster
    /// includes, so this must be accounted for on top of (not instead of) the chosen cluster's own
    /// widest chip interaction.
    fn max_public_values_interaction_arity() -> usize;
}
