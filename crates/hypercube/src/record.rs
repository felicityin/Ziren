use hashbrown::HashMap;

use slop_algebra::AbstractField;

use crate::{air::ZKMAirBuilder, lookup::LookupKind};

/// A record that can be proven by a machine.
pub trait MachineRecord: Default + Sized + Send + Sync + Clone {
    /// The statistics of the record.
    fn stats(&self) -> HashMap<String, usize>;

    /// Appends two records together.
    fn append(&mut self, other: &mut Self);

    /// Returns the public values of the record.
    fn public_values<F: AbstractField>(&self) -> Vec<F>;

    /// Constrains the public values of the record.
    fn eval_public_values<AB: ZKMAirBuilder>(builder: &mut AB);

    /// The lookup kinds that appear in `eval_public_values`. Needed so that the shard verifier
    /// knows how much randomness to allocate for the `LogUpGkr` `beta_seed` challenge.
    fn lookups_in_public_values() -> Vec<LookupKind>;
}
