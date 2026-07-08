use serde::{Deserialize, Serialize};
use slop_challenger::{CanObserve, IopCtx, VariableLengthChallenger};

use crate::septic_digest::SepticDigest;

/// A verifying key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineVerifyingKey<C: IopCtx> {
    /// The start pc of the program.
    pub pc_start: C::F,
    /// The starting global digest of the program, after incorporating the initial memory.
    pub initial_global_cumulative_sum: SepticDigest<C::F>,
    /// The preprocessed commitments.
    pub preprocessed_commit: C::Digest,
}

impl<C: IopCtx> PartialEq for MachineVerifyingKey<C> {
    fn eq(&self, other: &Self) -> bool {
        self.pc_start == other.pc_start
            && self.initial_global_cumulative_sum == other.initial_global_cumulative_sum
            && self.preprocessed_commit == other.preprocessed_commit
    }
}

impl<C: IopCtx> Eq for MachineVerifyingKey<C> {}

impl<C: IopCtx> MachineVerifyingKey<C> {
    /// Observes the values of the proving key into the challenger.
    pub fn observe_into(&self, challenger: &mut C::Challenger) {
        challenger.observe(self.preprocessed_commit);
        challenger.observe(self.pc_start);
        challenger.observe_constant_length_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_constant_length_slice(&self.initial_global_cumulative_sum.0.y.0);
    }
}
