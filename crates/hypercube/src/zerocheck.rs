use std::fmt::Debug;

use slop_air::Air;
use slop_algebra::{ExtensionField, Field};
use slop_uni_stark::SymbolicAirBuilder;

use crate::{air::MachineAir, ConstraintSumcheckFolder, DebugConstraintBuilder, VerifierConstraintFolder};

/// An AIR compatible with the zerocheck prover: it must be evaluable against every folder type
/// the zerocheck sumcheck, its verifier, and the constraint debugger use.
pub trait ZerocheckAir<F: Field, EF: ExtensionField<F>>:
    Debug
    + MachineAir<F>
    + Air<SymbolicAirBuilder<F>>
    + for<'b> Air<ConstraintSumcheckFolder<'b, F, F, EF>>
    + for<'b> Air<ConstraintSumcheckFolder<'b, F, EF, EF>>
    + for<'b> Air<DebugConstraintBuilder<'b, F, EF>>
    + for<'a> Air<VerifierConstraintFolder<'a, F, EF>>
{
}

impl<F: Field, EF: ExtensionField<F>, A> ZerocheckAir<F, EF> for A where
    A: MachineAir<F>
        + Debug
        + Air<SymbolicAirBuilder<F>>
        + for<'b> Air<ConstraintSumcheckFolder<'b, F, F, EF>>
        + for<'b> Air<ConstraintSumcheckFolder<'b, F, EF, EF>>
        + for<'b> Air<DebugConstraintBuilder<'b, F, EF>>
        + for<'a> Air<VerifierConstraintFolder<'a, F, EF>>
{
}
