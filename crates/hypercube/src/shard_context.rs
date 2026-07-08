use slop_challenger::IopCtx;
use slop_multilinear::MultilinearPcsVerifier;

use crate::{config::ZkmGlobalContext, ZerocheckAir};

/// A shortcut trait to package a multilinear PCS verifier and a zerocheck AIR. Reduces the number
/// of generic parameters in the `MachineVerifier` type and `AirProver` trait.
pub trait ShardContext<GC: IopCtx>: 'static + Send + Sync {
    /// The multilinear PCS verifier.
    type Config: MultilinearPcsVerifier<GC>;
    /// The AIR for which we'll be proving zerocheck.
    type Air: ZerocheckAir<GC::F, GC::EF>;
}

/// The canonical type implementing `ShardContext`.
pub struct ShardContextImpl<GC: IopCtx, Verifier, A>
where
    Verifier: MultilinearPcsVerifier<GC>,
    A: ZerocheckAir<GC::F, GC::EF>,
{
    _marker: std::marker::PhantomData<(GC, Verifier, A)>,
}

impl<GC: IopCtx, Verifier, A> ShardContext<GC> for ShardContextImpl<GC, Verifier, A>
where
    Verifier: MultilinearPcsVerifier<GC>,
    A: ZerocheckAir<GC::F, GC::EF>,
{
    type Config = Verifier;
    type Air = A;
}

/// A type alias assuming `ZkmPcs` (stacked Basefold) as the PCS verifier, generic in the AIR.
/// Used for all stages of Ziren proving except the outer (wrap) stage, which needs a
/// `ShardContext` over a BN254-bridged `IopCtx` not yet defined here.
pub type ZkmSC<A> = ShardContextImpl<ZkmGlobalContext, crate::config::ZkmStackedPcs, A>;
