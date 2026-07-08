use slop_challenger::IopCtx;

use crate::{air::MachineAir, ShardContext};

/// The concrete record type produced by a shard context's AIR.
pub type Record<GC, SC> = <<SC as ShardContext<GC>>::Air as MachineAir<<GC as IopCtx>::F>>::Record;
