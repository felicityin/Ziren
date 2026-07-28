use core::fmt::Display;

use serde::{Deserialize, Serialize};

mod builder;
pub mod debug;
#[allow(clippy::module_inception)]
mod lookup;

pub use builder::*;
pub use debug::*;
pub use lookup::*;

/// The type of a lookup argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LookupKind {
    Memory = 1,
    Program = 2,
    Instruction = 3,
    Byte = 4,
    Range = 5,
    Syscall = 6,
    Global = 7,
    SyscallResult = 8,
    State = 9,
    GlobalAccumulation = 10,
    MemoryGlobalInitControl = 11,
    MemoryGlobalFinalizeControl = 12,
    ShaExtend = 13,
    ShaCompress = 14,
    KeccakPermuteRound = 15,
    KeccakSpongeBlock = 16,
    Poseidon2SkinnyState = 20,
}

impl LookupKind {
    #[must_use]
    pub fn all_kinds() -> Vec<LookupKind> {
        vec![
            LookupKind::Memory,
            LookupKind::Program,
            LookupKind::Instruction,
            LookupKind::Byte,
            LookupKind::Range,
            LookupKind::Syscall,
            LookupKind::Global,
            LookupKind::SyscallResult,
            LookupKind::State,
            LookupKind::GlobalAccumulation,
            LookupKind::MemoryGlobalInitControl,
            LookupKind::MemoryGlobalFinalizeControl,
            LookupKind::ShaExtend,
            LookupKind::ShaCompress,
            LookupKind::KeccakPermuteRound,
            LookupKind::KeccakSpongeBlock,
            LookupKind::Poseidon2SkinnyState,
        ]
    }

    /// Whether this interaction kind appears in the core MIPS machine's own `eval_public_values`
    /// (`crates/core/executor/src/record.rs`). Used to build `interactions_in_public_values()`.
    #[must_use]
    pub fn appears_in_eval_public_values(&self) -> bool {
        matches!(
            self,
            LookupKind::State
                | LookupKind::MemoryGlobalInitControl
                | LookupKind::MemoryGlobalFinalizeControl
                | LookupKind::GlobalAccumulation
        )
    }

    /// The number of raw `values` sent/received for each interaction kind in the core MIPS
    /// machine (not counting the `LookupKind` discriminant itself, added separately by callers).
    /// The recursion machine reuses several of these same-named kinds with *different* arities
    /// (e.g. `Memory`=6, `Range`=2, variable-arity `Syscall`) -- safe to share this table only
    /// because the recursion machine's `interactions_in_public_values()` always returns `vec![]`
    /// (its `eval_public_values` is a no-op), so this method is never invoked for a
    /// recursion-specific kind in practice.
    #[must_use]
    pub fn num_values(&self) -> usize {
        match self {
            LookupKind::Memory => 7,
            LookupKind::Program => 14,
            LookupKind::Instruction => 28,
            LookupKind::Byte | LookupKind::Syscall => 5,
            LookupKind::Global => 10,
            LookupKind::SyscallResult => 8,
            LookupKind::State | LookupKind::ShaExtend => 4,
            LookupKind::GlobalAccumulation => 15,
            LookupKind::MemoryGlobalInitControl | LookupKind::MemoryGlobalFinalizeControl => 6,
            LookupKind::ShaCompress => 37,
            LookupKind::KeccakPermuteRound | LookupKind::KeccakSpongeBlock => 106,
            // Not used anywhere in the core MIPS machine.
            LookupKind::Range | LookupKind::Poseidon2SkinnyState => 0,
        }
    }
}

impl Display for LookupKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LookupKind::Memory => write!(f, "Memory"),
            LookupKind::Program => write!(f, "Program"),
            LookupKind::Instruction => write!(f, "Instruction"),
            LookupKind::Byte => write!(f, "Byte"),
            LookupKind::Range => write!(f, "Range"),
            LookupKind::Syscall => write!(f, "Syscall"),
            LookupKind::Global => write!(f, "Global"),
            LookupKind::SyscallResult => write!(f, "SyscallResult"),
            LookupKind::State => write!(f, "State"),
            LookupKind::GlobalAccumulation => write!(f, "GlobalAccumulation"),
            LookupKind::MemoryGlobalInitControl => write!(f, "MemoryGlobalInitControl"),
            LookupKind::MemoryGlobalFinalizeControl => write!(f, "MemoryGlobalFinalizeControl"),
            LookupKind::ShaExtend => write!(f, "ShaExtend"),
            LookupKind::ShaCompress => write!(f, "ShaCompress"),
            LookupKind::KeccakPermuteRound => write!(f, "KeccakPermuteRound"),
            LookupKind::KeccakSpongeBlock => write!(f, "KeccakSpongeBlock"),
            LookupKind::Poseidon2SkinnyState => write!(f, "Poseidon2SkinnyState"),
        }
    }
}
