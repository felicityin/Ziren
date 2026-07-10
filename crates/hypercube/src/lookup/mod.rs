use core::fmt::Display;

use serde::{Deserialize, Serialize};

mod builder;
#[allow(clippy::module_inception)]
mod lookup;

pub use builder::*;
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
    BatchFRIAccumulation = 17,
    FriFoldConstant = 18,
    ExpReverseBitsChain = 19,
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
            LookupKind::BatchFRIAccumulation,
            LookupKind::FriFoldConstant,
            LookupKind::ExpReverseBitsChain,
            LookupKind::Poseidon2SkinnyState,
        ]
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
            LookupKind::BatchFRIAccumulation => write!(f, "BatchFRIAccumulation"),
            LookupKind::FriFoldConstant => write!(f, "FriFoldConstant"),
            LookupKind::ExpReverseBitsChain => write!(f, "ExpReverseBitsChain"),
            LookupKind::Poseidon2SkinnyState => write!(f, "Poseidon2SkinnyState"),
        }
    }
}
