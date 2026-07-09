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
        }
    }
}
