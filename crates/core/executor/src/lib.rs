mod air;
mod consts;
mod context;
mod cost;
mod errors;
pub mod events;
mod executor;
#[cfg(test)]
mod golden;
pub mod hook;
mod instruction;
mod io;
pub mod memory;
mod minimal;
mod opcode;
mod pipeline;
mod program;
#[cfg(test)]
pub mod programs;
mod record;
mod register;
pub mod report;
mod splicing;
mod state;
pub mod subproof;
pub mod syscalls;
mod trace;
mod tracing;
mod utils;
mod vm;

pub use air::*;
pub use consts::*;
pub use context::*;
pub use cost::*;
pub use errors::*;
pub use executor::*;
pub use hook::*;
pub use instruction::*;
pub use opcode::*;
pub use pipeline::*;
pub use program::*;
pub use record::*;
pub use register::*;
pub use report::*;
pub use state::*;
pub use subproof::*;
pub use utils::*;
pub use zkm_hypercube::ZKMReduceProof;

#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum OptionValTag {
    Some = 0,
    None,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct OptionU32 {
    pub tag: OptionValTag,
    pub value: u32,
}

impl From<Option<u32>> for OptionU32 {
    fn from(val: Option<u32>) -> Self {
        match val {
            Some(value) => OptionU32 { tag: OptionValTag::Some, value },
            None => OptionU32 { tag: OptionValTag::None, value: 0 },
        }
    }
}
