use std::fmt::Display;

use p3_field::PrimeField64;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zkm_derive::AlignedBorrow;

use crate::air::{Block, RecursionPublicValues};

pub mod air;
pub mod builder;
pub mod chips;
pub mod machine;
pub mod runtime;
pub mod shape;
pub mod stark;
#[cfg(feature = "sys")]
pub mod sys;

pub use runtime::*;
pub use stark::hash_vkey_with_part_vk;

// Re-export the stark stuff from `zkm_recursion_core` for now, until we will migrate it here.
// pub use zkm_recursion_core::stark;

use crate::chips::poseidon2_skinny::WIDTH;

#[derive(Error, Debug, Serialize, Deserialize)]
pub struct RecursionChipError;

impl Display for RecursionChipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecursionChipError")
    }
}

#[derive(
    AlignedBorrow, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default,
)]
#[repr(transparent)]
pub struct Address<F>(pub F);

impl<F: PrimeField64> Address<F> {
    #[inline]
    pub fn as_usize(&self) -> usize {
        self.0.as_canonical_u64() as usize
    }
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to an operation of the base field ALU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct BaseAluIo<V> {
    pub out: V,
    pub in1: V,
    pub in2: V,
}

pub type BaseAluEvent<F> = BaseAluIo<F>;

/// An instruction invoking the base field ALU.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct BaseAluInstr<F> {
    pub opcode: BaseAluOpcode,
    pub mult: F,
    pub addrs: BaseAluIo<Address<F>>,
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to an operation of the extension field ALU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtAluIo<V> {
    pub out: V,
    pub in1: V,
    pub in2: V,
}

pub type ExtAluEvent<F> = ExtAluIo<Block<F>>;

/// An instruction invoking the extension field ALU.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtAluInstr<F> {
    pub opcode: ExtAluOpcode,
    pub mult: F,
    pub addrs: ExtAluIo<Address<F>>,
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to the manual memory management/memory initialization table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemIo<V> {
    pub inner: V,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemInstr<F> {
    pub addrs: MemIo<Address<F>>,
    pub vals: MemIo<Block<F>>,
    pub mult: F,
    pub kind: MemAccessKind,
}

pub type MemEvent<F> = MemIo<Block<F>>;

// -------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemAccessKind {
    Read,
    Write,
}

/// The inputs and outputs to a Poseidon2 permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2Io<V> {
    pub input: [V; WIDTH],
    pub output: [V; WIDTH],
}

/// An instruction invoking the Poseidon2 permutation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2SkinnyInstr<F> {
    pub addrs: Poseidon2Io<Address<F>>,
    pub mults: [F; WIDTH],
}

pub type Poseidon2Event<F> = Poseidon2Io<F>;

/// The inputs and outputs to a select operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct SelectIo<V> {
    pub bit: V,
    pub out1: V,
    pub out2: V,
    pub in1: V,
    pub in2: V,
}

/// An instruction invoking the select operation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct SelectInstr<F> {
    pub addrs: SelectIo<Address<F>>,
    pub mult1: F,
    pub mult2: F,
}

/// The event encoding the inputs and outputs of a select operation.
pub type SelectEvent<F> = SelectIo<F>;

pub type Poseidon2WideEvent<F> = Poseidon2Io<F>;
pub type Poseidon2Instr<F> = Poseidon2SkinnyInstr<F>;

/// The inputs and outputs to one linear-layer round over the permutation state, packed as
/// `WIDTH / D` extension-sized blocks (each `V` holds `D` base-field elements).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2LinearLayerIo<V> {
    pub input: [V; WIDTH / D],
    pub output: [V; WIDTH / D],
}

/// An instruction invoking one external or internal linear-layer round, row-local (degree <= 3),
/// unlike `Poseidon2SkinnyInstr` which invokes a full permutation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2LinearLayerInstr<F> {
    pub addrs: Poseidon2LinearLayerIo<Address<F>>,
    pub mults: [F; WIDTH / D],
    pub external: bool,
}

pub type Poseidon2LinearLayerEvent<F> = Poseidon2LinearLayerIo<Block<F>>;

/// The input and output to one Poseidon2 S-box application over a single extension-sized block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2SBoxIo<V> {
    pub input: V,
    pub output: V,
}

/// An instruction invoking one external or internal S-box application, row-local (degree <= 3).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2SBoxInstr<F> {
    pub addrs: Poseidon2SBoxIo<Address<F>>,
    pub mult: F,
    pub external: bool,
}

pub type Poseidon2SBoxEvent<F> = Poseidon2SBoxIo<Block<F>>;

/// The inputs and outputs to the operations for prefix sum checks. This struct doubles as both
/// the DSL/instruction-level "one entry per accumulation step" addresses (`Vec<Address<F>>`) and
/// the executed-value shape (`Vec<V>` for the runtime), matching SP1's own `PrefixSumChecksIo`
/// shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixSumChecksIo<V> {
    pub zero: V,
    pub one: V,
    pub x1: Vec<V>,
    pub x2: Vec<V>,
    pub accs: Vec<V>,
    pub field_accs: Vec<V>,
}

/// An instruction invoking the PrefixSumChecks operation. Every intermediate accumulator gets
/// its own address (`addrs.accs[i]`/`addrs.field_accs[i]`), which is what lets the AIR chip stay
/// fully row-local (each row reads its predecessor's output straight from memory) instead of
/// needing a physical `next`-row reference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixSumChecksInstr<F> {
    pub addrs: PrefixSumChecksIo<Address<F>>,
    pub acc_mults: Vec<F>,
    pub field_acc_mults: Vec<F>,
}

/// The event encoding the data of a single step within the prefix-sum-checks operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct PrefixSumChecksEvent<F> {
    pub x1: F,
    pub x2: Block<F>,
    pub zero: F,
    pub one: Block<F>,
    pub acc: Block<F>,
    pub new_acc: Block<F>,
    pub field_acc: F,
    pub new_field_acc: F,
}

/// An instruction that will save the public values to the execution record and will commit to
/// it's digest.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct CommitPublicValuesInstr<F> {
    pub pv_addrs: RecursionPublicValues<Address<F>>,
}

/// The event for committing to the public values.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct CommitPublicValuesEvent<F> {
    pub public_values: RecursionPublicValues<F>,
}
