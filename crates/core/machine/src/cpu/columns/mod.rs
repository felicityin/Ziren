mod instruction;
pub use instruction::*;

use p3_util::indices_arr;
use std::mem::{size_of, transmute};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
use zkm_hypercube::word::Word;

use crate::memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols};

pub const NUM_CPU_COLS: usize = size_of::<CpuCols<u8>>();

pub const CPU_COL_MAP: CpuCols<usize> = make_col_map();

/// The column layout for the CPU.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct CpuCols<T: Copy> {
    /// The current shard.
    pub shard: T,

    /// The least significant 16 bit limb of clk.
    pub clk_16bit_limb: T,
    /// The most significant 8 bit limb of clk.
    pub clk_8bit_limb: T,

    /// The shard to send to the opcode specific tables.  This should be 0 for all instructions other   
    /// than the syscall and memory instructions.
    pub shard_to_send: T,
    /// The clk to send to the opcode specific tables.  This should be 0 for all instructions other
    /// than the syscall and memory instructions.
    pub clk_to_send: T,

    /// The program counter value.
    pub pc: T,

    /// The next program counter value.
    pub next_pc: T,

    /// The expected next_next program counter value.
    pub next_next_pc: T,

    /// This row's own incoming-chain pc, i.e. what `local.next_pc` would be if it weren't
    /// overwritten to the public halt sentinel `0` on a halting row (see
    /// [`crate::cpu::air::CpuChip::eval_state_chain`]). Equal to `local.next_pc` for every
    /// non-halting row and to `local.pc + 4` for a halting one; witnessed separately because
    /// `LookupKind::State`'s values must stay affine in the trace columns.
    pub state_chain_next_pc: T,

    /// Columns related to the instruction.
    pub instruction: InstructionCols<T>,

    /// The number of extra cycles to add to the clk for a syscall instruction.
    pub num_extra_cycles: T,

    /// Whether the instruction will read and write a register.
    pub is_rw_a: T,

    /// Whether the instruction chip will check memory access.
    pub is_check_memory: T,

    /// Whether this is a halt instruction.
    pub is_halt: T,

    /// Whether this is a sequential instruction (not branch or jump or halt).
    pub is_sequential: T,

    /// Operand values, either from registers or immediate values.
    pub op_a_value: Word<T>,
    pub hi_or_prev_a: Word<T>,
    pub op_a_access: MemoryReadWriteCols<T>,
    pub op_b_access: MemoryReadCols<T>,
    pub op_c_access: MemoryReadCols<T>,

    /// Selector to label whether this row is a non padded row.
    pub is_real: T,

    /// Whether op_a is immutable
    pub op_a_immutable: T,
}

impl<T: Copy> CpuCols<T> {
    /// Gets the value of the first operand.
    pub fn op_a_val(&self) -> Word<T> {
        *self.op_a_access.value()
    }

    /// Gets the value of the second operand.
    pub fn op_b_val(&self) -> Word<T> {
        *self.op_b_access.value()
    }

    /// Gets the value of the third operand.
    pub fn op_c_val(&self) -> Word<T> {
        *self.op_c_access.value()
    }
}

/// Creates the column map for the CPU.
const fn make_col_map() -> CpuCols<usize> {
    let indices_arr = indices_arr::<NUM_CPU_COLS>();
    unsafe { transmute::<[usize; NUM_CPU_COLS], CpuCols<usize>>(indices_arr) }
}
