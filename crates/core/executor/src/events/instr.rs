use super::MemoryRecordEnum;
use super::MemoryWriteRecord;
use crate::Opcode;
use serde::{Deserialize, Serialize};

/// Emitted whenever the executor pre-emptively advances `clk` to the start of the next `1 << 24`
/// window because the current window doesn't have room for the widest `MemoryAccessPosition`
/// offset (see `Executor::bump_clk_high_if_need`'s doc comment). Consumed by the `StateBumpChip`
/// AIR that proves the resulting `clk_high` transition in-circuit: this event's `pc`/`next_pc`
/// are exactly what the *previous* instruction sent as its own outgoing
/// `(outgoing_next_pc, outgoing_next_next_pc)`, so `StateBumpChip` can receive that state and
/// pass `pc`/`next_pc` through unchanged to whatever instruction executes next.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct BumpClkHighEvent {
    /// The `clk` value before this jump.
    pub prev_clk: u64,
    /// The size of this jump (brings `clk` to the next `1 << 24` boundary).
    pub increment: u64,
    /// The pc of the instruction about to execute (matches the previous instruction's own
    /// outgoing `next_pc`).
    pub pc: u32,
    /// The `next_pc` the instruction about to execute will itself receive as incoming state
    /// (matches the previous instruction's own outgoing `next_next_pc`).
    pub next_pc: u32,
}

/// Arithmetic Logic Unit (ALU) Event.
///
/// This object encapsulated the information needed to prove an ALU operation. This includes its
/// shard, opcode, operands, and other relevant information.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct AluEvent {
    /// The clock cycle.
    pub clk: u64,
    pub pc: u32,
    pub next_pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The upper bits of the output operand.
    /// This is used for the MULT, MULTU, DIV and DIVU opcodes.
    pub hi: u32,
    /// The output operand.
    pub a: u32,
    /// The first input operand.
    pub b: u32,
    /// The second input operand.
    pub c: u32,
    /// The memory access record for the output operand `a`.
    pub a_record: Option<MemoryRecordEnum>,
    /// The memory access record for the first input operand `b`, if it's a register (not an
    /// immediate).
    pub b_record: Option<MemoryRecordEnum>,
    /// The memory access record for the second input operand `c`, if it's a register (not an
    /// immediate).
    pub c_record: Option<MemoryRecordEnum>,
}

impl AluEvent {
    /// Create a new [`AluEvent`].
    #[must_use]
    pub fn new(pc: u32, opcode: Opcode, a: u32, b: u32, c: u32) -> Self {
        Self {
            clk: 0,
            pc,
            next_pc: pc + 4,
            opcode,
            a,
            b,
            c,
            hi: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }

    /// Create a new [`AluEvent`].
    /// Used for opcode with LO and HI registers
    /// DIV DIVU MULT MULLTU
    #[must_use]
    pub fn new_with_hi(pc: u32, opcode: Opcode, a: u32, b: u32, c: u32, hi: u32) -> Self {
        Self {
            clk: 0,
            pc,
            next_pc: pc + 4,
            opcode,
            a,
            b,
            c,
            hi,
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }
}

/// Complicated Arithmetic Logic Unit (ALU) Event.
///
/// This object encapsulated the information needed to prove an ALU operation. This includes its
/// shard, opcode, operands, and other relevant information.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct CompAluEvent {
    /// The clock cycle.
    pub clk: u64,

    pub pc: u32,
    pub next_pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The upper bits of the output operand.
    /// This is used for the MULT, MULTU, DIV and DIVU opcodes.
    pub hi: u32,
    /// The output operand.
    pub a: u32,
    /// The first input operand.
    pub b: u32,
    /// The second input operand.
    pub c: u32,

    /// The `op_hi` memory write record.
    pub hi_record: MemoryWriteRecord,
    pub hi_record_is_real: bool,

    /// The memory access record for the output operand `a`.
    pub a_record: Option<MemoryRecordEnum>,
    /// The memory access record for the first input operand `b`, if it's a register (not an
    /// immediate).
    pub b_record: Option<MemoryRecordEnum>,
    /// The memory access record for the second input operand `c`, if it's a register (not an
    /// immediate).
    pub c_record: Option<MemoryRecordEnum>,
}

impl CompAluEvent {
    /// Create a new [`CompAluEvent`].
    #[must_use]
    pub fn new(pc: u32, opcode: Opcode, a: u32, b: u32, c: u32) -> Self {
        Self {
            clk: 0,
            pc,
            next_pc: pc + 4,
            opcode,
            hi: 0,
            a,
            b,
            c,
            hi_record_is_real: false,
            hi_record: MemoryWriteRecord::default(),
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }

    pub fn new_with_hi(pc: u32, opcode: Opcode, a: u32, b: u32, c: u32, hi: u32) -> Self {
        Self {
            clk: 0,
            pc,
            next_pc: pc + 4,
            opcode,
            hi,
            a,
            b,
            c,
            hi_record_is_real: false,
            hi_record: MemoryWriteRecord::default(),
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }
}

/// Memory Instruction Event.
///
/// This object encapsulated the information needed to prove a MIPS memory operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct MemInstrEvent {
    /// The clk.
    pub clk: u64,
    /// The program counter.
    pub pc: u32,
    pub next_pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// The memory access record for memory operations.
    pub mem_access: MemoryRecordEnum,
    /// The memory access record for memory operations.
    pub prev_a_val: u32,
    /// The memory access record for the output operand `a`.
    pub a_record: Option<MemoryRecordEnum>,
    /// The memory access record for the first input operand `b`, if it's a register (not an
    /// immediate).
    pub b_record: Option<MemoryRecordEnum>,
    /// The memory access record for the second input operand `c`, if it's a register (not an
    /// immediate).
    pub c_record: Option<MemoryRecordEnum>,
}

impl MemInstrEvent {
    /// Create a new [`MemInstrEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        mem_access: MemoryRecordEnum,
        prev_a_val: u32,
    ) -> Self {
        Self {
            clk,
            pc,
            next_pc,
            opcode,
            a,
            b,
            c,
            mem_access,
            prev_a_val,
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }
}

/// Branch Instruction Event.
///
/// This object encapsulated the information needed to prove a MIPS branch operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct BranchEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,
    /// The next program counter.
    pub next_next_pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// The memory access record for the output operand `a`.
    pub a_record: Option<MemoryRecordEnum>,
    /// The memory access record for the first input operand `b`, if it's a register (not an
    /// immediate).
    pub b_record: Option<MemoryRecordEnum>,
    /// The memory access record for the second input operand `c`, if it's a register (not an
    /// immediate).
    pub c_record: Option<MemoryRecordEnum>,
}

impl BranchEvent {
    /// Create a new [`BranchEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc: u32,
        next_pc: u32,
        next_next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
    ) -> Self {
        Self {
            clk,
            pc,
            next_pc,
            next_next_pc,
            opcode,
            a,
            b,
            c,
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }
}

/// Jump Instruction Event.
///
/// This object encapsulated the information needed to prove a MIPS jump operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct JumpEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,
    /// The next next program counter.
    pub next_next_pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// The memory access record for the output operand `a`.
    pub a_record: Option<MemoryRecordEnum>,
    /// The memory access record for the first input operand `b`, if it's a register (not an
    /// immediate).
    pub b_record: Option<MemoryRecordEnum>,
    /// The memory access record for the second input operand `c`, if it's a register (not an
    /// immediate).
    pub c_record: Option<MemoryRecordEnum>,
}

impl JumpEvent {
    /// Create a new [`JumpEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc: u32,
        next_pc: u32,
        next_next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
    ) -> Self {
        Self {
            clk,
            pc,
            next_pc,
            next_next_pc,
            opcode,
            a,
            b,
            c,
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }
}

/// Misc Instruction Event.
///
/// This object encapsulated the information needed to prove a MIPS misc operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct MiscEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc: u32,
    pub next_pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// The third operand value.
    pub prev_a: u32,
    /// The hi operand memory record.
    pub hi_record: MemoryWriteRecord,
    /// The memory access record for the output operand `a`.
    pub a_record: Option<MemoryRecordEnum>,
    /// The memory access record for the first input operand `b`, if it's a register (not an
    /// immediate).
    pub b_record: Option<MemoryRecordEnum>,
    /// The memory access record for the second input operand `c`, if it's a register (not an
    /// immediate).
    pub c_record: Option<MemoryRecordEnum>,
}

impl MiscEvent {
    /// Create a new [`MiscEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        prev_a: u32,
        hi_record: MemoryWriteRecord,
    ) -> Self {
        Self {
            clk,
            pc,
            next_pc,
            opcode,
            a,
            b,
            c,
            prev_a,
            hi_record,
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }
}

/// Misc Instruction Event.
///
/// This object encapsulated the information needed to prove a MIPS MovCond and WSBH operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct MovCondEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The program counter.
    pub pc: u32,
    pub next_pc: u32,
    /// The opcode.
    pub opcode: Opcode,
    /// The first operand value.
    pub a: u32,
    /// The second operand value.
    pub b: u32,
    /// The third operand value.
    pub c: u32,
    /// The third operand value.
    pub prev_a: u32,
    /// The memory access record for the output operand `a`.
    pub a_record: Option<MemoryRecordEnum>,
    /// The memory access record for the first input operand `b`, if it's a register (not an
    /// immediate).
    pub b_record: Option<MemoryRecordEnum>,
    /// The memory access record for the second input operand `c`, if it's a register (not an
    /// immediate).
    pub c_record: Option<MemoryRecordEnum>,
}

impl MovCondEvent {
    /// Create a new [`MovCondEvent`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        clk: u64,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        prev_a: u32,
    ) -> Self {
        Self {
            clk,
            pc,
            next_pc,
            opcode,
            a,
            b,
            c,
            prev_a,
            a_record: None,
            b_record: None,
            c_record: None,
        }
    }
}
