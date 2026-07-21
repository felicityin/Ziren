use super::{MemoryRecordEnum, MemoryWriteRecord};
use serde::{Deserialize, Serialize};

/// Syscall Event.
///
/// This object encapsulated the information needed to prove a syscall invocation from the CPU table.
/// This includes its shard, clk, syscall id, arguments, other relevant information.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct SyscallEvent {
    /// The program counter.
    pub pc: u32,
    /// The next program counter.
    pub next_pc: u32,
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u64,
    /// The `op_a` memory write record.
    pub a_record: MemoryWriteRecord,
    /// Whether the `op_a` memory write record is real.
    pub a_record_is_real: bool,
    /// The `op_b` memory read record, when this event backs a real SYSCALL instruction row.
    pub b_record: Option<MemoryRecordEnum>,
    /// The `op_c` memory read record, when this event backs a real SYSCALL instruction row.
    pub c_record: Option<MemoryRecordEnum>,
    /// The syscall id.
    pub syscall_id: u32,
    /// The first argument.
    pub arg1: u32,
    /// The second operand.
    pub arg2: u32,
}
