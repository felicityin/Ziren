use serde::{Deserialize, Serialize};

use crate::events::{
    memory::{MemoryReadRecord, MemoryWriteRecord},
    MemoryLocalEvent,
};

/// Elliptic Curve Add Event.
///
/// This event is emitted when an elliptic curve addition operation is performed.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct EllipticCurveAddEvent {
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u64,
    /// The pointer to the first point.
    pub p_ptr: u32,
    /// The first point as a list of words.
    pub p: Vec<u32>,
    /// The pointer to the second point.
    pub q_ptr: u32,
    /// The second point as a list of words.
    pub q: Vec<u32>,
    /// The memory records for the first point.
    pub p_memory_records: Vec<MemoryWriteRecord>,
    /// The memory records for the second point.
    pub q_memory_records: Vec<MemoryReadRecord>,
    /// The local memory access records.
    pub local_mem_access: Vec<MemoryLocalEvent>,
}

/// Elliptic Curve Double Event.
///
/// This event is emitted when an elliptic curve doubling operation is performed.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct EllipticCurveDoubleEvent {
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u64,
    /// The pointer to the point.
    pub p_ptr: u32,
    /// The point as a list of words.
    pub p: Vec<u32>,
    /// The memory records for the point.
    pub p_memory_records: Vec<MemoryWriteRecord>,
    /// The local memory access records.
    pub local_mem_access: Vec<MemoryLocalEvent>,
}

/// Elliptic Curve Point Decompress Event.
///
/// This event is emitted when an elliptic curve point decompression operation is performed.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct EllipticCurveDecompressEvent {
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u64,
    /// The pointer to the point.
    pub ptr: u32,
    /// The sign bit of the point.
    pub sign_bit: bool,
    /// The x coordinate as a list of bytes.
    pub x_bytes: Vec<u8>,
    /// The decompressed y coordinate as a list of bytes.
    pub decompressed_y_bytes: Vec<u8>,
    /// The memory records for the x coordinate.
    pub x_memory_records: Vec<MemoryReadRecord>,
    /// The memory records for the y coordinate.
    pub y_memory_records: Vec<MemoryWriteRecord>,
    /// The local memory access records.
    pub local_mem_access: Vec<MemoryLocalEvent>,
}
