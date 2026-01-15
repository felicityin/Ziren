use derive_new::new;
use serde::{Deserialize, Serialize};

use crate::{MAX_MEMORY, NUM_REGISTERS};

pub const MIPS_REGISTER_SPACE: u32 = 0;
pub const MIPS_MEMORY_SPACE: u32 = 1;

const DEFAULT_NATIVE_BLOCK_SIZE: usize = 1;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, new)]
pub struct AddressSpaceHostConfig {
    /// The number of memory cells in each address space, where a memory cell refers to a single
    /// addressable unit of memory as defined by the ISA.
    pub num_cells: usize,
    /// Minimum block size for memory accesses supported. This is a property of the address space
    /// that is determined by the ISA.
    ///
    /// **Note**: Block size is in terms of memory cells.
    pub min_block_size: usize,
    pub layout: MemoryCellType,
}

impl AddressSpaceHostConfig {
    /// The total size in bytes of the address space in a linear memory layout.
    pub fn size(&self) -> usize {
        self.num_cells * self.layout.size()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCellType {
    Null,
    U8,
    U16,
    /// Represented in little-endian format.
    U32,
    /// `size` is the size in bytes of the native field type. This should not exceed 8.
    Native {
        size: u8,
    },
}

impl MemoryCellType {
    pub fn native32() -> Self {
        Self::Native { size: size_of::<u32>() as u8 }
    }
}

/// Each address space in guest memory may be configured with a different type `T` to represent a
/// memory cell in the address space. On host, the address space will be mapped to linear host
/// memory in bytes. The type `T` must be plain old data (POD) and be safely transmutable from a
/// fixed size array of bytes. Moreover, each type `T` must be convertible to a field element `F`.
///
/// We currently implement this trait on the enum [MemoryCellType], which includes all cell types
/// that we expect to be used in the VM context.
pub trait AddressSpaceHostLayout {
    /// Size in bytes of the memory cell type.
    fn size(&self) -> usize;
}

impl AddressSpaceHostLayout for MemoryCellType {
    fn size(&self) -> usize {
        match self {
            Self::Null => 1, // to avoid divide by zero
            Self::U8 => size_of::<u8>(),
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
            Self::Native { size } => *size as usize,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, new)]
pub struct MemoryConfig {
    /// It is expected that the size of the list is `(1 << addr_space_height) + 1` and the first
    /// element is 0, which means no address space.
    pub addr_spaces: Vec<AddressSpaceHostConfig>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let mut addr_spaces = Self::empty_address_space_configs(2);
        const MAX_CELLS: usize = MAX_MEMORY >> 2;
        addr_spaces[MIPS_REGISTER_SPACE as usize].num_cells = NUM_REGISTERS;
        addr_spaces[MIPS_MEMORY_SPACE as usize].num_cells = MAX_CELLS;
        Self::new(addr_spaces)
    }
}

impl MemoryConfig {
    pub fn empty_address_space_configs(num_addr_spaces: usize) -> Vec<AddressSpaceHostConfig> {
        // All except address spaces 0..4 default to native 32-bit field.
        // By default only address spaces 1..=4 have non-empty cell counts.
        let mut addr_spaces = vec![
            AddressSpaceHostConfig::new(
                0,
                DEFAULT_NATIVE_BLOCK_SIZE,
                MemoryCellType::native32()
            );
            num_addr_spaces
        ];

        addr_spaces[MIPS_REGISTER_SPACE as usize] =
            AddressSpaceHostConfig::new(0, DEFAULT_NATIVE_BLOCK_SIZE, MemoryCellType::U32);

        addr_spaces[MIPS_MEMORY_SPACE as usize] =
            AddressSpaceHostConfig::new(0, DEFAULT_NATIVE_BLOCK_SIZE, MemoryCellType::U32);
        addr_spaces
    }
}
