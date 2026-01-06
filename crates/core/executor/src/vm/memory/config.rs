use std::{
    fs::File,
    io::{self, Write},
    path::Path,
};

use derive_new::new;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub const RV32_IMM_AS: u32 = 0;
pub const RV32_REGISTER_AS: u32 = 1;
pub const RV32_MEMORY_AS: u32 = 2;
pub const PUBLIC_VALUES_AS: u32 = 3;
pub const NATIVE_AS: u32 = 4;

// @dev Currently this is only used for debug assertions, but we may switch to making it constant
// and removing from MemoryConfig
pub const POINTER_MAX_BITS: usize = 29;

/// Offset for address space indices. This is used to distinguish between different memory spaces.
pub const ADDR_SPACE_OFFSET: u32 = 1;

pub const DEFAULT_MAX_NUM_PUBLIC_VALUES: usize = 32;

/// The minimum block size is 4, but RISC-V `lb` only requires alignment of 1 and `lh` only requires
/// alignment of 2 because the instructions are implemented by doing an access of block size 4.
const DEFAULT_U8_BLOCK_SIZE: usize = 4;
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

pub(crate) const MAX_CELL_BYTE_SIZE: usize = 8;

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
        Self::Native {
            size: size_of::<u32>() as u8,
        }
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

    // /// # Safety
    // /// - This function must only be called when `value` is guaranteed to be of size `self.size()`.
    // /// - Alignment of `value` must be a multiple of the alignment of `F`.
    // /// - The field type `F` must be plain old data.
    // unsafe fn to_field<F: Field>(&self, value: &[u8]) -> F;
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

    // /// # Safety
    // /// - This function must only be called when `value` is guaranteed to be of size `self.size()`.
    // /// - Alignment of `value` must be a multiple of the alignment of `F`.
    // /// - The field type `F` must be plain old data.
    // ///
    // /// # Panics
    // /// If the value is of integer type and overflows the field.
    // unsafe fn to_field<F: Field>(&self, value: &[u8]) -> F {
    //     match self {
    //         Self::Null => unreachable!(),
    //         Self::U8 => F::from_canonical_u8(*value.get_unchecked(0)),
    //         Self::U16 => F::from_canonical_u16(core::ptr::read(value.as_ptr() as *const u16)),
    //         Self::U32 => F::from_canonical_u32(core::ptr::read(value.as_ptr() as *const u32)),
    //         Self::Native { .. } => core::ptr::read(value.as_ptr() as *const F),
    //     }
    // }
}

#[derive(Debug, Serialize, Deserialize, Clone, new)]
pub struct MemoryConfig {
    /// The maximum height of the address space. This means the trie has `addr_space_height` layers
    /// for searching the address space. The allowed address spaces are those in the range `[1,
    /// 1 + 2^addr_space_height)` where it starts from 1 to not allow address space 0 in memory.
    pub addr_space_height: usize,
    /// It is expected that the size of the list is `(1 << addr_space_height) + 1` and the first
    /// element is 0, which means no address space.
    pub addr_spaces: Vec<AddressSpaceHostConfig>,
    pub pointer_max_bits: usize,
    /// All timestamps must be in the range `[0, 2^timestamp_max_bits)`. Maximum allowed: 29.
    pub timestamp_max_bits: usize,
    // /// Limb size used by the range checker
    // pub decomp: usize,
    // /// Maximum N AccessAdapter AIR to support.
    // pub max_access_adapter_n: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let mut addr_spaces =
            Self::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
        const MAX_CELLS: usize = 1 << 29;
        addr_spaces[RV32_REGISTER_AS as usize].num_cells = 32 * size_of::<u32>();
        addr_spaces[RV32_MEMORY_AS as usize].num_cells = MAX_CELLS;
        addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = DEFAULT_MAX_NUM_PUBLIC_VALUES;
        addr_spaces[NATIVE_AS as usize].num_cells = MAX_CELLS;
        Self::new(3, addr_spaces, POINTER_MAX_BITS, 29)
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
        addr_spaces[RV32_IMM_AS as usize] = AddressSpaceHostConfig::new(0, 1, MemoryCellType::Null);
        addr_spaces[RV32_REGISTER_AS as usize] =
            AddressSpaceHostConfig::new(0, DEFAULT_U8_BLOCK_SIZE, MemoryCellType::U8);

        addr_spaces[RV32_MEMORY_AS as usize] =
                AddressSpaceHostConfig::new(0, DEFAULT_U8_BLOCK_SIZE, MemoryCellType::U8);

        addr_spaces[PUBLIC_VALUES_AS as usize] =
            AddressSpaceHostConfig::new(0, DEFAULT_U8_BLOCK_SIZE, MemoryCellType::U8);

        addr_spaces
    }
}