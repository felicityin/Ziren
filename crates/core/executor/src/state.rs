use std::{
    collections::BTreeMap, fs::File, io::{Seek, Write}
};
use std::fmt::Debug;

use hashbrown::HashMap;
use itertools::zip_eq;
use serde::{Deserialize, Serialize};
use zkm_stark::{koala_bear_poseidon2::KoalaBearPoseidon2, StarkVerifyingKey};

use crate::{
    ExecutorMode, NUM_REGISTERS, ZKMReduceProof, events::{MemoryAccessMeta, MemoryRecord}, memory::Memory, record::{ExecutionRecord, MemoryAccessRecord}, syscalls::SyscallCode, vm::memory::{GuestMemory, LinearMemory, PagedVec, config::{AddressSpaceHostLayout, MIPS_MEMORY_SPACE, MIPS_REGISTER_SPACE}}
};

/// Default mmap page size. Change this if using THB.
pub const PAGE_SIZE: usize = 4096;

/// Holds data describing the current state of a program's execution.
#[derive(Debug, Clone, Default)]
#[repr(C)]
pub struct ExecutionState {
    /// The program counter.
    pub pc: u32,

    // because of delayed slot
    pub next_pc: u32,

    pub max_clks: Vec<u32>,

    /// The shard clock keeps track of how many shards have been executed.
    pub current_shard: u32,

    pub max_clks_index: u32,

    /// if exit
    pub exited: bool,

    /// if the next instruction is in delay slot for branch and jump.
    pub next_is_delayslot: bool,

    // /// The memory which instructions operate over. Values contain the memory value and last shard
    // /// + timestamp that each memory address was accessed.
    // pub memory: Memory<MemoryRecord>,

    pub mem: GuestMemory,
    pub mem_access_meta: Memory<MemoryAccessMeta>,
    /// For each `addr_space`, the minimum block size allowed for memory accesses. In other words,
    /// all memory accesses in `addr_space` must be aligned to this block size.
    // pub min_block_size: Vec<u32>,

    /// The global clock keeps track of how many instructions have been executed through all shards.
    pub global_clk: u64,

    /// The clock increments by 4 (possibly more in syscalls) for each instruction that has been
    /// executed in this shard.
    pub clk: u32,

    /// Uninitialized memory addresses that have a specific value they should be initialized with.
    /// `SyscallHintRead` uses this to write hint data into uninitialized memory.
    pub uninitialized_memory: Memory<u32>,

    /// A stream of input values (global to the entire program).
    pub input_stream: Vec<Vec<u8>>,

    /// A ptr to the current position in the input stream incremented by `HINT_READ` opcode.
    pub input_stream_ptr: usize,

    /// A stream of proofs (reduce vk, proof, verifying key) inputted to the program.
    pub proof_stream:
        Vec<(ZKMReduceProof<KoalaBearPoseidon2>, StarkVerifyingKey<KoalaBearPoseidon2>)>,

    /// A ptr to the current position in the proof stream, incremented after verifying a proof.
    pub proof_stream_ptr: usize,

    /// A stream of public values from the program (global to entire program).
    pub public_values_stream: Vec<u8>,

    /// A ptr to the current position in the public values stream, incremented when reading from
    /// `public_values_stream`.
    pub public_values_stream_ptr: usize,
    // /// Keeps track of how many times a certain syscall has been called.
    pub syscall_counts: HashMap<SyscallCode, u64>,
}

impl ExecutionState {
    #[must_use]
    /// Create a new [`ExecutionState`].
    pub fn new(pc_start: u32, next_pc: u32, image: &BTreeMap<u32, u32>) -> Self {
        let mem_access_meta = Memory::new_preallocated();
        // println!("------pc_start: {}", pc_start);
        // let mem = GuestMemory::new(image, &mut mem_access_meta);
        let mem = GuestMemory::default();

        // let (meta, min_block_size): (Vec<_>, Vec<_>) =
        //     zip_eq(mem.memory.get_memory(), &mem.memory.config)
        //         .map(|(mem, addr_sp)| {
        //             let num_cells = mem.size() / addr_sp.layout.size();
        //             let min_block_size = addr_sp.min_block_size;
        //             let total_metadata_len = num_cells.div_ceil(min_block_size);
        //             (PagedVec::new(total_metadata_len), min_block_size as u32)
        //         })
        //         .unzip();

        let mut state = Self {
            global_clk: 0,
            // Start at shard 1 since shard 0 is reserved for memory initialization.
            current_shard: 1,
            clk: 0,
            max_clks: vec![],
            max_clks_index: 0,
            pc: pc_start,
            next_pc,
            exited: false,
            next_is_delayslot: false,
            // memory: Memory::new_preallocated(),
            mem,
            mem_access_meta,
            // min_block_size,
            uninitialized_memory: Memory::new_preallocated(),
            input_stream: Vec::new(),
            input_stream_ptr: 0,
            public_values_stream: Vec::new(),
            public_values_stream_ptr: 0,
            proof_stream: Vec::new(),
            proof_stream_ptr: 0,
            syscall_counts: HashMap::new(),
        };

        // Initialize memory with program image.
        for (addr, value) in image {
            if *addr < NUM_REGISTERS as u32 {
                println!("Initializing register at addr {} with value {}", addr, value);
                state.mem_access_meta.registers.insert(*addr, MemoryAccessMeta::default());
                state.write_register(*addr, *value);
            } else {
                state.mem_access_meta.page_table.insert(*addr, MemoryAccessMeta::default());
                state.write_memory(*addr, *value);
            }
        }

        state
    }

    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn vm_read<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE] {
        // SAFETY:
        // - T is stack-allocated repr(C) or repr(transparent), usually u8 or F where F is the base
        //   field
        // - T is the exact memory cell type for this address space, satisfying the type requirement
        unsafe { self.mem.read(addr_space, ptr) }
    }

    /// Runtime write operation for a block of memory
    #[inline(always)]
    pub fn vm_write<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; BLOCK_SIZE],
    ) {
        // SAFETY:
        // - T is stack-allocated repr(C) or repr(transparent), usually u8 or F where F is the base
        //   field
        // - T is the exact memory cell type for this address space, satisfying the type requirement
        unsafe { self.mem.write(addr_space, ptr, *data) }
    }

    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn read_register(
        &self,
        ptr: u32,
    ) -> u32 {
        let value = self.vm_read::<u32, 1>(MIPS_REGISTER_SPACE, ptr);
        value[0]
        // u32::from_le_bytes(value)
    }

    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn write_register(
        &mut self,
        ptr: u32,
        value: u32,
    ) {
        // let value = value.to_le_bytes();
        self.vm_write::<u32, 1>(MIPS_REGISTER_SPACE, ptr, &[value]);

        // let value = self.vm_read::<u8, 4>(addr_space, ptr);
        // println!("-----vm write: {} {}", ptr, u32::from_le_bytes(value));
    }

    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn read_memory(
        &self,
        ptr: u32,
    ) -> u32 {
        let value = self.vm_read::<u32, 1>(MIPS_MEMORY_SPACE, ptr >> 2);
        value[0]
        // u32::from_le_bytes(value)
    }

    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn write_memory(
        &mut self,
        ptr: u32,
        value: u32,
    ) {
        // let value = value.to_le_bytes();
        self.vm_write::<u32, 1>(MIPS_MEMORY_SPACE, ptr >> 2, &[value]);

        // let value = self.vm_read::<u8, 4>(addr_space, ptr);
        // println!("-----vm write: {} {}", ptr, u32::from_le_bytes(value));
    }

    #[inline(always)]
    pub fn vm_read_slice<T: Copy + Debug>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        len: usize,
    ) -> &[T] {
        // SAFETY:
        // - T is stack-allocated repr(C) or repr(transparent), usually u8 or F where F is the base
        //   field
        // - T is the exact memory cell type for this address space, satisfying the type requirement
        // - panics if the slice is out of bounds
        unsafe { self.mem.get_slice(addr_space, ptr, len) }
    }
}

/// Holds data to track changes made to the runtime since a fork point.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ForkState {
    /// The `global_clk` value at the fork point.
    pub global_clk: u64,
    /// The original `clk` value at the fork point.
    pub clk: u32,
    /// The original `pc` value at the fork point.
    pub pc: u32,
    /// All memory changes since the fork point.
    pub memory_diff: HashMap<u32, Option<MemoryRecord>>,
    /// The original memory access record at the fork point.
    pub op_record: MemoryAccessRecord,
    /// The original execution record at the fork point.
    pub record: ExecutionRecord,
    // /// Whether `emit_events` was enabled at the fork point.
    pub executor_mode: ExecutorMode,
}

impl ExecutionState {
    /// Save the execution state to a file.
    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
        // let mut writer = std::io::BufWriter::new(file);
        // bincode::serialize_into(&mut writer, self).unwrap();
        // writer.flush()?;
        // writer.seek(std::io::SeekFrom::Start(0))?;
        Ok(())
    }
}
