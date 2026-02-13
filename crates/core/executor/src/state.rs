use std::{fmt::Debug, fs::File};

use hashbrown::HashMap;
use zkm_stark::{koala_bear_poseidon2::KoalaBearPoseidon2, StarkVerifyingKey};

use crate::{
    memory::{
        config::{MIPS_MEMORY_SPACE, MIPS_REGISTER_SPACE},
        GuestMemory, Memory,
    },
    record::{ExecutionRecord, MemoryAccessRecord},
    syscalls::SyscallCode,
    ExecutorMode, ZKMReduceProof,
};

/// Holds data describing the current state of a program's execution.
#[derive(Clone, Default)]
#[repr(C)]
pub struct ExecutionState {
    /// The program counter.
    pub pc: u32,

    // because of delayed slot
    pub next_pc: u32,

    /// The shard clock keeps track of how many shards have been executed.
    pub current_shard: u32,

    /// The clock increments by 5 (possibly more in syscalls) for each instruction that has been
    /// executed in this shard.
    pub clk: u32,

    /// The global clock keeps track of how many instructions have been executed through all shards.
    pub global_clk: u64,

    /// if exit
    pub exited: bool,

    /// if the next instruction is in delay slot for branch and jump.
    pub next_is_delayslot: bool,

    /// The memory which instructions operate over.
    pub memory: GuestMemory,

    #[cfg(not(feature = "aot-access"))]
    pub accessed: Memory<bool>,
    #[cfg(feature = "aot-access")]
    pub accessed: GuestMemory,

    /// Values contain the memory value and last shard + timestamp that each memory address was accessed.
    pub access_shard: GuestMemory,
    pub access_clk: GuestMemory,

    /// Max clocks for each record.
    pub records_clk: Vec<u32>,
    pub records_clk_index: u32,

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
    /// Keeps track of how many times a certain syscall has been called.
    pub syscall_counts: HashMap<SyscallCode, u64>,
}

impl ExecutionState {
    #[must_use]
    /// Create a new [`ExecutionState`].
    pub fn new(pc_start: u32, next_pc: u32) -> Self {
        let accessed = {
            #[cfg(not(feature = "aot-access"))]
            {
                Memory::new_preallocated()
            }
            #[cfg(feature = "aot-access")]
            {
                let mut accessed = GuestMemory::default();
                accessed.fill_zero();
                accessed
            }
        };

        Self {
            global_clk: 0,
            // Start at shard 1 since shard 0 is reserved for memory initialization.
            current_shard: 1,
            clk: 0,
            records_clk: vec![],
            records_clk_index: 0,
            pc: pc_start,
            next_pc,
            exited: false,
            next_is_delayslot: false,
            memory: GuestMemory::default(),
            accessed,
            access_shard: GuestMemory::default(),
            access_clk: GuestMemory::default(),
            uninitialized_memory: Memory::new_preallocated(),
            input_stream: Vec::new(),
            input_stream_ptr: 0,
            public_values_stream: Vec::new(),
            public_values_stream_ptr: 0,
            proof_stream: Vec::new(),
            proof_stream_ptr: 0,
            syscall_counts: HashMap::new(),
        }
    }

    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn read_register(&self, ptr: u32) -> u32 {
        let value = self.vm_read::<u32, 1>(MIPS_REGISTER_SPACE, ptr);
        value[0]
    }

    /// Runtime write operation for a block of memory
    #[inline(always)]
    pub fn write_register(&mut self, ptr: u32, value: u32) {
        self.vm_write::<u32, 1>(MIPS_REGISTER_SPACE, ptr, &[value]);
    }

    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn read_memory(&self, ptr: u32) -> u32 {
        let value = self.vm_read::<u32, 1>(MIPS_MEMORY_SPACE, ptr >> 2);
        value[0]
    }

    /// Runtime write operation for a block of memory
    #[inline(always)]
    pub fn write_memory(&mut self, ptr: u32, value: u32) {
        self.vm_write::<u32, 1>(MIPS_MEMORY_SPACE, ptr >> 2, &[value]);
    }

    /// Reads the shard and timestamp of the current register access.
    #[inline(always)]
    pub fn read_register_access_meta(&self, ptr: u32) -> (u32, u32) {
        let shard: [u32; 1] = unsafe { self.access_shard.read(MIPS_REGISTER_SPACE, ptr) };
        let clk: [u32; 1] = unsafe { self.access_clk.read(MIPS_REGISTER_SPACE, ptr) };
        (shard[0], clk[0])
    }

    /// Writes the shard and timestamp of the current register access to `access_shard` and `access_clk`.
    #[inline(always)]
    pub fn write_register_access_meta(&mut self, ptr: u32, shard: u32, clk: u32) {
        let shard: [u32; 1] = [shard];
        let clk: [u32; 1] = [clk];
        unsafe { self.access_shard.write(MIPS_REGISTER_SPACE, ptr, shard) };
        unsafe { self.access_clk.write(MIPS_REGISTER_SPACE, ptr, clk) };
    }

    /// Reads the shard and timestamp of the current memory access.
    #[inline(always)]
    pub fn read_memory_access_meta(&self, ptr: u32) -> (u32, u32) {
        let shard: [u32; 1] = unsafe { self.access_shard.read(MIPS_MEMORY_SPACE, ptr >> 2) };
        let clk: [u32; 1] = unsafe { self.access_clk.read(MIPS_MEMORY_SPACE, ptr >> 2) };
        (shard[0], clk[0])
    }

    // Writes the shard and timestamp of the current memory access to `access_shard` and `access_clk`.
    #[inline(always)]
    pub fn write_memory_access_meta(&mut self, ptr: u32, shard: u32, clk: u32) {
        let shard: [u32; 1] = [shard];
        let clk: [u32; 1] = [clk];
        unsafe { self.access_shard.write(MIPS_MEMORY_SPACE, ptr >> 2, shard) };
        unsafe { self.access_clk.write(MIPS_MEMORY_SPACE, ptr >> 2, clk) };
    }

    /// Mark a register as accessed.
    #[cfg(feature = "aot-access")]
    #[inline(always)]
    pub fn access_register(&mut self, ptr: u32) {
        let accessed: [u32; 1] = [1u32];
        unsafe { self.accessed.write(MIPS_REGISTER_SPACE, ptr, accessed) };
    }

    /// Mark a memory address as accessed.
    #[cfg(feature = "aot-access")]
    #[inline(always)]
    pub fn access_memory(&mut self, ptr: u32) {
        let accessed: [u32; 1] = [1u32];
        unsafe { self.accessed.write(MIPS_MEMORY_SPACE, ptr >> 2, accessed) };
    }

    #[cfg(feature = "aot-access")]
    #[inline(always)]
    pub fn register_not_accessed(&mut self, ptr: u32) -> bool {
        let accessed: [u32; 1] = unsafe { self.accessed.read(MIPS_REGISTER_SPACE, ptr) };
        accessed[0] == 0
    }

    #[cfg(feature = "aot-access")]
    #[inline(always)]
    pub fn memory_not_accessed(&mut self, ptr: u32) -> bool {
        let accessed: [u32; 1] = unsafe { self.accessed.read(MIPS_MEMORY_SPACE, ptr >> 2) };
        accessed[0] == 0
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
        unsafe { self.memory.read(addr_space, ptr) }
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
        unsafe { self.memory.write(addr_space, ptr, *data) }
    }
}

/// Holds data to track changes made to the runtime since a fork point.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
#[repr(C)]
pub struct ForkState {
    /// The `global_clk` value at the fork point.
    pub global_clk: u64,
    /// The original `clk` value at the fork point.
    pub clk: u32,
    /// The original `pc` value at the fork point.
    pub pc: u32,
    /// The original memory which instructions operate over.
    pub memory: GuestMemory,
    /// The original values contain the memory value and last shard + timestamp that each memory address was accessed.
    pub access_shard: GuestMemory,
    pub access_clk: GuestMemory,
    /// The original memory access record at the fork point.
    pub op_record: MemoryAccessRecord,
    /// The original execution record at the fork point.
    pub record: ExecutionRecord,
    // /// Whether `emit_events` was enabled at the fork point.
    pub executor_mode: ExecutorMode,
}

impl ExecutionState {
    /// Save the execution state to a file.
    pub fn save(&self, _file: &mut File) -> std::io::Result<()> {
        // let mut writer = std::io::BufWriter::new(file);
        // bincode::serialize_into(&mut writer, self).unwrap();
        // writer.flush()?;
        // writer.seek(std::io::SeekFrom::Start(0))?;
        Ok(())
    }
}
