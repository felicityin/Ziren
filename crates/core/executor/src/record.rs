use hashbrown::HashMap;
use itertools::{EitherOrBoth, Itertools};
use p3_field::FieldAlgebra;
use zkm_hypercube::{
    air::{
        AirLookup, LookupScope, PublicValues, ZKMAirBuilder, DEFAULT_PC_INC, ZKM_PROOF_NUM_PV_ELTS,
    },
    lookup::LookupKind,
    record::MachineRecord,
    septic_digest::SepticDigest,
};
use zkm_stark::SplitOpts;

use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, iter::once, mem::take, sync::Arc};

use crate::{
    events::{
        AluEvent, BranchEvent, BumpClkHighEvent, ByteLookupEvent, ByteRecord, CompAluEvent,
        CpuEvent, GlobalLookupEvent, JumpEvent, MemInstrEvent, MemoryInitializeFinalizeEvent,
        MemoryLocalEvent, MemoryRecordEnum, MiscEvent, MovCondEvent, PrecompileEvent,
        PrecompileEvents, SyscallEvent,
    },
    syscalls::{precompiles::keccak::sponge::GENERAL_BLOCK_SIZE_U32S, SyscallCode},
    Program,
};

/// A record of the execution of a program.
///
/// The trace of the execution is represented as a list of "events" that occur every cycle.
// todo: add logic opcode here, use bitwise_events
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ExecutionRecord {
    /// The program.
    pub program: Arc<Program>,
    /// A trace of the CPU events which get emitted during execution.
    pub cpu_events: Vec<CpuEvent>,
    /// The `pc` of this record's first retired instruction, if any. Updated once per
    /// instruction in `Executor::emit_events` independent of which per-opcode event `Vec` (or
    /// `cpu_events`) the instruction also lands in, so it stays correct even for opcodes that
    /// don't push a [`CpuEvent`].
    pub first_instruction_pc: Option<u32>,
    /// The `clk` of this record's first retired instruction, if any. Set alongside
    /// [`Self::first_instruction_pc`].
    pub first_instruction_clk: Option<u64>,
    /// The `next_pc` of this record's most recently retired instruction.
    pub last_next_pc: u32,
    /// The `exit_code` of this record's most recently retired instruction.
    pub last_exit_code: u32,
    /// The `clk` of this record's most recently retired instruction itself (*before* the
    /// `+ 5 + num_extra_cycles` step to [`Self::last_timestamp`]) -- i.e. the same `clk` that
    /// instruction's own chip populated its `CpuState`/sent via `eval_state_chain` with. Needed
    /// to recover *that row's* `clk_high` (`last_instruction_clk >> 24`), which is what
    /// `PublicValues::last_clk_high` must equal -- not `last_timestamp >> 24`, which silently
    /// carries into the next window's `clk_high` whenever this instruction's increment is what
    /// crosses the `clk_high` boundary (see `crates/core/machine/src/utils/prove.rs`'s use of
    /// this field for why that distinction matters).
    pub last_instruction_clk: u64,
    /// The expected `clk` of the instruction following this record's most recently retired one
    /// (`last_instruction_clk + 5 + num_extra_cycles`), deliberately unreduced/uncorrected even
    /// when it overflows past [`Self::last_instruction_clk`]'s own 24-bit `clk_low` window (see
    /// `eval_state_chain`'s doc comment on why `send_state` sends it exactly that way). Combined
    /// with [`Self::last_instruction_clk`]'s high limb into `PublicValues::last_clk_low` when
    /// populated (see `crates/core/machine/src/utils/prove.rs`) -- never masked to 24 bits on its
    /// own, unlike `last_instruction_clk`'s high limb.
    pub last_timestamp: u64,
    /// A trace of the register-form ADD and ADDU events (plus internal dependency-check rows
    /// from other chips reusing this arithmetic circuit).
    pub add_events: Vec<AluEvent>,
    /// A trace of the immediate-form ADDI and ADDIU events (register `b` plus an encoded
    /// immediate `c`, no register read for `c`).
    pub addi_events: Vec<AluEvent>,
    /// A trace of the SYNC and Pref events: compile-time-constant `op_a=0, op_b=0, op_c=0`, no
    /// register read for `b`/`c` at all (see `AddNoopChip`'s doc comment).
    pub add_noop_events: Vec<AluEvent>,
    /// A trace of the SUB and SUBU events (plus internal dependency-check rows from other chips
    /// reusing this arithmetic circuit).
    pub sub_events: Vec<AluEvent>,
    /// A trace of every real, retired `RTypeReader`-family instruction whose destination register
    /// is register 0 (`$zero`) -- today, register-form ADD/ADDU/SUB/SUBU with `op_a==0` (see
    /// `AluX0Chip`'s doc comment). Shared across opcodes: `AluEvent::opcode` distinguishes them.
    pub alu_x0_events: Vec<AluEvent>,
    /// A trace of the MUL, MULT and MULTU events.
    pub mul_events: Vec<CompAluEvent>,
    /// A trace of the XOR, OR, AND and NOR events.
    pub bitwise_events: Vec<AluEvent>,
    /// A trace of the SLL and SLLV events.
    pub shift_left_events: Vec<AluEvent>,
    /// A trace of the LUI events -- LUI decodes to `Opcode::SLL` with `imm_b=true` (`op_b` is the
    /// instruction's own encoded immediate, never a register), a distinct shape from SLL/SLLV
    /// that `LuiChip` handles separately (see its doc comment).
    pub lui_events: Vec<AluEvent>,
    /// A trace of the SRL, SRLV, SRA, and SRAV events.
    pub shift_right_events: Vec<AluEvent>,
    /// A trace of the DIV, DIVU events.
    pub divrem_events: Vec<CompAluEvent>,
    /// A trace of the register-form SLT and SLTU events (plus internal dependency-check rows
    /// from other chips reusing this comparison circuit).
    pub lt_events: Vec<AluEvent>,
    /// A trace of the immediate-form SLTI and SLTIU events (register `b` plus an encoded
    /// immediate `c`, see `SltiChip`'s doc comment).
    pub slti_events: Vec<AluEvent>,
    /// A trace of the CLO and CLZ events.
    pub cloclz_events: Vec<AluEvent>,
    /// A trace of the LW events.
    pub load_word_events: Vec<MemInstrEvent>,
    /// A trace of real, retired `lw $zero, ...` events (`LoadWordChip`'s zero-destination case,
    /// see `LoadX0Chip`'s doc comment).
    pub load_x0_events: Vec<MemInstrEvent>,
    /// A trace of the SW events.
    pub store_word_events: Vec<MemInstrEvent>,
    /// A trace of the LB/LBU events.
    pub load_byte_events: Vec<MemInstrEvent>,
    /// A trace of the LH/LHU events.
    pub load_half_events: Vec<MemInstrEvent>,
    /// A trace of the LWL/LWR events.
    pub load_word_unaligned_events: Vec<MemInstrEvent>,
    /// A trace of the SB events.
    pub store_byte_events: Vec<MemInstrEvent>,
    /// A trace of the SH events.
    pub store_half_events: Vec<MemInstrEvent>,
    /// A trace of the SWL/SWR events.
    pub store_word_unaligned_events: Vec<MemInstrEvent>,
    /// A trace of the SC events.
    pub store_conditional_events: Vec<MemInstrEvent>,
    /// A trace of the branch events.
    pub branch_events: Vec<BranchEvent>,
    /// A trace of the JR/JALR events.
    pub jump_events: Vec<JumpEvent>,
    /// A trace of the J/JAL events.
    pub jumpi_events: Vec<JumpEvent>,
    /// A trace of the BAL events.
    pub jumpdirect_events: Vec<JumpEvent>,
    /// A trace of the conditional move events.
    pub movcond_events: Vec<MovCondEvent>,
    /// A trace of the SEXT events.
    pub sext_events: Vec<MiscEvent>,
    /// A trace of the INS events.
    pub ins_events: Vec<MiscEvent>,
    /// A trace of the EXT events.
    pub ext_events: Vec<MiscEvent>,
    /// A trace of the MADD/MADDU/MSUB/MSUBU events.
    pub maddsub_events: Vec<MiscEvent>,
    /// A trace of the TEQ events.
    pub teq_events: Vec<MiscEvent>,
    /// A trace of the byte lookups that are needed.
    pub byte_lookups: HashMap<ByteLookupEvent, usize>,
    /// A trace of the precompile events.
    pub precompile_events: PrecompileEvents,
    // /// A trace of the global memory initialize events.
    pub global_memory_initialize_events: Vec<MemoryInitializeFinalizeEvent>,
    // /// A trace of the global memory finalize events.
    pub global_memory_finalize_events: Vec<MemoryInitializeFinalizeEvent>,
    /// A trace of all the shard's local memory events.
    pub cpu_local_memory_access: Vec<MemoryLocalEvent>,
    /// A trace of all the syscall events.
    pub syscall_events: Vec<SyscallEvent>,
    /// A trace of all the global lookup events.
    pub global_lookup_events: Vec<GlobalLookupEvent>,
    /// A trace of `clk_high` boundary crossings (see [`BumpClkHighEvent`]'s doc comment).
    pub bump_clk_high_events: Vec<BumpClkHighEvent>,
    /// Register timestamp re-stamps: `(the record, register address)`, emitted whenever a real
    /// register access's own `clk_high` differs from that register's previous access. Consumed
    /// by the `MemoryBumpChip` AIR, which re-validates the access at full cost so that every
    /// other register access can assume its previous access shares the same `clk_high` (see
    /// `MemoryBumpChip`'s doc comment).
    pub bump_memory_events: Vec<(MemoryRecordEnum, u32)>,
    /// The public values.
    pub public_values: PublicValues<u32, u32>,
}

impl ExecutionRecord {
    /// Create a new [`ExecutionRecord`].
    #[must_use]
    #[cfg(feature = "pre-alloc")]
    pub fn new(program: Arc<Program>) -> Self {
        let cpu_events = Vec::with_capacity(1 << 22);
        let add_events = Vec::with_capacity(1 << 22);
        let addi_events = Vec::with_capacity(1 << 22);
        let sub_events = Vec::with_capacity(1 << 21);
        let load_word_events = Vec::with_capacity(1 << 22);
        let store_word_events = Vec::with_capacity(1 << 22);
        Self {
            program,
            cpu_events,
            load_word_events,
            store_word_events,
            add_events,
            addi_events,
            sub_events,
            ..Default::default()
        }
    }

    #[must_use]
    #[cfg(not(feature = "pre-alloc"))]
    pub fn new(program: Arc<Program>) -> Self {
        Self { program, ..Default::default() }
    }

    /// Add a mul event to the execution record.
    pub fn add_mul_event(&mut self, mul_event: CompAluEvent) {
        self.mul_events.push(mul_event);
    }

    /// Add a lt event to the execution record.
    pub fn add_lt_event(&mut self, lt_event: AluEvent) {
        self.lt_events.push(lt_event);
    }

    /// Take out events from the [`ExecutionRecord`] that should be deferred to a separate shard.
    ///
    /// Note: we usually defer events that would increase the recursion cost significantly if
    /// included in every shard.
    #[must_use]
    pub fn defer(&mut self) -> ExecutionRecord {
        let mut execution_record = ExecutionRecord::new(self.program.clone());
        execution_record.precompile_events = std::mem::take(&mut self.precompile_events);
        execution_record.global_memory_initialize_events =
            std::mem::take(&mut self.global_memory_initialize_events);
        execution_record.global_memory_finalize_events =
            std::mem::take(&mut self.global_memory_finalize_events);
        execution_record
    }

    /// Splits the deferred [`ExecutionRecord`] into multiple [`ExecutionRecord`]s, each which
    /// contain a "reasonable" number of deferred events.
    ///
    /// The optional `last_record` will be provided if there are few enough deferred events that
    /// they can all be packed into the already existing last record.
    pub fn split(
        &mut self,
        last: bool,
        last_record: Option<&mut ExecutionRecord>,
        opts: SplitOpts,
    ) -> Vec<ExecutionRecord> {
        let mut shards = Vec::new();

        let precompile_events = take(&mut self.precompile_events);

        for (syscall_code, events) in precompile_events.into_iter() {
            let threshold = match syscall_code {
                SyscallCode::KECCAK_SPONGE => opts.keccak,
                SyscallCode::SHA_EXTEND => opts.sha_extend,
                SyscallCode::SHA_COMPRESS => opts.sha_compress,
                _ => opts.deferred,
            };

            let mut shards_input = Vec::new();
            let remainder = match syscall_code {
                SyscallCode::KECCAK_SPONGE => {
                    let mut current_shard = Vec::new();
                    let mut current_len = 0;

                    for (syscall_event, event) in events {
                        if let PrecompileEvent::KeccakSponge(event) = &event {
                            // Here, input_len_u32s must be a multiple of GENERAL_BLOCK_SIZE_U32S.
                            let input_len = event.input_len_u32s as usize / GENERAL_BLOCK_SIZE_U32S;

                            if current_len + input_len > threshold && !current_shard.is_empty() {
                                let mut record = ExecutionRecord::new(self.program.clone());
                                record.precompile_events.insert(syscall_code, current_shard);
                                shards_input.push(record);
                                current_shard = Vec::new();
                                current_len = 0;
                            }
                            current_len += input_len;
                        }
                        current_shard.push((syscall_event, event));
                    }
                    current_shard
                }
                _ => {
                    let chunks = events.chunks_exact(threshold);
                    let remainder = chunks.remainder().to_vec();
                    for chunk in chunks {
                        let mut record = ExecutionRecord::new(self.program.clone());
                        record.precompile_events.insert(syscall_code, chunk.to_vec());
                        shards_input.push(record);
                    }
                    remainder
                }
            };

            if !remainder.is_empty() {
                if last {
                    let mut record = ExecutionRecord::new(self.program.clone());
                    record.precompile_events.insert(syscall_code, remainder);
                    shards_input.push(record);
                } else {
                    self.precompile_events.insert(syscall_code, remainder);
                }
            }

            shards.extend(shards_input);
        }

        if last {
            self.global_memory_initialize_events.sort_by_key(|event| event.addr);
            self.global_memory_finalize_events.sort_by_key(|event| event.addr);

            // If there are no precompile shards, and `last_record` is provided, pack the memory events
            // into the last record.
            let pack_memory_events_into_last_record = last_record.is_some() && shards.is_empty();
            let mut blank_record = ExecutionRecord::new(self.program.clone());

            // If `last_record` is None, use a blank record to store the memory events.
            let last_record_ref = if pack_memory_events_into_last_record {
                last_record.unwrap()
            } else {
                &mut blank_record
            };

            let mut init_addr = 0;
            let mut finalize_addr = 0;
            for mem_chunks in self
                .global_memory_initialize_events
                .chunks(opts.memory)
                .zip_longest(self.global_memory_finalize_events.chunks(opts.memory))
            {
                let (mem_init_chunk, mem_finalize_chunk) = match mem_chunks {
                    EitherOrBoth::Both(mem_init_chunk, mem_finalize_chunk) => {
                        (mem_init_chunk, mem_finalize_chunk)
                    }
                    EitherOrBoth::Left(mem_init_chunk) => (mem_init_chunk, [].as_slice()),
                    EitherOrBoth::Right(mem_finalize_chunk) => ([].as_slice(), mem_finalize_chunk),
                };
                last_record_ref.global_memory_initialize_events.extend_from_slice(mem_init_chunk);
                last_record_ref.public_values.previous_init_addr = init_addr;
                if let Some(last_event) = mem_init_chunk.last() {
                    init_addr = last_event.addr;
                }
                last_record_ref.public_values.last_init_addr = init_addr;

                last_record_ref.global_memory_finalize_events.extend_from_slice(mem_finalize_chunk);
                last_record_ref.public_values.previous_finalize_addr = finalize_addr;
                if let Some(last_event) = mem_finalize_chunk.last() {
                    finalize_addr = last_event.addr;
                }
                last_record_ref.public_values.last_finalize_addr = finalize_addr;

                if !pack_memory_events_into_last_record {
                    // If not packing memory events into the last record, add 'last_record_ref'
                    // to the returned records. `take` replaces `blank_program` with the default.
                    shards.push(take(last_record_ref));

                    // Reset the last record so its program is the correct one. (The default program
                    // provided by `take` contains no instructions.)
                    last_record_ref.program = self.program.clone();
                }
            }
        }

        shards
    }

    /// Determines whether the execution record retired any instructions.
    #[must_use]
    pub fn contains_cpu(&self) -> bool {
        self.first_instruction_pc.is_some()
    }

    #[inline]
    /// Add a precompile event to the execution record.
    pub fn add_precompile_event(
        &mut self,
        syscall_code: SyscallCode,
        syscall_event: SyscallEvent,
        event: PrecompileEvent,
    ) {
        self.precompile_events.add_event(syscall_code, syscall_event, event);
    }

    /// Get all the precompile events for a syscall code.
    #[inline]
    #[must_use]
    pub fn get_precompile_events(
        &self,
        syscall_code: SyscallCode,
    ) -> &Vec<(SyscallEvent, PrecompileEvent)> {
        self.precompile_events.get_events(syscall_code).expect("Precompile events not found")
    }

    /// Get all the local memory events.
    #[inline]
    pub fn get_local_mem_events(&self) -> impl Iterator<Item = &MemoryLocalEvent> {
        let precompile_local_mem_events = self.precompile_events.get_local_mem_events();
        precompile_local_mem_events.chain(self.cpu_local_memory_access.iter())
    }
}

/// A memory access record.
#[derive(Debug, Copy, Clone, Default)]
pub struct MemoryAccessRecord {
    /// The memory access of the `a` register. read && write
    pub a: Option<MemoryRecordEnum>,
    /// The memory access of the `b` register.
    pub b: Option<MemoryRecordEnum>,
    /// The memory access of the `c` register.
    pub c: Option<MemoryRecordEnum>,
    /// The memory access of the `hi` register and other special registers.
    /// read && write
    pub hi: Option<MemoryRecordEnum>,
    /// The memory access of the `memory` register.
    pub memory: Option<MemoryRecordEnum>,
}

impl MachineRecord for ExecutionRecord {
    fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cpu_events".to_string(), self.cpu_events.len());
        stats.insert("add_events".to_string(), self.add_events.len());
        stats.insert("addi_events".to_string(), self.addi_events.len());
        stats.insert("add_noop_events".to_string(), self.add_noop_events.len());
        stats.insert("sub_events".to_string(), self.sub_events.len());
        stats.insert("alu_x0_events".to_string(), self.alu_x0_events.len());
        stats.insert("mul_events".to_string(), self.mul_events.len());
        stats.insert("bitwise_events".to_string(), self.bitwise_events.len());
        stats.insert("shift_left_events".to_string(), self.shift_left_events.len());
        stats.insert("lui_events".to_string(), self.lui_events.len());
        stats.insert("shift_right_events".to_string(), self.shift_right_events.len());
        stats.insert("divrem_events".to_string(), self.divrem_events.len());
        stats.insert("lt_events".to_string(), self.lt_events.len());
        stats.insert("slti_events".to_string(), self.slti_events.len());
        stats.insert("cloclz_events".to_string(), self.cloclz_events.len());
        stats.insert("load_word_events".to_string(), self.load_word_events.len());
        stats.insert("load_x0_events".to_string(), self.load_x0_events.len());
        stats.insert("store_word_events".to_string(), self.store_word_events.len());
        stats.insert("load_byte_events".to_string(), self.load_byte_events.len());
        stats.insert("load_half_events".to_string(), self.load_half_events.len());
        stats.insert(
            "load_word_unaligned_events".to_string(),
            self.load_word_unaligned_events.len(),
        );
        stats.insert("store_byte_events".to_string(), self.store_byte_events.len());
        stats.insert("store_half_events".to_string(), self.store_half_events.len());
        stats.insert(
            "store_word_unaligned_events".to_string(),
            self.store_word_unaligned_events.len(),
        );
        stats.insert("store_conditional_events".to_string(), self.store_conditional_events.len());
        stats.insert("branch_events".to_string(), self.branch_events.len());
        stats.insert("jump_events".to_string(), self.jump_events.len());
        stats.insert("jumpi_events".to_string(), self.jumpi_events.len());
        stats.insert("jumpdirect_events".to_string(), self.jumpdirect_events.len());
        stats.insert("sext_events".to_string(), self.sext_events.len());
        stats.insert("ins_events".to_string(), self.ins_events.len());
        stats.insert("ext_events".to_string(), self.ext_events.len());
        stats.insert("maddsub_events".to_string(), self.maddsub_events.len());
        stats.insert("teq_events".to_string(), self.teq_events.len());

        for (syscall_code, events) in self.precompile_events.iter() {
            stats.insert(format!("syscall {syscall_code:?}"), events.len());
        }

        stats.insert(
            "global_memory_initialize_events".to_string(),
            self.global_memory_initialize_events.len(),
        );
        stats.insert(
            "global_memory_finalize_events".to_string(),
            self.global_memory_finalize_events.len(),
        );
        stats.insert("local_memory_access_events".to_string(), self.cpu_local_memory_access.len());
        stats.insert("bump_memory_events".to_string(), self.bump_memory_events.len());
        stats.insert("byte_lookups".to_string(), self.byte_lookups.len());
        // Filter out the empty events.
        stats.retain(|_, v| *v != 0);
        stats
    }

    fn append(&mut self, other: &mut ExecutionRecord) {
        self.cpu_events.append(&mut other.cpu_events);
        self.add_events.append(&mut other.add_events);
        self.addi_events.append(&mut other.addi_events);
        self.add_noop_events.append(&mut other.add_noop_events);
        self.sub_events.append(&mut other.sub_events);
        self.alu_x0_events.append(&mut other.alu_x0_events);
        self.mul_events.append(&mut other.mul_events);
        self.bitwise_events.append(&mut other.bitwise_events);
        self.shift_left_events.append(&mut other.shift_left_events);
        self.lui_events.append(&mut other.lui_events);
        self.shift_right_events.append(&mut other.shift_right_events);
        self.divrem_events.append(&mut other.divrem_events);
        self.lt_events.append(&mut other.lt_events);
        self.slti_events.append(&mut other.slti_events);
        self.cloclz_events.append(&mut other.cloclz_events);
        self.load_word_events.append(&mut other.load_word_events);
        self.load_x0_events.append(&mut other.load_x0_events);
        self.store_word_events.append(&mut other.store_word_events);
        self.load_byte_events.append(&mut other.load_byte_events);
        self.load_half_events.append(&mut other.load_half_events);
        self.load_word_unaligned_events.append(&mut other.load_word_unaligned_events);
        self.store_byte_events.append(&mut other.store_byte_events);
        self.store_half_events.append(&mut other.store_half_events);
        self.store_word_unaligned_events.append(&mut other.store_word_unaligned_events);
        self.store_conditional_events.append(&mut other.store_conditional_events);
        self.branch_events.append(&mut other.branch_events);
        self.jump_events.append(&mut other.jump_events);
        self.jumpi_events.append(&mut other.jumpi_events);
        self.jumpdirect_events.append(&mut other.jumpdirect_events);
        self.sext_events.append(&mut other.sext_events);
        self.ins_events.append(&mut other.ins_events);
        self.ext_events.append(&mut other.ext_events);
        self.maddsub_events.append(&mut other.maddsub_events);
        self.teq_events.append(&mut other.teq_events);
        self.syscall_events.append(&mut other.syscall_events);

        self.precompile_events.append(&mut other.precompile_events);

        if self.byte_lookups.is_empty() {
            self.byte_lookups = std::mem::take(&mut other.byte_lookups);
        } else {
            self.add_byte_lookup_events_from_maps(vec![&other.byte_lookups]);
        }

        self.global_memory_initialize_events.append(&mut other.global_memory_initialize_events);
        self.global_memory_finalize_events.append(&mut other.global_memory_finalize_events);
        self.cpu_local_memory_access.append(&mut other.cpu_local_memory_access);
        self.global_lookup_events.append(&mut other.global_lookup_events);
        self.bump_clk_high_events.append(&mut other.bump_clk_high_events);
        self.bump_memory_events.append(&mut other.bump_memory_events);

        // `Machine::generate_dependencies` calls each chip's `generate_dependencies` with a
        // fresh, per-chip `other` record and merges it in via this method -- `MemoryGlobalChip`
        // writes its shard-local event count into `other.public_values.global_{init,finalize}_count`
        // (see `memory/global.rs`), which must be carried into `self` here or it's silently
        // dropped when `other` goes out of scope.
        self.public_values.global_init_count += other.public_values.global_init_count;
        self.public_values.global_finalize_count += other.public_values.global_finalize_count;

        // Same story for `GlobalChip`'s `global_count`/`global_cumulative_sum_{x,y}` (see
        // `global/mod.rs::generate_dependencies`).
        self.public_values.global_count += other.public_values.global_count;
        for i in 0..7 {
            self.public_values.global_cumulative_sum_x[i] +=
                other.public_values.global_cumulative_sum_x[i];
            self.public_values.global_cumulative_sum_y[i] +=
                other.public_values.global_cumulative_sum_y[i];
        }
    }

    /// Retrieves the public values.  This method is needed for the `MachineRecord` trait, since
    fn public_values<F: FieldAlgebra>(&self) -> Vec<F> {
        self.public_values.to_vec()
    }

    fn eval_public_values<AB: ZKMAirBuilder>(builder: &mut AB) {
        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<zkm_hypercube::word::Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        // Anchor the CPU's `LookupKind::State` local interaction chain at the shard's boundary:
        // the shard's first real CPU row has nothing in-shard to receive its incoming state from,
        // and the last real row has nothing in-shard to receive its outgoing state -- these two
        // sends/receives close that chain against public values instead. A shard boundary is only
        // ever sequential (see `CpuChip::eval_pc`), so the paired pc is always `pc + 4`. Gated on
        // `is_execution_shard` so a shard that retired no instructions contributes nothing to the
        // chain, rather than relying on `start_pc == next_pc` / `(initial_clk_high,
        // initial_clk_low) == (last_clk_high, last_clk_low)` to self-cancel.
        let pc_inc = AB::Expr::from_canonical_u32(DEFAULT_PC_INC);
        builder.send_state(
            public_values.initial_clk_high,
            public_values.initial_clk_low,
            public_values.start_pc,
            public_values.start_pc.into() + pc_inc.clone(),
            public_values.is_execution_shard.into(),
        );
        builder.receive_state(
            public_values.last_clk_high,
            public_values.last_clk_low,
            public_values.next_pc,
            public_values.next_pc.into() + pc_inc,
            public_values.is_execution_shard.into(),
        );

        // Anchor `MemoryGlobalChip`'s (Init and Finalize instantiations) `index`-keyed
        // sorted-address chains the same way: the shard's first real row's `prev_addr` receive
        // has nothing in-shard to match, and the last real row's `addr` send (at
        // `index + 1 == global_*_count`) has nothing in-shard to match either.
        builder.send(
            AirLookup::new(
                once(AB::Expr::zero())
                    .chain(public_values.previous_init_addr.0.iter().cloned().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                LookupKind::MemoryGlobalInitControl,
            ),
            LookupScope::Local,
        );
        builder.receive(
            AirLookup::new(
                once(public_values.global_init_count.into())
                    .chain(public_values.last_init_addr.0.iter().cloned().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                LookupKind::MemoryGlobalInitControl,
            ),
            LookupScope::Local,
        );
        builder.send(
            AirLookup::new(
                once(AB::Expr::zero())
                    .chain(public_values.previous_finalize_addr.0.iter().cloned().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                LookupKind::MemoryGlobalFinalizeControl,
            ),
            LookupScope::Local,
        );
        builder.receive(
            AirLookup::new(
                once(public_values.global_finalize_count.into())
                    .chain(public_values.last_finalize_addr.0.iter().cloned().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                LookupKind::MemoryGlobalFinalizeControl,
            ),
            LookupScope::Local,
        );

        // Anchor `GlobalChip`'s `LookupKind::GlobalAccumulation` chain: the start (`index == 0`)
        // is always the constant zero digest (not a cross-shard value, so it's witnessed as a
        // constant here rather than threaded through a `previous_*` field), and the end (at
        // `index == global_count`) is this shard's own accumulated digest.
        let zero_digest = SepticDigest::<AB::F>::zero().0;
        builder.send(
            AirLookup::new(
                once(AB::Expr::zero())
                    .chain(zero_digest.x.0.into_iter().map(Into::into))
                    .chain(zero_digest.y.0.into_iter().map(Into::into))
                    .collect(),
                AB::Expr::one(),
                LookupKind::GlobalAccumulation,
            ),
            LookupScope::Local,
        );
        builder.receive(
            AirLookup::new(
                once(public_values.global_count.into())
                    .chain(public_values.global_cumulative_sum_x.iter().cloned().map(Into::into))
                    .chain(public_values.global_cumulative_sum_y.iter().cloned().map(Into::into))
                    .collect(),
                AB::Expr::one(),
                LookupKind::GlobalAccumulation,
            ),
            LookupScope::Local,
        );
    }

    fn interactions_in_public_values() -> Vec<LookupKind> {
        LookupKind::all_kinds().into_iter().filter(LookupKind::appears_in_eval_public_values).collect()
    }
}

impl ByteRecord for ExecutionRecord {
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent) {
        *self.byte_lookups.entry(blu_event).or_insert(0) += 1;
    }

    #[inline]
    fn add_byte_lookup_events_from_maps(
        &mut self,
        new_events: Vec<&HashMap<ByteLookupEvent, usize>>,
    ) {
        for new_blu_map in new_events {
            for (blu_event, count) in new_blu_map.iter() {
                *self.byte_lookups.entry(*blu_event).or_insert(0) += count;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forces `global_memory_initialize_events`/`global_memory_finalize_events` across more than
    /// one shard (via a tiny `opts.memory` threshold) and checks that the
    /// `previous_init_addr`/`last_init_addr`/`previous_finalize_addr`/`last_finalize_addr` chain
    /// threads correctly shard-to-shard -- the one path the plain-`u32` retype (replacing the old
    /// `[T; 32]` bit-array reconstruction) doesn't otherwise get exercised by, since no existing
    /// end-to-end test's memory footprint is large enough to cross `SplitOpts::memory`'s
    /// production-scale default threshold.
    #[test]
    fn test_split_memory_chain_across_multiple_shards() {
        let program = Arc::new(Program::default());
        let mut record = ExecutionRecord::new(program);
        for i in 0..10u32 {
            record.global_memory_initialize_events.push(MemoryInitializeFinalizeEvent {
                addr: (i + 1) * 4,
                value: i,
                timestamp: 0,
            });
            record.global_memory_finalize_events.push(MemoryInitializeFinalizeEvent {
                addr: (i + 1) * 4,
                value: i * 2,
                timestamp: 1,
            });
        }

        let mut opts = SplitOpts::new(zkm_stark::MAX_DEFERRED_SPLIT_THRESHOLD);
        opts.memory = 3;
        let shards = record.split(true, None, opts);
        assert!(
            shards.len() >= 4,
            "expected the memory chain to be split across multiple shards, got {}",
            shards.len()
        );

        let mut expected_previous_init = 0u32;
        let mut expected_previous_finalize = 0u32;
        for shard in &shards {
            assert_eq!(shard.public_values.previous_init_addr, expected_previous_init);
            assert_eq!(shard.public_values.previous_finalize_addr, expected_previous_finalize);

            if let Some(last) = shard.global_memory_initialize_events.last() {
                expected_previous_init = last.addr;
            }
            if let Some(last) = shard.global_memory_finalize_events.last() {
                expected_previous_finalize = last.addr;
            }

            assert_eq!(shard.public_values.last_init_addr, expected_previous_init);
            assert_eq!(shard.public_values.last_finalize_addr, expected_previous_finalize);
        }
    }
}
