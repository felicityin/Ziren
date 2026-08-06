//! `TracingVM` -- replays a [`SplicedMinimalTrace`]-shaped [`MinimalTrace`] shard and constructs
//! the `ExecutionRecord` events that describe it, using the same oracle-driven `CoreVM` primitives
//! `SplicingVM` uses for shard-cut accounting. Event *values* (opcode, operands, results) and
//! event *counts* (which `ExecutionRecord` vector each instruction routes to, including the
//! `op_a == 0` special cases the legacy executor's chips route separately) are built to match the
//! legacy `Executor` exactly.
//!
//! Per-operand `MemoryRecordEnum`/`MemoryWriteRecord` fields on each event (the register/memory
//! timestamp bookkeeping the AIR's memory-consistency argument needs for real proving) carry real
//! chased-through-every-access values, matching the legacy executor exactly, as does
//! `bump_memory_events`. `cpu_local_memory_access` is populated for every register touch and
//! every load/store's own RAM access; a precompile's own scratch-buffer touches (SHA/EC/Keccak/
//! etc's oracle-popped words) go into that event's own `local_mem_access` field instead, via
//! `precompile_local_events` -- both eventually feed `ExecutionRecord::get_local_mem_events()`,
//! which chains them, but they're populated through separate paths because `ExecutionRecord::
//! split` always carves a precompile event into its own record, so its own RAM touches must
//! travel with it (see `precompile_local_events`'s doc comment for the full reasoning, including
//! why building this via the shared map instead caused a real cross-shard bridging bug).
//! `global_memory_initialize_events`/`global_memory_finalize_events`/`global_lookup_events` are
//! not populated at all: they need a whole-run pass over `MinimalExecutor`'s own backing memory
//! (this scoped, one-shard-at-a-time `TracingVM` has no view of addresses touched in other
//! shards, or of whether this is the last shard at all) -- follow-up work, not built here yet.

#![allow(dead_code)]

use std::sync::Arc;

use crate::{
    events::{
        AluEvent, BranchEvent, BumpClkHighEvent, CompAluEvent, EdDecompressEvent,
        EllipticCurveAddEvent, EllipticCurveDecompressEvent, EllipticCurveDoubleEvent,
        FieldOperation, Fp2AddSubEvent, Fp2MulEvent, FpOpEvent, JumpEvent, KeccakSpongeEvent,
        LinuxEvent, MemInstrEvent, MemoryAccessPosition, MemoryLocalEvent, MemoryReadRecord,
        MemoryRecordEnum, MemoryWriteRecord, MiscEvent, MovCondEvent, Poseidon2PermuteEvent, PrecompileEvent,
        ShaCompressEvent, ShaExtendEvent, SyscallEvent, U256xU2048MulEvent, Uint256MulEvent,
    },
    opcode::Opcode,
    register::Register,
    syscalls::SyscallCode,
    trace::{MemValue, MinimalTrace},
    vm::{
        align_size, ec_add, ec_decompress, ec_double, ec_num_limb_words, ec_num_words,
        ed25519_decompress, fcntl_result, fp2_addsub, fp2_mul, fp2_num_words, fp_num_words, fp_op,
        keccak_xor_block, keccakf, poseidon2_permute, read_result, resolve_brk, sha256_compress,
        sha256_extend_word, u256xu2048_mul, uint256_mul, CoreVM, CoreVMStatus,
        KECCAK_GENERAL_BLOCK_SIZE_U64S, KECCAK_GENERAL_OUTPUT_U64S, KECCAK_STATE_SIZE_U64S,
        POSEIDON2_STATE_SIZE, U2048_NUM_WORDS, U256_NUM_WORDS,
    },
    ExecutionError, ExecutionRecord, Instruction, Program,
};
use zkm_curves::{
    edwards::{ed25519::Ed25519, WORDS_FIELD_ELEMENT},
    weierstrass::{
        bls12_381::{Bls12381, Bls12381BaseField},
        bn254::{Bn254, Bn254BaseField},
        secp256k1::Secp256k1,
        secp256r1::Secp256r1,
        FpOpField,
    },
    EllipticCurve, COMPRESSED_POINT_BYTES,
};

pub(crate) struct TracingVM<'a> {
    core: CoreVM<'a>,
    record: &'a mut ExecutionRecord,
    /// Per-shard first/last-touch bookkeeping, drained into `record.cpu_local_memory_access` at
    /// the end of `execute()` -- mirrors `Executor::local_memory_access`. Fed by register touches,
    /// load/store's own RAM access, and `execute_syscall`'s own register-only touches (A2/A3/HEAP
    /// for the Linux syscall shims) -- every touch that isn't a precompile's own private RAM
    /// buffer. A precompile's own scratch-buffer touches (SHA/EC/Keccak/etc's oracle-popped words)
    /// go into that event's own `local_mem_access` field instead, via `precompile_local_events` --
    /// see that method's doc comment for why.
    local_memory_access: std::collections::HashMap<u32, MemoryLocalEvent>,
    /// Some multi-row precompiles (SHA_EXTEND, SHA_COMPRESS, KECCAK_SPONGE, ...) process many
    /// internal memory-touching rows under one shared, flat instruction-level `clk` (`clk` only
    /// advances between instructions, never within one), disambiguated only *symbolically* at
    /// AIR-evaluation time via `clk_low + <some per-row offset>` (see e.g. `sha256/extend/air.rs`'s
    /// `clk_low + (local.i - i_start)`) -- never actually stored anywhere `MinimalExecutor`'s
    /// oracle log could reflect, since it has no notion of "rows" within one instruction. When one
    /// of these rows touches an address a *previous* row (of this same precompile call, or an
    /// earlier call to the same kind of precompile) already touched with such an adjusted
    /// timestamp, the oracle's own `.clk` for that address is stale -- it still reports the flat
    /// instruction clk of whichever touch produced it, not the adjusted timestamp that touch's own
    /// row actually sent into the interaction argument. This map, consulted via `oracle_entry_at`
    /// and populated via `record_adjusted_touch`, overrides the oracle's `.clk` with the real
    /// adjusted value in that case. Scoped to one `TracingVM` (one shard) deliberately: a
    /// cross-shard version of this same reference is bridged by the separate, already-existing
    /// `Global`-scope mechanism, not this shard-local one.
    adjusted_prev_timestamps: std::collections::HashMap<u32, u64>,
}

impl<'a> TracingVM<'a> {
    #[must_use]
    pub(crate) fn new<T: MinimalTrace>(
        trace: &'a T,
        program: Arc<Program>,
        max_syscall_cycles: u32,
        record: &'a mut ExecutionRecord,
    ) -> Self {
        Self {
            core: CoreVM::new(trace, program, max_syscall_cycles),
            record,
            local_memory_access: std::collections::HashMap::new(),
            adjusted_prev_timestamps: std::collections::HashMap::new(),
        }
    }

    /// Pops the next oracle entry for a touch to `addr`, overriding its `.clk` with this
    /// address's adjusted timestamp if a previous row recorded one via `record_adjusted_touch`
    /// (see that field's doc comment) -- and consuming that record, since this touch's own
    /// timestamp (adjusted or not) is what the *next* toucher of `addr` should see as previous.
    fn oracle_entry_at(&mut self, addr: u32) -> MemValue {
        let mut e = self.core.next_oracle_entry();
        if let Some(adjusted) = self.adjusted_prev_timestamps.remove(&addr) {
            e.clk = adjusted;
        }
        e
    }

    /// Records that `addr`'s current state, as of this row, should be considered to exist at
    /// `adjusted_timestamp` (a symbolic, AIR-side timestamp the oracle log itself has no notion
    /// of) rather than the flat instruction `clk` the oracle would otherwise report -- see
    /// `adjusted_prev_timestamps`'s doc comment. Call after constructing every read or write
    /// record a multi-row precompile's own row produces at a non-flat effective timestamp (reads
    /// included: a read still "produces" a fresh same-value token at its own row's timestamp, via
    /// `eval_memory_access`'s receive half, for whichever touch comes next to consume).
    fn record_adjusted_touch(&mut self, addr: u32, adjusted_timestamp: u64) {
        self.adjusted_prev_timestamps.insert(addr, adjusted_timestamp);
    }

    #[must_use]
    pub(crate) fn registers(&self) -> [u32; crate::register::NUM_REGISTERS] {
        self.core.registers()
    }

    #[must_use]
    pub(crate) fn pc(&self) -> u32 {
        self.core.pc()
    }

    #[must_use]
    pub(crate) fn clk(&self) -> u64 {
        self.core.clk()
    }

    /// Replays instructions until the program halts or `clk` reaches `clk_end`, constructing
    /// events for each one.
    ///
    /// # Errors
    ///
    /// Propagates any [`ExecutionError`] from executing an instruction.
    pub(crate) fn execute(&mut self) -> Result<CoreVMStatus, ExecutionError> {
        let status = loop {
            self.execute_instruction()?;
            if self.core.is_halted() {
                break CoreVMStatus::Done;
            }
            if self.core.clk() >= self.core.clk_end() {
                break CoreVMStatus::TraceEnd;
            }
        };
        self.record.cpu_local_memory_access.extend(self.local_memory_access.drain().map(|(_, v)| v));
        Ok(status)
    }

    /// Records `record`'s access to `addr` for `cpu_local_memory_access`'s per-shard first/
    /// last-touch bookkeeping, mirroring `Executor`'s identical `local_memory_access.entry(addr)`
    /// pattern (called from every `mr`/`mw`/`rr_traced`/`rw_traced`-equivalent site). `addr` is a
    /// register index for a register touch, a real memory address for RAM.
    fn touch_local(&mut self, addr: u32, record: &MemoryRecordEnum) {
        let initial = record.previous_record();
        let current = record.current_record();
        self.local_memory_access
            .entry(addr)
            .and_modify(|e| e.final_mem_access = current)
            .or_insert(MemoryLocalEvent { addr, initial_mem_access: initial, final_mem_access: current });
    }

    /// `touch_local` for a contiguous run of words starting at `base_ptr`, one per `records[i]`
    /// at address `base_ptr + 4*i` -- the word-indexed addressing every precompile's own
    /// `mr`/`mw`/`mr_slice`/`mw_slice` call uses (see `minimal/syscall.rs`'s dispatch functions,
    /// which this must match exactly).
    fn touch_local_slice<T: Copy>(
        &mut self,
        base_ptr: u32,
        records: &[T],
        wrap: impl Fn(T) -> MemoryRecordEnum,
    ) {
        for (i, &r) in records.iter().enumerate() {
            self.touch_local(base_ptr + 4 * i as u32, &wrap(r));
        }
    }

    /// Builds a precompile event's own `local_mem_access` (first/last touch per address, merging
    /// touches to the same address exactly like `touch_local` does), from one or more
    /// `(addr, record)` pairs given in the exact chronological order they were touched.
    /// Deliberately independent of `self.local_memory_access` (`touch_local`/`touch_local_slice`'s
    /// destination): `record.split()` always carves a precompile event's own private scratch RAM
    /// out into its own separate `ExecutionRecord` (never merged back into the record containing
    /// the surrounding CPU trace -- see `ExecutionRecord::split`'s precompile-event handling), so
    /// that event's own RAM touches must travel with it via its own field.
    ///
    /// Before building that isolated view, drains any *existing* entry for each of these
    /// addresses out of the shared map and pushes it directly into `cpu_local_memory_access`,
    /// closing it out as its own, self-contained event -- mirrors the legacy `Executor`'s
    /// `SyscallContext::postprocess` (`syscalls/context.rs`), which does the same drain-and-flush
    /// before a precompile's own reads/writes (which never touch the shared map at all -- see
    /// `Executor::mr`/`mw`'s `local_mem_access` parameter) get recorded. Without this, a regular
    /// instruction touching this address *before* the precompile call leaves a stale entry in the
    /// shared map; a later regular instruction touching it *after* the call then merges into that
    /// stale entry instead of starting fresh, producing one event spanning from the pre-call state
    /// straight to the post-call state with the precompile's own write invisible in between --
    /// breaking the `GlobalChip`-mediated bridge between this event and the split-off precompile
    /// event's own "final" value, since nothing in *this* record's own chain for the address
    /// matches it anymore.
    fn precompile_local_events(
        &mut self,
        touches: impl IntoIterator<Item = (u32, MemoryRecordEnum)>,
    ) -> Vec<MemoryLocalEvent> {
        let mut map: std::collections::HashMap<u32, MemoryLocalEvent> = Default::default();
        for (addr, record) in touches {
            if let Some(existing) = self.local_memory_access.remove(&addr) {
                self.record.cpu_local_memory_access.push(existing);
            }
            let initial = record.previous_record();
            let current = record.current_record();
            map.entry(addr)
                .and_modify(|e| e.final_mem_access = current)
                .or_insert(MemoryLocalEvent { addr, initial_mem_access: initial, final_mem_access: current });
        }
        map.into_values().collect()
    }

    fn execute_instruction(&mut self) -> Result<(), ExecutionError> {
        let instruction = self.core.program().fetch(self.core.pc());

        let pre_clk = self.core.clk();
        let pc = self.core.pc();
        let next_pc_before = self.core.next_pc();
        self.core.bump_clk();
        let post_clk = self.core.clk();
        if post_clk != pre_clk {
            self.record.bump_clk_high_events.push(BumpClkHighEvent {
                prev_clk: pre_clk,
                increment: post_clk - pre_clk,
                pc,
                next_pc: next_pc_before,
            });
        }

        let clk = self.core.clk();
        self.record.first_instruction_pc.get_or_insert(self.core.pc());
        self.record.first_instruction_clk.get_or_insert(clk);

        self.execute_operation(&instruction, clk)?;

        self.record.last_next_pc = self.core.pc();
        self.record.last_instruction_clk = clk;

        self.core.advance_clk();
        // Read back after `advance_clk` (rather than hardcoding `clk + 5`) since a `SYSCALL` with
        // nonzero `num_extra_cycles` (e.g. SHA-256 compress/extend) already bumped `self.core`'s
        // clk further inside `execute_syscall`.
        self.record.last_timestamp = self.core.clk();
        Ok(())
    }

    // ---- register-consistency record builders --------------------------------------------
    //
    // `op_a` is always tagged `MemoryAccessPosition::A`, `op_b` always `B`, `op_c` always `C`,
    // regardless of instruction type or read/write role (confirmed by reading every
    // `MemoryAccessPosition` call site in `executor.rs`'s `rr_cpu`/`rw_cpu` callers). These
    // helpers must be called in the same order the real register touch happens in: reads before
    // any write in the same instruction that could alias the same register (`alu_operands` before
    // `set_alu_dest`, etc.), or a read built *after* an aliased write would see the write's own
    // just-updated timestamp instead of the correct pre-instruction one.

    /// Builds a real read record for `op_a`, given its already-resolved value -- only a handful
    /// of instruction families (branches) ever *read* `op_a` rather than write it. Re-stamps
    /// `r`'s own consistency-timestamp afterward (same value, new timestamp) -- mirrors
    /// `Executor::rr_traced`'s unconditional `record.timestamp = timestamp`: a read participates
    /// in the same access-timestamp chain as a write, so a later access's `prev_timestamp` must
    /// match THIS read's own timestamp, not some earlier write's.
    fn read_op_a(&mut self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        let rec = MemoryRecordEnum::Read(MemoryReadRecord {
            value,
            timestamp: clk + MemoryAccessPosition::A as u64,
            prev_timestamp: self.core.reg_timestamp(r),
        });
        self.core.set_reg(r, value, MemoryAccessPosition::A);
        self.touch_local(r as u32, &rec);
        rec
    }

    /// Builds a real read record for `op_b`, given its already-resolved value. See `read_op_a`'s
    /// doc comment for why this also re-stamps `r`'s consistency-timestamp.
    fn read_op_b(&mut self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        let rec = MemoryRecordEnum::Read(MemoryReadRecord {
            value,
            timestamp: clk + MemoryAccessPosition::B as u64,
            prev_timestamp: self.core.reg_timestamp(r),
        });
        self.core.set_reg(r, value, MemoryAccessPosition::B);
        self.touch_local(r as u32, &rec);
        rec
    }

    /// Builds a real read record for `op_c`, given its already-resolved value. See `read_op_a`'s
    /// doc comment for why this also re-stamps `r`'s consistency-timestamp.
    fn read_op_c(&mut self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        let rec = MemoryRecordEnum::Read(MemoryReadRecord {
            value,
            timestamp: clk + MemoryAccessPosition::C as u64,
            prev_timestamp: self.core.reg_timestamp(r),
        });
        self.core.set_reg(r, value, MemoryAccessPosition::C);
        self.touch_local(r as u32, &rec);
        rec
    }

    /// Writes `value` to register `op_a` (`r`), returning the real write record for it.
    fn write_op_a(&mut self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        let prev_value = self.core.reg(r);
        let prev_timestamp = self.core.reg_timestamp(r);
        self.core.set_reg(r, value, MemoryAccessPosition::A);
        let rec = MemoryRecordEnum::Write(MemoryWriteRecord {
            value: self.core.reg(r),
            timestamp: clk + MemoryAccessPosition::A as u64,
            prev_value,
            prev_timestamp,
        });
        self.touch_local(r as u32, &rec);
        rec
    }

    /// Writes `value` to `Register::HI`, returning the real (non-`Option`) write record for it --
    /// `CompAluEvent`/`MiscEvent`'s `hi_record` field, unlike `a_record`/`b_record`/`c_record`,
    /// isn't wrapped in `MemoryRecordEnum`/`Option` since it's only ever a write.
    fn write_hi(&mut self, value: u32, clk: u64) -> MemoryWriteRecord {
        let prev_value = self.core.reg(Register::HI);
        let prev_timestamp = self.core.reg_timestamp(Register::HI);
        self.core.set_reg(Register::HI, value, MemoryAccessPosition::HI);
        let rec = MemoryWriteRecord {
            value: self.core.reg(Register::HI),
            timestamp: clk + MemoryAccessPosition::HI as u64,
            prev_value,
            prev_timestamp,
        };
        self.touch_local(Register::HI as u32, &MemoryRecordEnum::Write(rec));
        rec
    }

    /// Emits a `MemoryBumpChip` event if `record`'s own `clk_high` differs from that register's
    /// previous access -- i.e. a real access that would otherwise violate the cheap
    /// register-access scheme's `clk_high`-alignment invariant. Mirrors
    /// `Executor::emit_memory_bump_events`; only call this for opcodes whose chip has actually
    /// been migrated to that cheap scheme (see the call sites' own comments for exactly which).
    fn maybe_bump(&mut self, addr: Register, record: &MemoryRecordEnum) {
        let prev = record.previous_record();
        let current = record.current_record();
        if prev.timestamp >> 24 != current.timestamp >> 24 {
            self.record.bump_memory_events.push((
                MemoryRecordEnum::Read(MemoryReadRecord {
                    value: prev.value,
                    timestamp: (current.timestamp >> 24) << 24,
                    prev_timestamp: prev.timestamp,
                }),
                addr as u32,
            ));
        }
    }

    /// Emits `MemoryBumpChip` events for `a`/`b`/`c`'s records where present -- convenience
    /// wrapper for the instruction families whose entire register-operand set unconditionally
    /// uses the cheap scheme (memory load/store, branch, jump, misc, syscall).
    fn maybe_bump_abc(
        &mut self,
        a: Option<(Register, &MemoryRecordEnum)>,
        b: Option<(Register, &MemoryRecordEnum)>,
        c: Option<(Register, &MemoryRecordEnum)>,
    ) {
        if let Some((r, record)) = a {
            self.maybe_bump(r, record);
        }
        if let Some((r, record)) = b {
            self.maybe_bump(r, record);
        }
        if let Some((r, record)) = c {
            self.maybe_bump(r, record);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn execute_operation(&mut self, instruction: &Instruction, clk: u64) -> Result<(), ExecutionError> {
        let pc = self.core.pc();
        let next_pc_in = self.core.next_pc();
        let mut next_next_pc = self.core.next_pc().wrapping_add(4);
        self.core.set_next_is_delayslot(false);
        let op_a_is_zero = instruction.op_a == Register::ZERO as u8;

        if instruction.is_alu_instruction() {
            let (rd, b, b_record, c, c_record) = self.alu_operands(instruction, clk);
            let (a, hi) = crate::vm::alu_compute(instruction.opcode, b, c)?;
            self.emit_alu_event(
                clk, pc, next_pc_in, instruction, op_a_is_zero, rd, a, b, c, hi, b_record,
                c_record,
            );
        } else if instruction.is_memory_load_instruction() {
            self.execute_load(instruction, clk, pc, next_pc_in, op_a_is_zero)?;
        } else if instruction.is_memory_store_instruction() {
            self.execute_store(instruction, clk, pc, next_pc_in)?;
        } else if instruction.is_branch_instruction() {
            // `src1`/`src2` are read (never written) at positions A/B respectively -- mirrors
            // `Executor::branch_rr` exactly, including its B-then-A read order (matches
            // `MemoryAccessPosition`'s documented C-B-A ordering: if `rs == rt`, B's record must
            // chain from the pre-instruction state and A's record must chain from B's). The
            // offset (`op_c`) is always a raw immediate for this opcode family, never a register,
            // so there is no `c_record`.
            let rs: Register = instruction.op_a.into();
            let src1 = self.core.reg(rs);
            let (src2, b_record) = if instruction.opcode.only_one_operand() {
                (0, None)
            } else {
                let rt: Register = (instruction.op_b as u8).into();
                let src2 = self.core.reg(rt);
                (src2, Some(self.read_op_b(rt, src2, clk)))
            };
            let a_record = self.read_op_a(rs, src1, clk);
            let offset = instruction.op_c;
            // Every branch opcode's register operands use the cheap register-access scheme (see
            // `Executor::emit_memory_bump_events`'s call site comment), so this is unconditional.
            self.maybe_bump(rs, &a_record);
            if let Some(rec) = &b_record {
                let rt: Register = (instruction.op_b as u8).into();
                self.maybe_bump(rt, rec);
            }
            if crate::vm::branch_taken(instruction.opcode, src1, src2) {
                next_next_pc = crate::vm::branch_target(next_pc_in, offset);
            }
            self.core.set_next_is_delayslot(true);
            let mut event = BranchEvent::new(
                clk,
                pc,
                next_pc_in,
                next_next_pc,
                instruction.opcode,
                src1,
                src2,
                offset,
            );
            event.a_record = Some(a_record);
            event.b_record = b_record;
            self.record.branch_events.push(event);
        } else if instruction.is_jump_instruction() {
            // `Jump` (JR/JALR) reads its target register at position B (`Executor::jump_rr`
            // reads it *before* `link` is written, so the record is built from that same
            // pre-write read -- never re-read afterwards, which would see `link`'s just-written
            // value/timestamp if the two happen to alias). `Jumpi`/`JumpDirect`'s target is
            // always an immediate, so they have no `b_record`. All three write `link` at A.
            let link: Register = instruction.op_a.into();
            let (return_pc, target, b, b_record) = match instruction.opcode {
                Opcode::Jump => {
                    let target_reg: Register = (instruction.op_b as u8).into();
                    let target_pc = self.core.reg(target_reg);
                    let b_record = self.read_op_b(target_reg, target_pc, clk);
                    let (return_pc, target) = crate::vm::jump_jr_result(next_pc_in, target_pc);
                    (return_pc, target, target_pc, Some(b_record))
                }
                Opcode::Jumpi => {
                    let (return_pc, target) =
                        crate::vm::jump_jumpi_result(next_pc_in, instruction.op_b);
                    (return_pc, target, instruction.op_b, None)
                }
                Opcode::JumpDirect => {
                    let (return_pc, target) =
                        crate::vm::jump_direct_result(next_pc_in, instruction.op_b);
                    (return_pc, target, instruction.op_b, None)
                }
                _ => unreachable!("not a jump opcode: {:?}", instruction.opcode),
            };
            let a_record = self.write_op_a(link, return_pc, clk);
            // Every jump opcode's register operands use the cheap register-access scheme (see
            // `Executor::emit_memory_bump_events`'s call site comment), so this is unconditional.
            self.maybe_bump(link, &a_record);
            if let (Opcode::Jump, Some(rec)) = (instruction.opcode, &b_record) {
                let target_reg: Register = (instruction.op_b as u8).into();
                self.maybe_bump(target_reg, rec);
            }
            next_next_pc = target;
            self.core.set_next_is_delayslot(true);
            let mut event =
                JumpEvent::new(clk, pc, next_pc_in, next_next_pc, instruction.opcode, return_pc, b, 0);
            event.a_record = Some(a_record);
            event.b_record = b_record;
            match instruction.opcode {
                Opcode::Jump => self.record.jump_events.push(event),
                Opcode::Jumpi => self.record.jumpi_events.push(event),
                Opcode::JumpDirect => self.record.jumpdirect_events.push(event),
                _ => unreachable!(),
            }
        } else if instruction.is_mov_cond_instruction() {
            // `prev_a` (the "keep old value if condition false" input) is an untracked live peek
            // -- `Executor::execute_condmov` reads it via plain `self.register(rd)`, not
            // `rr_cpu`, so it carries no record of its own; the real `a_record` comes entirely
            // from the write below (which captures the same `prev_a` as its own `prev_value`).
            let rd: Register = instruction.op_a.into();
            let rs: Register = (instruction.op_b as u8).into();
            let rt: Register = (instruction.op_c as u8).into();
            let prev_a = self.core.reg(rd);
            // Must read C before B (mirrors `Executor::execute_condmov`'s exact order -- see
            // `alu_operands`'s identical comment on why this matters when `rs == rt`).
            let c = self.core.reg(rt);
            let c_record = self.read_op_c(rt, c, clk);
            let b = self.core.reg(rs);
            let b_record = self.read_op_b(rs, b, clk);
            let a = crate::vm::condmov_result(instruction.opcode, prev_a, b, c);
            let a_record = self.write_op_a(rd, a, clk);
            // MEQ/MNE/WSBH are dispatched through the legacy `Executor`'s misc branch, whose
            // register operands unconditionally use the cheap register-access scheme (see
            // `Executor::emit_memory_bump_events`'s call site comment).
            self.maybe_bump_abc(Some((rd, &a_record)), Some((rs, &b_record)), Some((rt, &c_record)));
            if op_a_is_zero {
                let mut event = AluEvent::new(pc, instruction.opcode, a, b, c);
                event.clk = clk;
                event.a_record = Some(a_record);
                event.b_record = Some(b_record);
                event.c_record = Some(c_record);
                self.record.alu_x0_events.push(event);
            } else {
                let mut event =
                    MovCondEvent::new(clk, pc, next_pc_in, instruction.opcode, a, b, c, prev_a);
                event.a_record = Some(a_record);
                event.b_record = Some(b_record);
                event.c_record = Some(c_record);
                self.record.movcond_events.push(event);
            }
        } else if instruction.is_misc_instruction() {
            self.execute_misc(instruction, clk, pc, next_pc_in, op_a_is_zero)?;
        } else if instruction.is_syscall_instruction() {
            let syscall_next_pc = self.execute_syscall(clk, pc)?;
            next_next_pc = syscall_next_pc.wrapping_add(4);
            self.core.set_pc(syscall_next_pc);
            self.core.set_next_pc(next_next_pc);
            return Ok(());
        } else {
            return Err(ExecutionError::UnsupportedInstruction(instruction.opcode as u32));
        }

        if next_next_pc == 0 {
            return Err(ExecutionError::NullPointerReference());
        }
        self.core.set_pc(next_pc_in);
        self.core.set_next_pc(next_next_pc);
        Ok(())
    }

    /// Resolves an ALU instruction's operands, along with real read records for `op_b`/`op_c`
    /// wherever they're actually registers (`None` for an immediate -- it has no backing
    /// register, so no consistency record applies). Must run before any write this same
    /// instruction performs (see the record-builder helpers' shared doc comment).
    #[allow(clippy::type_complexity)]
    fn alu_operands(
        &mut self,
        instruction: &Instruction,
        clk: u64,
    ) -> (Register, u32, Option<MemoryRecordEnum>, u32, Option<MemoryRecordEnum>) {
        if !instruction.imm_c {
            let rd = instruction.op_a.into();
            let b_reg: Register = (instruction.op_b as u8).into();
            let c_reg: Register = (instruction.op_c as u8).into();
            let b = self.core.reg(b_reg);
            let c = self.core.reg(c_reg);
            // Must read C before B (mirrors `Executor::alu_rr`'s exact order) -- if `b_reg ==
            // c_reg`, C's own record must chain from the pre-instruction state and B's record
            // must chain from C's, matching `MemoryAccessPosition`'s documented C-B-A ordering.
            let c_record = self.read_op_c(c_reg, c, clk);
            let b_record = self.read_op_b(b_reg, b, clk);
            (rd, b, Some(b_record), c, Some(c_record))
        } else if !instruction.imm_b {
            let rd = instruction.op_a.into();
            let b_reg: Register = (instruction.op_b as u8).into();
            let b = self.core.reg(b_reg);
            (rd, b, Some(self.read_op_b(b_reg, b, clk)), instruction.op_c, None)
        } else {
            (instruction.op_a.into(), instruction.op_b, None, instruction.op_c, None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_alu_event(
        &mut self,
        clk: u64,
        pc: u32,
        next_pc: u32,
        instruction: &Instruction,
        op_a_is_zero: bool,
        rd: Register,
        a: u32,
        b: u32,
        c: u32,
        hi: u32,
        b_record: Option<MemoryRecordEnum>,
        c_record: Option<MemoryRecordEnum>,
    ) {
        // Dual-result opcodes (MULT/MULTU/DIV/DIVU-family) write `LO` through the `A` slot and
        // `HI` through its own slot; everything else writes `rd` through `A` alone. Written
        // exactly once regardless of which of `event`/`event_comp` below actually gets pushed --
        // see `write_op_a`/`write_hi`'s doc comments on why record-building must happen at the
        // real write site, not be reconstructed from a stale post-write register read.
        let (a_record, hi_record, hi_record_is_real) = if instruction.opcode.is_use_lo_hi_alu() {
            let a_record = self.write_op_a(Register::LO, a, clk);
            let hi_record = self.write_hi(hi, clk);
            (a_record, hi_record, true)
        } else {
            let a_record = self.write_op_a(rd, a, clk);
            (a_record, MemoryWriteRecord::default(), false)
        };

        // Mirrors `Executor::emit_memory_bump_events`'s `uses_cheap_register_scheme` gating --
        // must match this function's own opcode routing exactly.
        if matches!(
            instruction.opcode,
            Opcode::ADD
                | Opcode::SUB
                | Opcode::SLT
                | Opcode::SLTU
                | Opcode::XOR
                | Opcode::OR
                | Opcode::AND
                | Opcode::NOR
                | Opcode::SRL
                | Opcode::SRA
                | Opcode::ROR
                | Opcode::SLL
                | Opcode::MUL
                | Opcode::MULT
                | Opcode::MULTU
                | Opcode::DIV
                | Opcode::DIVU
                | Opcode::MOD
                | Opcode::MODU
                | Opcode::CLZ
                | Opcode::CLO
        ) {
            // `HI` (written alongside `a_addr` above for `is_use_lo_hi_alu()` opcodes) is
            // deliberately never bumped here: `MulChip`/`DivRemChip`/`MaddsubChip` all write it
            // via the general-purpose `eval_memory_access` scheme (which handles an arbitrary
            // `clk_high` gap natively), never the cheap `RegisterAccessCols` scheme -- mirrors
            // `Executor::emit_memory_bump_events`'s `record: &MemoryAccessRecord`, which has no
            // `hi` slot at all. Bumping it anyway double-validates the same transition on the
            // shared memory argument and unbalances it (see that function's own doc comment).
            let a_addr = if instruction.opcode.is_use_lo_hi_alu() { Register::LO } else { rd };
            self.maybe_bump(a_addr, &a_record);
            if let Some(rec) = &b_record {
                let b_reg: Register = (instruction.op_b as u8).into();
                self.maybe_bump(b_reg, rec);
            }
            if let Some(rec) = &c_record {
                let c_reg: Register = (instruction.op_c as u8).into();
                self.maybe_bump(c_reg, rec);
            }
        }

        let event = AluEvent {
            clk,
            pc,
            next_pc,
            opcode: instruction.opcode,
            hi,
            a,
            b,
            c,
            a_record: Some(a_record),
            b_record,
            c_record,
        };
        let mut event_comp = CompAluEvent::new_with_hi(pc, instruction.opcode, a, b, c, hi);
        event_comp.clk = clk;
        event_comp.next_pc = next_pc;
        event_comp.hi_record_is_real = hi_record_is_real;
        event_comp.hi_record = hi_record;
        event_comp.a_record = Some(a_record);
        event_comp.b_record = b_record;
        event_comp.c_record = c_record;
        let imm_b = instruction.imm_b;
        match instruction.opcode {
            // Register+immediate form (`imm_c && !imm_b`) is `Addi`; fully-immediate
            // (`imm_b && imm_c`) is `AddNoop`; register-register (`!imm_c`) falls through to `Add`.
            Opcode::ADD if instruction.imm_c && !imm_b => self.record.addi_events.push(event),
            Opcode::ADD if imm_b => self.record.add_noop_events.push(event),
            Opcode::ADD if op_a_is_zero => self.record.alu_x0_events.push(event),
            Opcode::ADD => self.record.add_events.push(event),
            Opcode::SUB if op_a_is_zero => self.record.alu_x0_events.push(event),
            Opcode::SUB => self.record.sub_events.push(event),
            Opcode::XOR | Opcode::OR | Opcode::AND | Opcode::NOR if op_a_is_zero => {
                self.record.alu_x0_events.push(event);
            }
            Opcode::XOR | Opcode::OR | Opcode::AND | Opcode::NOR => self.record.bitwise_events.push(event),
            Opcode::SLL if imm_b => self.record.lui_events.push(event),
            Opcode::SLL if op_a_is_zero => self.record.alu_x0_events.push(event),
            Opcode::SLL => self.record.shift_left_events.push(event),
            Opcode::SRL | Opcode::SRA | Opcode::ROR if op_a_is_zero => {
                self.record.alu_x0_events.push(event);
            }
            Opcode::SRL | Opcode::SRA | Opcode::ROR => self.record.shift_right_events.push(event),
            Opcode::SLT | Opcode::SLTU if instruction.imm_c => self.record.slti_events.push(event),
            Opcode::SLT | Opcode::SLTU if op_a_is_zero => self.record.alu_x0_events.push(event),
            Opcode::SLT | Opcode::SLTU => self.record.lt_events.push(event),
            Opcode::MUL if op_a_is_zero => self.record.alu_x0_events.push(event),
            Opcode::MUL | Opcode::MULT | Opcode::MULTU => self.record.mul_events.push(event_comp),
            Opcode::MOD | Opcode::MODU if op_a_is_zero => self.record.alu_x0_events.push(event),
            Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU => self.record.divrem_events.push(event_comp),
            Opcode::CLZ | Opcode::CLO if op_a_is_zero => self.record.alu_x0_events.push(event),
            Opcode::CLZ | Opcode::CLO => self.record.cloclz_events.push(event),
            _ => {}
        }
    }

    fn execute_load(
        &mut self,
        instruction: &Instruction,
        clk: u64,
        pc: u32,
        next_pc: u32,
        op_a_is_zero: bool,
    ) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs_raw = self.core.reg(rs_reg);
        let b_record = self.read_op_b(rs_reg, rs_raw, clk);
        // `rt`'s current value is an untracked live peek (only needed for LWL/LWR's byte-merge)
        // -- matches `Executor::execute_load`'s identical comment: the real `a_record` comes
        // entirely from the write below, which captures this same value as its own `prev_value`.
        let rt = self.core.reg(rt_reg);

        let addr = rs_raw.wrapping_add(offset);
        let mem_entry = self.core.next_oracle_entry();
        let mem = mem_entry.value;
        let rs = addr;

        let val = match instruction.opcode {
            Opcode::LH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LH, addr));
                }
                sign_extend::<16>((mem >> ((rs & 2) * 8)) & 0xffff)
            }
            Opcode::LWL => {
                let i = rs & 3;
                let val = mem << (24 - i * 8);
                let mask: u32 = 0xFFFF_FFFF_u32 << (24 - i * 8);
                (rt & (!mask)) | val
            }
            Opcode::LW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LW, addr));
                }
                mem
            }
            Opcode::LBU => (mem >> ((rs & 3) * 8)) & 0xff,
            Opcode::LHU => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LHU, addr));
                }
                (mem >> ((rs & 2) * 8)) & 0xffff
            }
            Opcode::LWR => {
                let i = rs & 3;
                let val = mem >> (i * 8);
                let mask = 0xFFFF_FFFF_u32 >> (i * 8);
                (rt & (!mask)) | val
            }
            Opcode::LL => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LL, addr));
                }
                mem
            }
            Opcode::LB => sign_extend::<8>((mem >> ((rs & 3) * 8)) & 0xff),
            _ => unreachable!("not a load opcode: {:?}", instruction.opcode),
        };
        let a_record = self.write_op_a(rt_reg, val, clk);
        // Every memory-load opcode's register operands use the cheap register-access scheme
        // (see `Executor::emit_memory_bump_events`'s call site comment), so this is unconditional.
        self.maybe_bump_abc(Some((rt_reg, &a_record)), Some((rs_reg, &b_record)), None);

        let mem_record = MemoryRecordEnum::Read(MemoryReadRecord {
            value: mem,
            timestamp: clk,
            prev_timestamp: mem_entry.clk,
        });
        // `touch_local` must key on the actual RAM word (word-aligned, matching every `mr`/`mw`
        // call's addressing scheme), not the instruction's own possibly-unaligned byte/halfword
        // address -- otherwise a byte/halfword load/store splits one real memory word's local-
        // access chain into up to four spurious per-byte-offset addresses.
        self.touch_local(addr & 0xFFFF_FFFC, &mem_record);
        let mut event =
            MemInstrEvent::new(clk, pc, next_pc, instruction.opcode, val, rs_raw, offset, mem_record, rt);
        event.a_record = Some(a_record);
        event.b_record = Some(b_record);
        match instruction.opcode {
            Opcode::LW | Opcode::LL if op_a_is_zero => self.record.load_x0_events.push(event),
            Opcode::LW | Opcode::LL => self.record.load_word_events.push(event),
            Opcode::LB | Opcode::LBU => self.record.load_byte_events.push(event),
            Opcode::LH | Opcode::LHU => self.record.load_half_events.push(event),
            Opcode::LWL | Opcode::LWR => self.record.load_word_unaligned_events.push(event),
            _ => unreachable!("not a load opcode: {:?}", instruction.opcode),
        }
        Ok(())
    }

    fn execute_store(
        &mut self,
        instruction: &Instruction,
        clk: u64,
        pc: u32,
        next_pc: u32,
    ) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs = self.core.reg(rs_reg);
        let b_record = self.read_op_b(rs_reg, rs, clk);
        // `SC`'s `rt` (the value about to be stored) is an untracked live peek -- unlike every
        // other store, which reads it as a real record at A (see the real `a_record` built after
        // `val` below: SC's own A-slot is a *write* of the success flag, not a read of `rt`).
        let rt = self.core.reg(rt_reg);

        let addr = rs.wrapping_add(offset);
        let mem_entry = self.core.next_oracle_entry();
        let mem = mem_entry.value;

        let val = match instruction.opcode {
            Opcode::SB => {
                let i = addr & 3;
                let val = (rt & 0xff) << (i * 8);
                let mask = 0xFFFF_FFFF_u32 ^ (0xff << (i * 8));
                (mem & mask) | val
            }
            Opcode::SH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SH, addr));
                }
                let i = addr & 2;
                let val = (rt & 0xffff) << (i * 8);
                let mask = 0xFFFF_FFFF_u32 ^ (0xffff << (i * 8));
                (mem & mask) | val
            }
            Opcode::SWL => {
                let i = addr & 3;
                let val = rt >> (24 - i * 8);
                let mask = 0xFFFF_FFFF_u32 >> (24 - i * 8);
                (mem & (!mask)) | val
            }
            Opcode::SW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SW, addr));
                }
                rt
            }
            Opcode::SWR => {
                let i = addr & 3;
                let val = rt << (i * 8);
                let mask = 0xFFFF_FFFF_u32 << (i * 8);
                (mem & (!mask)) | val
            }
            Opcode::SC => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SC, addr));
                }
                rt
            }
            _ => unreachable!("not a store opcode: {:?}", instruction.opcode),
        };

        let (a, a_record) = if instruction.opcode == Opcode::SC {
            (1, self.write_op_a(rt_reg, 1, clk))
        } else {
            (rt, self.read_op_a(rt_reg, rt, clk))
        };
        // Every memory-store opcode's register operands use the cheap register-access scheme
        // (see `Executor::emit_memory_bump_events`'s call site comment), so this is unconditional.
        self.maybe_bump_abc(Some((rt_reg, &a_record)), Some((rs_reg, &b_record)), None);
        let mem_record = MemoryRecordEnum::Write(MemoryWriteRecord {
            value: val,
            timestamp: clk,
            prev_value: mem,
            prev_timestamp: mem_entry.clk,
        });
        // See `execute_load`'s identical comment: `touch_local` must key on the word-aligned RAM
        // address, not the instruction's own possibly-unaligned byte/halfword address.
        self.touch_local(addr & 0xFFFF_FFFC, &mem_record);
        let mut event =
            MemInstrEvent::new(clk, pc, next_pc, instruction.opcode, a, rs, offset, mem_record, rt);
        event.a_record = Some(a_record);
        event.b_record = Some(b_record);
        match instruction.opcode {
            Opcode::SW => self.record.store_word_events.push(event),
            Opcode::SB => self.record.store_byte_events.push(event),
            Opcode::SH => self.record.store_half_events.push(event),
            Opcode::SWL | Opcode::SWR => self.record.store_word_unaligned_events.push(event),
            Opcode::SC => self.record.store_conditional_events.push(event),
            _ => unreachable!("not a store opcode: {:?}", instruction.opcode),
        }
        Ok(())
    }

    fn execute_misc(
        &mut self,
        instruction: &Instruction,
        clk: u64,
        pc: u32,
        next_pc: u32,
        op_a_is_zero: bool,
    ) -> Result<(), ExecutionError> {
        if instruction.opcode == Opcode::WSBH {
            let rd: Register = instruction.op_a.into();
            let rt: Register = (instruction.op_b as u8).into();
            let b = self.core.reg(rt);
            let b_record = self.read_op_b(rt, b, clk);
            let a = crate::vm::wsbh(b);
            let a_record = self.write_op_a(rd, a, clk);
            self.maybe_bump_abc(Some((rd, &a_record)), Some((rt, &b_record)), None);
            if op_a_is_zero {
                let mut event = AluEvent::new(pc, instruction.opcode, a, b, 0);
                event.clk = clk;
                event.a_record = Some(a_record);
                event.b_record = Some(b_record);
                self.record.alu_x0_events.push(event);
            } else {
                let mut event =
                    MovCondEvent::new(clk, pc, next_pc, instruction.opcode, a, b, 0, 0);
                event.a_record = Some(a_record);
                event.b_record = Some(b_record);
                self.record.movcond_events.push(event);
            }
            return Ok(());
        }

        let rd: Register = instruction.op_a.into();
        let rt: Register = (instruction.op_b as u8).into();
        let c = instruction.op_c;
        match instruction.opcode {
            Opcode::SEXT => {
                let b = self.core.reg(rt);
                let b_record = self.read_op_b(rt, b, clk);
                let a = crate::vm::sext(b, c);
                let a_record = self.write_op_a(rd, a, clk);
                self.maybe_bump_abc(Some((rd, &a_record)), Some((rt, &b_record)), None);
                self.push_misc_or_x0(op_a_is_zero, clk, pc, next_pc, instruction.opcode, a, b, c, 0, a_record, Some(b_record));
            }
            Opcode::EXT => {
                let b = self.core.reg(rt);
                let b_record = self.read_op_b(rt, b, clk);
                let a = crate::vm::ext(b, c)?;
                let a_record = self.write_op_a(rd, a, clk);
                self.maybe_bump_abc(Some((rd, &a_record)), Some((rt, &b_record)), None);
                self.push_misc_or_x0(op_a_is_zero, clk, pc, next_pc, instruction.opcode, a, b, c, 0, a_record, Some(b_record));
            }
            Opcode::INS => {
                let b = self.core.reg(rt);
                let b_record = self.read_op_b(rt, b, clk);
                // `prev_a` (rd's current value, merged with `b`) is an untracked live peek --
                // matches `MOVCOND`'s identical pattern (see its doc comment): the real
                // `a_record` comes from the write below, which captures this same value as its
                // own `prev_value`.
                let prev_a = self.core.reg(rd);
                let a = crate::vm::ins(prev_a, b, c)?;
                let a_record = self.write_op_a(rd, a, clk);
                self.maybe_bump_abc(Some((rd, &a_record)), Some((rt, &b_record)), None);
                self.push_misc_or_x0(op_a_is_zero, clk, pc, next_pc, instruction.opcode, a, b, c, prev_a, a_record, Some(b_record));
            }
            Opcode::TEQ => {
                let rs: Register = instruction.op_a.into();
                let rt: Register = (instruction.op_b as u8).into();
                let src2 = self.core.reg(rt);
                let src2_record = self.read_op_b(rt, src2, clk);
                let src1 = self.core.reg(rs);
                let src1_record = self.read_op_a(rs, src1, clk);
                self.maybe_bump_abc(Some((rs, &src1_record)), Some((rt, &src2_record)), None);
                crate::vm::teq(src1, src2)?;
                let mut event = MiscEvent::new(clk, pc, next_pc, instruction.opcode, src1, src2, 0, 0, MemoryWriteRecord::default());
                event.a_record = Some(src1_record);
                event.b_record = Some(src2_record);
                self.record.teq_events.push(event);
            }
            Opcode::MADDU | Opcode::MSUBU | Opcode::MADD | Opcode::MSUB => {
                let lo_reg: Register = instruction.op_a.into();
                let rs: Register = (instruction.op_c as u8).into();
                let c_val = self.core.reg(rs);
                let c_record = self.read_op_c(rs, c_val, clk);
                let b = self.core.reg(rt);
                let b_record = self.read_op_b(rt, b, clk);
                let lo = self.core.reg(Register::LO);
                let hi = self.core.reg(Register::HI);
                let (out_lo, out_hi) = match instruction.opcode {
                    Opcode::MADDU => crate::vm::maddu(b, c_val, lo, hi),
                    Opcode::MSUBU => crate::vm::msubu(b, c_val, lo, hi),
                    Opcode::MADD => crate::vm::madd(b, c_val, lo, hi),
                    Opcode::MSUB => crate::vm::msub(b, c_val, lo, hi),
                    _ => unreachable!(),
                };
                let a_record = self.write_op_a(lo_reg, out_lo, clk);
                // `HI` deliberately not bumped -- see the identical MULT/DIV-family comment above.
                let hi_record = self.write_hi(out_hi, clk);
                self.maybe_bump_abc(Some((lo_reg, &a_record)), Some((rt, &b_record)), Some((rs, &c_record)));
                let mut event = MiscEvent::new(
                    clk,
                    pc,
                    next_pc,
                    instruction.opcode,
                    out_lo,
                    b,
                    c_val,
                    lo,
                    hi_record,
                );
                event.a_record = Some(a_record);
                event.b_record = Some(b_record);
                event.c_record = Some(c_record);
                self.record.maddsub_events.push(event);
            }
            _ => unreachable!("not a misc opcode: {:?}", instruction.opcode),
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn push_misc_or_x0(
        &mut self,
        op_a_is_zero: bool,
        clk: u64,
        pc: u32,
        next_pc: u32,
        opcode: Opcode,
        a: u32,
        b: u32,
        c: u32,
        prev_a: u32,
        a_record: MemoryRecordEnum,
        b_record: Option<MemoryRecordEnum>,
    ) {
        if op_a_is_zero {
            let mut event = AluEvent::new(pc, opcode, a, b, c);
            event.clk = clk;
            event.a_record = Some(a_record);
            event.b_record = b_record;
            self.record.alu_x0_events.push(event);
            return;
        }
        let mut event = MiscEvent::new(clk, pc, next_pc, opcode, a, b, c, prev_a, MemoryWriteRecord::default());
        event.a_record = Some(a_record);
        event.b_record = b_record;
        match opcode {
            Opcode::SEXT => self.record.sext_events.push(event),
            Opcode::INS => self.record.ins_events.push(event),
            Opcode::EXT => self.record.ext_events.push(event),
            _ => unreachable!("not a sext/ins/ext opcode: {opcode:?}"),
        }
    }

    /// Builds an `EllipticCurveAddEvent`: pops `q`'s reads, then `p`'s write-preimage (the value
    /// actually needed as an input, unlike a discarded write pop -- see
    /// `minimal/syscall.rs::ec_add_dispatch`'s doc comment for why the oracle order is q-then-p
    /// even though `p` is conceptually read first). `p`'s writes are timestamped `clk + 1`, not
    /// `clk`, matching `ec_add_dispatch`'s own `self.clk += 1` between reading `q` and writing
    /// `p` -- the real `MemoryRecord.timestamp` a later access chains from is `clk + 1`.
    fn ec_add_event<E: EllipticCurve>(&mut self, clk: u64, p_ptr: u32, q_ptr: u32) -> EllipticCurveAddEvent {
        let num_words = ec_num_words::<E>();
        let q_entries = self.core.next_oracle_entries(num_words);
        let q: Vec<u32> = q_entries.iter().map(|e| e.value).collect();
        let q_memory_records: Vec<MemoryReadRecord> = q_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let p_entries = self.core.next_oracle_entries(num_words);
        let p: Vec<u32> = p_entries.iter().map(|e| e.value).collect();
        let result = ec_add::<E>(&p, &q);
        let p_memory_records: Vec<MemoryWriteRecord> = result
            .iter()
            .zip(&p_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk + 1, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            q_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (q_ptr + 4 * i as u32, MemoryRecordEnum::Read(r)))
                .chain(
                    p_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (p_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                ),
        );
        EllipticCurveAddEvent { shard: 0, clk, p_ptr, p, q_ptr, q, p_memory_records, q_memory_records, local_mem_access }
    }

    /// Builds an `EllipticCurveDoubleEvent`: pops `p`'s write-preimage (the value actually needed
    /// as an input).
    fn ec_double_event<E: EllipticCurve>(&mut self, clk: u64, p_ptr: u32) -> EllipticCurveDoubleEvent {
        let num_words = ec_num_words::<E>();
        let p_entries = self.core.next_oracle_entries(num_words);
        let p: Vec<u32> = p_entries.iter().map(|e| e.value).collect();
        let result = ec_double::<E>(&p);
        let p_memory_records: Vec<MemoryWriteRecord> = result
            .iter()
            .zip(&p_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            p_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (p_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
        );
        EllipticCurveDoubleEvent { shard: 0, clk, p_ptr, p, p_memory_records, local_mem_access }
    }

    /// Builds an `EllipticCurveDecompressEvent`: pops `x`'s reads (used), then `y`'s
    /// write-preimage (discarded -- the old `y` value isn't an input to computing the new one).
    fn ec_decompress_event<E: EllipticCurve>(
        &mut self,
        clk: u64,
        ptr: u32,
        sign_bit: u32,
    ) -> Result<EllipticCurveDecompressEvent, ExecutionError> {
        let num_words_field_element = ec_num_limb_words::<E>();
        let x_entries = self.core.next_oracle_entries(num_words_field_element);
        let x: Vec<u32> = x_entries.iter().map(|e| e.value).collect();
        let x_memory_records: Vec<MemoryReadRecord> = x_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let x_bytes = zkm_primitives::consts::words_to_bytes_le_vec(&x);
        let mut x_bytes_be = x_bytes.clone();
        x_bytes_be.reverse();
        let decompressed_y_bytes =
            ec_decompress::<E>(&x_bytes_be, sign_bit).map_err(ExecutionError::CurveError)?;
        let y_words = zkm_primitives::consts::bytes_to_words_le_vec(&decompressed_y_bytes);
        let y_entries = self.core.next_oracle_entries(y_words.len()); // write preimages; see SHA_COMPRESS.
        let y_memory_records: Vec<MemoryWriteRecord> = y_words
            .iter()
            .zip(&y_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            x_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (ptr + (num_words_field_element * 4) as u32 + 4 * i as u32, MemoryRecordEnum::Read(r)))
                .chain(
                    y_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                ),
        );
        Ok(EllipticCurveDecompressEvent {
            shard: 0,
            clk,
            ptr,
            sign_bit: sign_bit != 0,
            x_bytes,
            decompressed_y_bytes,
            x_memory_records,
            y_memory_records,
            local_mem_access,
        })
    }

    /// Builds an `EdDecompressEvent`: pops `y`'s reads (used), then `x`'s write-preimage
    /// (discarded).
    fn ed_decompress_event(
        &mut self,
        clk: u64,
        ptr: u32,
        sign: u32,
    ) -> Result<EdDecompressEvent, ExecutionError> {
        let y_entries = self.core.next_oracle_entries(WORDS_FIELD_ELEMENT);
        let y: Vec<u32> = y_entries.iter().map(|e| e.value).collect();
        let y_memory_records: [MemoryReadRecord; WORDS_FIELD_ELEMENT] = y_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let y_bytes: [u8; COMPRESSED_POINT_BYTES] =
            zkm_primitives::consts::words_to_bytes_le_vec(&y).try_into().unwrap();
        let decompressed_x_bytes =
            ed25519_decompress(y_bytes, sign).map_err(ExecutionError::CurveError)?;
        let x_words = zkm_primitives::consts::bytes_to_words_le_vec(&decompressed_x_bytes);
        let x_entries = self.core.next_oracle_entries(x_words.len()); // write preimages; see SHA_COMPRESS.
        let x_memory_records: [MemoryWriteRecord; WORDS_FIELD_ELEMENT] = x_words
            .iter()
            .zip(&x_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let local_mem_access = self.precompile_local_events(
            y_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (ptr + COMPRESSED_POINT_BYTES as u32 + 4 * i as u32, MemoryRecordEnum::Read(r)))
                .chain(
                    x_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                ),
        );
        Ok(EdDecompressEvent {
            shard: 0,
            clk,
            ptr,
            sign: sign != 0,
            y_bytes,
            decompressed_x_bytes,
            x_memory_records,
            y_memory_records,
            local_mem_access,
        })
    }

    /// Builds an `FpOpEvent`: pops `y`'s reads, then `x`'s write-preimage (the value actually
    /// needed as an input -- see `ec_add_event`'s doc comment for why the oracle order is
    /// y-then-x). `x`'s writes are timestamped `clk + 1` -- see `ec_add_event`'s identical
    /// comment on why (`fp_dispatch` has the same mid-dispatch `self.clk += 1`).
    fn fp_op_event<P: FpOpField>(
        &mut self,
        clk: u64,
        x_ptr: u32,
        y_ptr: u32,
        op: FieldOperation,
    ) -> FpOpEvent {
        let num_words = fp_num_words::<P>();
        let y_entries = self.core.next_oracle_entries(num_words);
        let y: Vec<u32> = y_entries.iter().map(|e| e.value).collect();
        let y_memory_records: Vec<MemoryReadRecord> = y_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let x_entries = self.core.next_oracle_entries(num_words);
        let x: Vec<u32> = x_entries.iter().map(|e| e.value).collect();
        let result = fp_op::<P>(&x, &y, op);
        let x_memory_records: Vec<MemoryWriteRecord> = result
            .iter()
            .zip(&x_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk + 1, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            y_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (y_ptr + 4 * i as u32, MemoryRecordEnum::Read(r)))
                .chain(
                    x_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (x_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                ),
        );
        FpOpEvent { shard: 0, clk, x_ptr, x, y_ptr, y, op, x_memory_records, y_memory_records, local_mem_access }
    }

    /// Builds an `Fp2AddSubEvent`: pops `y`'s reads, then `x`'s write-preimage. `x`'s writes are
    /// timestamped `clk + 1` -- see `ec_add_event`'s identical comment on why (`fp2_addsub_dispatch`
    /// has the same mid-dispatch `self.clk += 1`).
    fn fp2_addsub_event<P: FpOpField>(
        &mut self,
        clk: u64,
        x_ptr: u32,
        y_ptr: u32,
        op: FieldOperation,
    ) -> Fp2AddSubEvent {
        let num_words = fp2_num_words::<P>();
        let y_entries = self.core.next_oracle_entries(num_words);
        let y: Vec<u32> = y_entries.iter().map(|e| e.value).collect();
        let y_memory_records: Vec<MemoryReadRecord> = y_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let x_entries = self.core.next_oracle_entries(num_words);
        let x: Vec<u32> = x_entries.iter().map(|e| e.value).collect();
        let result = fp2_addsub::<P>(&x, &y, op);
        let x_memory_records: Vec<MemoryWriteRecord> = result
            .iter()
            .zip(&x_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk + 1, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            y_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (y_ptr + 4 * i as u32, MemoryRecordEnum::Read(r)))
                .chain(
                    x_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (x_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                ),
        );
        Fp2AddSubEvent { shard: 0, clk, op, x_ptr, x, y_ptr, y, x_memory_records, y_memory_records, local_mem_access }
    }

    /// Builds an `Fp2MulEvent`: pops `y`'s reads, then `x`'s write-preimage. `x`'s writes are
    /// timestamped `clk + 1` -- see `ec_add_event`'s identical comment on why (`fp2_mul_dispatch`
    /// has the same mid-dispatch `self.clk += 1`).
    fn fp2_mul_event<P: FpOpField>(&mut self, clk: u64, x_ptr: u32, y_ptr: u32) -> Fp2MulEvent {
        let num_words = fp2_num_words::<P>();
        let y_entries = self.core.next_oracle_entries(num_words);
        let y: Vec<u32> = y_entries.iter().map(|e| e.value).collect();
        let y_memory_records: Vec<MemoryReadRecord> = y_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let x_entries = self.core.next_oracle_entries(num_words);
        let x: Vec<u32> = x_entries.iter().map(|e| e.value).collect();
        let result = fp2_mul::<P>(&x, &y);
        let x_memory_records: Vec<MemoryWriteRecord> = result
            .iter()
            .zip(&x_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk + 1, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            y_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (y_ptr + 4 * i as u32, MemoryRecordEnum::Read(r)))
                .chain(
                    x_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (x_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                ),
        );
        Fp2MulEvent { shard: 0, clk, x_ptr, x, y_ptr, y, x_memory_records, y_memory_records, local_mem_access }
    }

    /// Builds a `Uint256MulEvent`: pops `y`'s reads, `modulus`'s reads, then `x`'s write-preimage
    /// (the value actually needed as an input). `x`'s writes are timestamped `clk + 1` -- see
    /// `ec_add_event`'s identical comment on why (`uint256_mul_dispatch` has the same mid-dispatch
    /// `self.clk += 1`).
    fn uint256_mul_event(&mut self, clk: u64, x_ptr: u32, y_ptr: u32) -> Uint256MulEvent {
        let y_entries = self.core.next_oracle_entries(8);
        let y: [u32; 8] = y_entries.iter().map(|e| e.value).collect::<Vec<_>>().try_into().unwrap();
        let y_memory_records: Vec<MemoryReadRecord> = y_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let modulus_ptr = y_ptr + 8 * 4;
        let modulus_entries = self.core.next_oracle_entries(8);
        let modulus: [u32; 8] =
            modulus_entries.iter().map(|e| e.value).collect::<Vec<_>>().try_into().unwrap();
        let modulus_memory_records: Vec<MemoryReadRecord> = modulus_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let x_entries = self.core.next_oracle_entries(8);
        let x: [u32; 8] = x_entries.iter().map(|e| e.value).collect::<Vec<_>>().try_into().unwrap();
        let result = uint256_mul(&x, &y, &modulus);
        let x_memory_records: Vec<MemoryWriteRecord> = result
            .iter()
            .zip(&x_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk + 1, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            y_memory_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (y_ptr + 4 * i as u32, MemoryRecordEnum::Read(r)))
                .chain(
                    modulus_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (modulus_ptr + 4 * i as u32, MemoryRecordEnum::Read(r))),
                )
                .chain(
                    x_memory_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (x_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                ),
        );
        Uint256MulEvent {
            shard: 0,
            clk,
            x_ptr,
            x: x.to_vec(),
            y_ptr,
            y: y.to_vec(),
            modulus: modulus.to_vec(),
            x_memory_records,
            y_memory_records,
            modulus_memory_records,
            local_mem_access,
        }
    }

    /// Builds a `U256xU2048MulEvent`: reads `$a2`/`$a3` (live, unlogged) for `lo_ptr`/`hi_ptr`,
    /// pops `a`'s reads, `b`'s reads, then `lo`'s and `hi`'s write-preimages (discarded). `lo`'s
    /// and `hi`'s writes are timestamped `clk + 1` -- see `ec_add_event`'s identical comment on
    /// why (`u256xu2048_mul_dispatch` has the same mid-dispatch `self.clk += 1`).
    fn u256xu2048_mul_event(&mut self, clk: u64, a_ptr: u32, b_ptr: u32) -> U256xU2048MulEvent {
        let lo_ptr = self.core.reg(Register::A2);
        let hi_ptr = self.core.reg(Register::A3);
        let lo_ptr_memory = MemoryReadRecord {
            value: lo_ptr,
            timestamp: clk,
            prev_timestamp: self.core.reg_timestamp(Register::A2),
        };
        let hi_ptr_memory = MemoryReadRecord {
            value: hi_ptr,
            timestamp: clk,
            prev_timestamp: self.core.reg_timestamp(Register::A3),
        };
        self.core.set_reg_aux(Register::A2, lo_ptr);
        self.core.set_reg_aux(Register::A3, hi_ptr);
        // `U256x2048MulChip`'s AIR (`.../u256x2048_mul/air.rs`) evaluates its own
        // `eval_memory_access` for `lo_ptr_memory`/`hi_ptr_memory` (the `A2`/`A3` pointer-argument
        // register reads) from within the chip itself, which -- like the rest of this event --
        // gets carved into its own separate `ExecutionRecord` by `ExecutionRecord::split`'s
        // precompile-event handling. So this touch must travel with the event via
        // `local_mem_access` (`precompile_local_events`'s isolation), not the shared
        // `touch_local`/`self.local_memory_access` map: routing it through the shared map leaves
        // the chip that actually emits the interaction (now in a different record) with no
        // matching bracket, since nothing else in the surrounding CPU record's own chain for `A2`/
        // `A3` references this specific touch anymore.
        let a_entries = self.core.next_oracle_entries(U256_NUM_WORDS);
        let a: [u32; U256_NUM_WORDS] =
            a_entries.iter().map(|e| e.value).collect::<Vec<_>>().try_into().unwrap();
        let a_memory_records: Vec<MemoryReadRecord> = a_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
        let b_entries = self.core.next_oracle_entries(U2048_NUM_WORDS);
        let b: [u32; U2048_NUM_WORDS] =
            b_entries.iter().map(|e| e.value).collect::<Vec<_>>().try_into().unwrap();
        let b_memory_records: Vec<MemoryReadRecord> = b_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();

        let (lo, hi) = u256xu2048_mul(&a, &b);
        let lo_entries = self.core.next_oracle_entries(lo.len());
        let lo_memory_records: Vec<MemoryWriteRecord> = lo
            .iter()
            .zip(&lo_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk + 1, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let hi_entries = self.core.next_oracle_entries(hi.len());
        let hi_memory_records: Vec<MemoryWriteRecord> = hi
            .iter()
            .zip(&hi_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk + 1, prev_value: e.value, prev_timestamp: e.clk })
            .collect();

        let local_mem_access = self.precompile_local_events(
            [
                (Register::A2 as u32, MemoryRecordEnum::Read(lo_ptr_memory)),
                (Register::A3 as u32, MemoryRecordEnum::Read(hi_ptr_memory)),
            ]
            .into_iter()
            .chain(
                a_memory_records
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| (a_ptr + 4 * i as u32, MemoryRecordEnum::Read(r))),
            )
            .chain(
                b_memory_records
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| (b_ptr + 4 * i as u32, MemoryRecordEnum::Read(r))),
            )
            .chain(
                lo_memory_records
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| (lo_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
            )
            .chain(
                hi_memory_records
                    .iter()
                    .enumerate()
                    .map(|(i, &r)| (hi_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
            ),
        );

        U256xU2048MulEvent {
            shard: 0,
            clk,
            a_ptr,
            a: a.to_vec(),
            b_ptr,
            b: b.to_vec(),
            lo_ptr,
            lo_ptr_memory,
            lo: lo.to_vec(),
            hi_ptr,
            hi_ptr_memory,
            hi: hi.to_vec(),
            a_memory_records,
            b_memory_records,
            lo_memory_records,
            hi_memory_records,
            local_mem_access,
        }
    }

    /// Builds a `Poseidon2PermuteEvent`: pops the state's write-preimage (the value actually
    /// needed as an input).
    fn poseidon2_permute_event(&mut self, clk: u64, state_ptr: u32) -> Poseidon2PermuteEvent {
        let pre_state_entries = self.core.next_oracle_entries(POSEIDON2_STATE_SIZE);
        let pre_state: [u32; POSEIDON2_STATE_SIZE] =
            pre_state_entries.iter().map(|e| e.value).collect::<Vec<_>>().try_into().unwrap();
        let post_state = poseidon2_permute(pre_state);
        let state_records: Vec<MemoryWriteRecord> = post_state
            .iter()
            .zip(&pre_state_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let local_mem_access = self.precompile_local_events(
            state_records
                .iter()
                .enumerate()
                .map(|(i, &r)| (state_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
        );
        Poseidon2PermuteEvent {
            shard: 0,
            clk,
            pre_state,
            post_state,
            state_records,
            state_addr: state_ptr,
            local_mem_access,
        }
    }

    /// Builds a `LinuxEvent` for one of the Linux syscall shims (`SYS_BRK`/`SYS_MMAP`/etc, all
    /// bucketed under the synthetic `SyscallCode::SYS_LINUX` key like the legacy `Executor`
    /// itself does -- see `minimal/syscall.rs`'s corresponding dispatch arms for the compute logic
    /// each `read_records`/`write_records`/`v0` pairing mirrors). `SysLinuxChip`'s AIR
    /// (`crates/core/machine/src/syscall/precompiles/sys_linux/air.rs`) evaluates its own
    /// `eval_memory_access` calls for these register touches (A2/A3/HEAP/BRK) from within the chip
    /// itself, which -- like every other precompile event -- gets carved into its own separate
    /// `ExecutionRecord` by `ExecutionRecord::split`'s precompile-event handling. So despite these
    /// addresses also being touched constantly by ordinary instructions elsewhere in the record,
    /// *this* call's own touch must travel with the event via `local_mem_access`
    /// (`precompile_local_events`'s isolation), exactly like `ec_add_event` and friends -- routing
    /// it through the shared `touch_local`/`self.local_memory_access` map instead leaves the chip
    /// that actually emits the interaction (now in a different record) with no matching bracket.
    fn linux_event(
        &self,
        clk: u64,
        a0: u32,
        a1: u32,
        v0: u32,
        syscall_id: u32,
        read_records: Vec<MemoryReadRecord>,
        write_records: Vec<MemoryWriteRecord>,
        local_mem_access: Vec<MemoryLocalEvent>,
    ) -> LinuxEvent {
        LinuxEvent {
            shard: 0,
            clk,
            a0,
            a1,
            v0,
            syscall_code: syscall_id,
            read_records,
            write_records,
            local_mem_access,
        }
    }

    /// See `minimal/syscall.rs`'s module doc for scope (`HALT`/`WRITE`/`SYS_BRK` real, everything
    /// else a documented no-op). Returns `next_pc` (the caller still adds 4 for `next_next_pc`).
    fn execute_syscall(&mut self, clk: u64, pc: u32) -> Result<u32, ExecutionError> {
        let syscall_id = self.core.reg(Register::V0);
        let code = SyscallCode::from_u32(syscall_id);
        // Mirrors `Executor::execute_operation`'s `SYSCALL` branch exactly: `A0`/`A1` are read at
        // B/C (the syscall instruction's own op_b/op_c slots), C before B (matches legacy's exact
        // order -- harmless here since A0/A1 are always distinct fixed registers, but kept
        // consistent with `alu_operands`'s documented C-B-A ordering regardless); `V0` is written
        // at A once the result is known, at the bottom of this function.
        let arg2 = self.core.reg(Register::A1);
        let c_record = self.read_op_c(Register::A1, arg2, clk);
        let arg1 = self.core.reg(Register::A0);
        let b_record = self.read_op_b(Register::A0, arg1, clk);

        let mut next_pc = pc.wrapping_add(4);
        let mut extra_cycles = 0u32;
        let a0_result: Option<u32> = match code {
            // See `CoreVM::execute_syscall`'s identical arm for the full explanation: replay
            // always sees `0` here (regardless of what `MinimalExecutor` actually computed), which
            // makes the guest's own branch-on-return-value check skip the entire unconstrained
            // block -- so this instruction gets a completely ordinary `SyscallEvent` (built at the
            // bottom of this function, same as any other syscall) and nothing else about
            // unconstrained mode needs any code here.
            SyscallCode::ENTER_UNCONSTRAINED => Some(0),
            SyscallCode::HALT => {
                let exit_code = arg1;
                next_pc = 0;
                if exit_code != 0 {
                    return Err(ExecutionError::HaltWithNonZeroExitCode(exit_code));
                }
                self.record.last_exit_code = exit_code;
                None
            }
            SyscallCode::WRITE => {
                let fd = arg1;
                let write_buf = arg2;
                let nbytes = self.core.reg(Register::A2);
                let mut bytes = Vec::with_capacity(nbytes as usize);
                for i in 0..nbytes {
                    let word = self.core.next_oracle_value();
                    bytes.push((word >> (((write_buf + i) % 4) * 8)) as u8);
                }
                let _ = (fd, bytes); // public_values_stream lives on MinimalExecutor/state, not
                                     // ExecutionRecord -- TracingVM has no field to append it to
                                     // yet (deferred alongside the other scoped-out bookkeeping).
                None
            }
            SyscallCode::COMMIT => {
                let word_idx = arg1 as usize;
                if word_idx >= self.record.public_values.committed_value_digest.len() {
                    return Err(ExecutionError::InvalidSyscallArgs());
                }
                self.record.public_values.committed_value_digest[word_idx] = arg2;
                None
            }
            SyscallCode::COMMIT_DEFERRED_PROOFS => {
                let word_idx = arg1 as usize;
                if word_idx >= self.record.public_values.deferred_proofs_digest.len() {
                    return Err(ExecutionError::InvalidSyscallArgs());
                }
                self.record.public_values.deferred_proofs_digest[word_idx] = arg2;
                None
            }
            SyscallCode::SHA_COMPRESS => {
                let w_ptr = arg1;
                let h_ptr = arg2;
                // `ShaCompressChip`'s AIR (`crates/core/machine/src/syscall/precompiles/sha256/
                // compress/air.rs`) evaluates every one of its 80 internal rows' single memory
                // access at `clk_low + is_finalize` -- `clk_low` during the initialize (`h`
                // reads) and compression (`w` reads) phases, `clk_low + 1` only during the
                // finalize phase (the `h` writes). Since `h` is typically a running hash state
                // reused across many SHA_COMPRESS calls (one per block), a later call's own `h`
                // reads need `oracle_entry_at`'s adjusted-timestamp override to see the correct
                // `clk + 1` an earlier call's finalize write actually sent into the interaction
                // argument, not the oracle's own flat (unadjusted) clk.
                let h_entries: [MemValue; 8] =
                    std::array::from_fn(|i| self.oracle_entry_at(h_ptr + i as u32 * 4));
                let h: [u32; 8] = h_entries.map(|e| e.value);
                let h_read_records: [MemoryReadRecord; 8] = h_entries
                    .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk });
                for i in 0..8u32 {
                    self.record_adjusted_touch(h_ptr + i * 4, clk);
                }
                let w_entries: [MemValue; 64] =
                    std::array::from_fn(|i| self.oracle_entry_at(w_ptr + i as u32 * 4));
                let w: [u32; 64] = w_entries.map(|e| e.value);
                let w_i_read_records: Vec<MemoryReadRecord> = w_entries
                    .iter()
                    .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
                    .collect();
                for i in 0..64u32 {
                    self.record_adjusted_touch(w_ptr + i * 4, clk);
                }
                let out = sha256_compress(h, &w);
                let h_write_entries: [MemValue; 8] =
                    std::array::from_fn(|i| self.oracle_entry_at(h_ptr + i as u32 * 4));
                let h_write_records: [MemoryWriteRecord; 8] = std::array::from_fn(|i| MemoryWriteRecord {
                    value: out[i],
                    timestamp: clk + 1,
                    prev_value: h_write_entries[i].value,
                    prev_timestamp: h_write_entries[i].clk,
                });
                for i in 0..8u32 {
                    self.record_adjusted_touch(h_ptr + i * 4, clk + 1);
                }
                let local_mem_access = self.precompile_local_events(
                    h_read_records
                        .iter()
                        .enumerate()
                        .map(|(i, &r)| (h_ptr + 4 * i as u32, MemoryRecordEnum::Read(r)))
                        .chain(
                            w_i_read_records
                                .iter()
                                .enumerate()
                                .map(|(i, &r)| (w_ptr + 4 * i as u32, MemoryRecordEnum::Read(r))),
                        )
                        .chain(
                            h_write_records
                                .iter()
                                .enumerate()
                                .map(|(i, &r)| (h_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                        ),
                );
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc,
                        next_pc,
                        clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None,
                        c_record: None,
                        syscall_id: code.syscall_id(),
                        arg1,
                        arg2,
                    },
                    PrecompileEvent::ShaCompress(ShaCompressEvent {
                        shard: 0,
                        clk,
                        w_ptr,
                        h_ptr,
                        w: w.to_vec(),
                        h,
                        h_read_records,
                        w_i_read_records,
                        h_write_records,
                        local_mem_access,
                    }),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::SHA_EXTEND => {
                let w_ptr = arg1;
                let mut w_i_minus_15_reads = Vec::with_capacity(48);
                let mut w_i_minus_2_reads = Vec::with_capacity(48);
                let mut w_i_minus_16_reads = Vec::with_capacity(48);
                let mut w_i_minus_7_reads = Vec::with_capacity(48);
                let mut w_i_writes = Vec::with_capacity(48);
                // Accumulated in the exact chronological order these addresses are touched --
                // `precompile_local_events` needs that order to correctly resolve which touch is
                // "first" vs "last" per address, since the sliding window means later iterations
                // re-read addresses written by earlier ones.
                let mut touches: Vec<(u32, MemoryRecordEnum)> = Vec::with_capacity(48 * 5);
                // `ShaExtendChip`'s AIR (`crates/core/machine/src/syscall/precompiles/sha256/
                // extend/air.rs`) disambiguates this syscall's 48 internal iterations -- which
                // all share this one instruction's own `clk`, since `clk` only advances between
                // instructions, not within one -- by evaluating every iteration `i`'s memory
                // accesses at the symbolic timestamp `clk_low + (i - 16)`, not the raw `clk_low`.
                // A read's `w[i-15/2/16/7]` can reference the original message block (a real,
                // externally-timestamped previous touch the oracle reports correctly), a word an
                // earlier iteration of this same loop wrote, or (this buffer being reused, e.g. a
                // guest calling this syscall repeatedly on the same `w`) a word an *earlier call*
                // to this same syscall wrote -- the latter two both need `oracle_entry_at`'s
                // adjusted-timestamp override, since the oracle only ever reflects the flat
                // instruction clk. `record_adjusted_touch` records each row's own effective
                // timestamp for whichever touch (another row of this call, or the next call)
                // comes next to consume.
                for i in 16..64u32 {
                    let row_clk = clk + u64::from(i - 16);
                    let addr = w_ptr + (i - 15) * 4;
                    let e = self.oracle_entry_at(addr);
                    let w_i_minus_15 = e.value;
                    let r = MemoryReadRecord { value: w_i_minus_15, timestamp: row_clk, prev_timestamp: e.clk };
                    w_i_minus_15_reads.push(r);
                    touches.push((addr, MemoryRecordEnum::Read(r)));
                    self.record_adjusted_touch(addr, row_clk);
                    let addr = w_ptr + (i - 2) * 4;
                    let e = self.oracle_entry_at(addr);
                    let w_i_minus_2 = e.value;
                    let r = MemoryReadRecord { value: w_i_minus_2, timestamp: row_clk, prev_timestamp: e.clk };
                    w_i_minus_2_reads.push(r);
                    touches.push((addr, MemoryRecordEnum::Read(r)));
                    self.record_adjusted_touch(addr, row_clk);
                    let addr = w_ptr + (i - 16) * 4;
                    let e = self.oracle_entry_at(addr);
                    let w_i_minus_16 = e.value;
                    let r = MemoryReadRecord { value: w_i_minus_16, timestamp: row_clk, prev_timestamp: e.clk };
                    w_i_minus_16_reads.push(r);
                    touches.push((addr, MemoryRecordEnum::Read(r)));
                    self.record_adjusted_touch(addr, row_clk);
                    let addr = w_ptr + (i - 7) * 4;
                    let e = self.oracle_entry_at(addr);
                    let w_i_minus_7 = e.value;
                    let r = MemoryReadRecord { value: w_i_minus_7, timestamp: row_clk, prev_timestamp: e.clk };
                    w_i_minus_7_reads.push(r);
                    touches.push((addr, MemoryRecordEnum::Read(r)));
                    self.record_adjusted_touch(addr, row_clk);
                    let w_i =
                        sha256_extend_word(w_i_minus_15, w_i_minus_2, w_i_minus_16, w_i_minus_7);
                    let addr = w_ptr + i * 4;
                    let e = self.oracle_entry_at(addr);
                    let w = MemoryWriteRecord { value: w_i, timestamp: row_clk, prev_value: e.value, prev_timestamp: e.clk };
                    w_i_writes.push(w);
                    touches.push((addr, MemoryRecordEnum::Write(w)));
                    self.record_adjusted_touch(addr, row_clk);
                }
                let local_mem_access = self.precompile_local_events(touches);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc,
                        next_pc,
                        clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None,
                        c_record: None,
                        syscall_id: code.syscall_id(),
                        arg1,
                        arg2,
                    },
                    PrecompileEvent::ShaExtend(ShaExtendEvent {
                        shard: 0,
                        clk,
                        w_ptr,
                        w_i_minus_15_reads,
                        w_i_minus_2_reads,
                        w_i_minus_16_reads,
                        w_i_minus_7_reads,
                        w_i_writes,
                        local_mem_access,
                    }),
                );
                extra_cycles = 48;
                None
            }
            SyscallCode::KECCAK_SPONGE => {
                let input_ptr = arg1;
                let result_ptr = arg2;
                // `KeccakSpongeChip`'s AIR (`crates/core/machine/src/syscall/precompiles/
                // keccak_sponge/air.rs`'s `eval_memory_access`) reads the input length and each
                // input block word at plain `clk_low`, but writes the output at `clk_low + 1` --
                // and since a sponge construction commonly chains one call's output into the next
                // call's input (or reuses a scratch `result_ptr`/`input_ptr` across calls), any of
                // these addresses can reference a value an earlier call's own (possibly adjusted)
                // write produced. `oracle_entry_at`/`record_adjusted_touch` keep every touch here
                // in sync with whatever the producing row actually sent into the interaction
                // argument, the same as `SHA_EXTEND`/`SHA_COMPRESS`.
                let input_len_addr = result_ptr + 16 * 4;
                let input_len_entry = self.oracle_entry_at(input_len_addr);
                let input_len_u32s = input_len_entry.value;
                let input_length_record = MemoryReadRecord {
                    value: input_len_u32s,
                    timestamp: clk,
                    prev_timestamp: input_len_entry.clk,
                };
                self.record_adjusted_touch(input_len_addr, clk);
                let mut input_values = Vec::with_capacity(input_len_u32s as usize);
                let mut input_read_records = Vec::with_capacity(input_len_u32s as usize);
                for i in 0..input_len_u32s {
                    let addr = input_ptr + 4 * i;
                    let e = self.oracle_entry_at(addr);
                    input_values.push(e.value);
                    input_read_records.push(MemoryReadRecord {
                        value: e.value,
                        timestamp: clk,
                        prev_timestamp: e.clk,
                    });
                    self.record_adjusted_touch(addr, clk);
                }
                let input_u64_values: Vec<u64> = input_values
                    .chunks_exact(2)
                    .map(|pair| pair[0] as u64 + ((pair[1] as u64) << 32))
                    .collect();

                let mut state = [0u64; KECCAK_STATE_SIZE_U64S];
                let mut xored_state_list = Vec::new();
                for block in input_u64_values.chunks_exact(KECCAK_GENERAL_BLOCK_SIZE_U64S) {
                    keccak_xor_block(&mut state, block);
                    xored_state_list.push(state);
                    keccakf(&mut state);
                }

                let mut values_to_write = Vec::with_capacity(2 * KECCAK_GENERAL_OUTPUT_U64S);
                for &lane in state.iter().take(KECCAK_GENERAL_OUTPUT_U64S) {
                    values_to_write.push((lane & 0xFFFF_FFFF) as u32);
                    values_to_write.push((lane >> 32) as u32);
                }
                let output_write_records: Vec<MemoryWriteRecord> = values_to_write
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        let addr = result_ptr + 4 * i as u32;
                        let e = self.oracle_entry_at(addr);
                        let r = MemoryWriteRecord {
                            value,
                            timestamp: clk + 1,
                            prev_value: e.value,
                            prev_timestamp: e.clk,
                        };
                        self.record_adjusted_touch(addr, clk + 1);
                        r
                    })
                    .collect();
                let local_mem_access = self.precompile_local_events(
                    std::iter::once((result_ptr + 16 * 4, MemoryRecordEnum::Read(input_length_record)))
                        .chain(
                            input_read_records
                                .iter()
                                .enumerate()
                                .map(|(i, &r)| (input_ptr + 4 * i as u32, MemoryRecordEnum::Read(r))),
                        )
                        .chain(
                            output_write_records
                                .iter()
                                .enumerate()
                                .map(|(i, &r)| (result_ptr + 4 * i as u32, MemoryRecordEnum::Write(r))),
                        ),
                );

                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc,
                        next_pc,
                        clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None,
                        c_record: None,
                        syscall_id: code.syscall_id(),
                        arg1,
                        arg2,
                    },
                    PrecompileEvent::KeccakSponge(KeccakSpongeEvent {
                        shard: 0,
                        clk,
                        input: input_values,
                        output: values_to_write.try_into().unwrap(),
                        input_len_u32s,
                        input_read_records,
                        input_length_record,
                        output_write_records,
                        xored_state_list,
                        input_addr: input_ptr,
                        output_addr: result_ptr,
                        local_mem_access,
                    }),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::SECP256K1_ADD => {
                let event = self.ec_add_event::<Secp256k1>(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Secp256k1Add(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::SECP256R1_ADD => {
                let event = self.ec_add_event::<Secp256r1>(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Secp256r1Add(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_ADD => {
                let event = self.ec_add_event::<Bn254>(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bn254Add(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_ADD => {
                let event = self.ec_add_event::<Bls12381>(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bls12381Add(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::SECP256K1_DOUBLE => {
                let event = self.ec_double_event::<Secp256k1>(clk, arg1);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Secp256k1Double(event),
                );
                None
            }
            SyscallCode::SECP256R1_DOUBLE => {
                let event = self.ec_double_event::<Secp256r1>(clk, arg1);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Secp256r1Double(event),
                );
                None
            }
            SyscallCode::BN254_DOUBLE => {
                let event = self.ec_double_event::<Bn254>(clk, arg1);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bn254Double(event),
                );
                None
            }
            SyscallCode::BLS12381_DOUBLE => {
                let event = self.ec_double_event::<Bls12381>(clk, arg1);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bls12381Double(event),
                );
                None
            }
            SyscallCode::SECP256K1_DECOMPRESS => {
                let event = self.ec_decompress_event::<Secp256k1>(clk, arg1, arg2)?;
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Secp256k1Decompress(event),
                );
                None
            }
            SyscallCode::SECP256R1_DECOMPRESS => {
                let event = self.ec_decompress_event::<Secp256r1>(clk, arg1, arg2)?;
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Secp256r1Decompress(event),
                );
                None
            }
            SyscallCode::BLS12381_DECOMPRESS => {
                let event = self.ec_decompress_event::<Bls12381>(clk, arg1, arg2)?;
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bls12381Decompress(event),
                );
                None
            }
            SyscallCode::ED_ADD => {
                let event = self.ec_add_event::<Ed25519>(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::EdAdd(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::ED_DECOMPRESS => {
                let event = self.ed_decompress_event(clk, arg1, arg2)?;
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::EdDecompress(event),
                );
                None
            }
            SyscallCode::BN254_FP_ADD | SyscallCode::BN254_FP_SUB | SyscallCode::BN254_FP_MUL => {
                let op = match code {
                    SyscallCode::BN254_FP_ADD => FieldOperation::Add,
                    SyscallCode::BN254_FP_SUB => FieldOperation::Sub,
                    _ => FieldOperation::Mul,
                };
                let event = self.fp_op_event::<Bn254BaseField>(clk, arg1, arg2, op);
                self.record.precompile_events.add_event(
                    SyscallCode::BN254_FP_ADD,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bn254Fp(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP_ADD | SyscallCode::BLS12381_FP_SUB | SyscallCode::BLS12381_FP_MUL => {
                let op = match code {
                    SyscallCode::BLS12381_FP_ADD => FieldOperation::Add,
                    SyscallCode::BLS12381_FP_SUB => FieldOperation::Sub,
                    _ => FieldOperation::Mul,
                };
                let event = self.fp_op_event::<Bls12381BaseField>(clk, arg1, arg2, op);
                self.record.precompile_events.add_event(
                    SyscallCode::BLS12381_FP_ADD,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bls12381Fp(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_FP2_ADD | SyscallCode::BN254_FP2_SUB => {
                let op = if code == SyscallCode::BN254_FP2_ADD { FieldOperation::Add } else { FieldOperation::Sub };
                let event = self.fp2_addsub_event::<Bn254BaseField>(clk, arg1, arg2, op);
                self.record.precompile_events.add_event(
                    SyscallCode::BN254_FP2_ADD,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bn254Fp2AddSub(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP2_ADD | SyscallCode::BLS12381_FP2_SUB => {
                let op = if code == SyscallCode::BLS12381_FP2_ADD { FieldOperation::Add } else { FieldOperation::Sub };
                let event = self.fp2_addsub_event::<Bls12381BaseField>(clk, arg1, arg2, op);
                self.record.precompile_events.add_event(
                    SyscallCode::BLS12381_FP2_ADD,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bls12381Fp2AddSub(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_FP2_MUL => {
                let event = self.fp2_mul_event::<Bn254BaseField>(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bn254Fp2Mul(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP2_MUL => {
                let event = self.fp2_mul_event::<Bls12381BaseField>(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Bls12381Fp2Mul(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::UINT256_MUL => {
                let event = self.uint256_mul_event(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Uint256Mul(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::U256XU2048_MUL => {
                let event = self.u256xu2048_mul_event(clk, arg1, arg2);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::U256xU2048Mul(event),
                );
                extra_cycles = 1;
                None
            }
            SyscallCode::POSEIDON2_PERMUTE => {
                let event = self.poseidon2_permute_event(clk, arg1);
                self.record.precompile_events.add_event(
                    code,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: syscall_id,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Poseidon2Permute(event),
                );
                None
            }
            // See `CoreVM::hint_len`/`CoreVM::advance_input_stream_ptr`'s doc comments:
            // `SYSHINTLEN` pops one oracle entry; `SYSHINTREAD` pops none and never touches RAM
            // directly -- its seeding of `MinimalExecutor::hint_seed` only ever materializes into
            // the oracle log the moment a real load/store first touches that address, exactly like
            // every other RAM preimage.
            SyscallCode::SYSHINTLEN => Some(self.core.hint_len()),
            SyscallCode::SYSHINTREAD => {
                self.core.advance_input_stream_ptr();
                None
            }
            SyscallCode::SYS_BRK => {
                let initial_brk = self
                    .core
                    .program()
                    .image
                    .get(&(Register::BRK as u32))
                    .copied()
                    .unwrap_or_else(|| self.core.reg(Register::BRK));
                let brk_ts = self.core.reg_timestamp(Register::BRK);
                self.core.set_reg_aux(Register::BRK, initial_brk);
                let v0 = resolve_brk(initial_brk, initial_brk, arg1)?;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let brk_record =
                    MemoryReadRecord { value: initial_brk, timestamp: clk, prev_timestamp: brk_ts };
                let a3_record = MemoryWriteRecord {
                    value: 0,
                    timestamp: clk,
                    prev_value: prev_a3,
                    prev_timestamp: prev_a3_ts,
                };
                let local_mem_access = self.precompile_local_events([
                    (Register::BRK as u32, MemoryRecordEnum::Read(brk_record)),
                    (Register::A3 as u32, MemoryRecordEnum::Write(a3_record)),
                ]);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id,
                    vec![brk_record],
                    vec![a3_record],
                    local_mem_access,
                );
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_MMAP | SyscallCode::SYS_MMAP2 => {
                let size = align_size(arg2)?;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let a3_record = MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts };
                let (v0, write_records, heap_touch) = if arg1 == 0 {
                    let heap = self.core.reg(Register::HEAP);
                    let prev_heap_ts = self.core.reg_timestamp(Register::HEAP);
                    self.core.set_reg_aux(Register::HEAP, heap.wrapping_add(size));
                    let heap_record = MemoryWriteRecord {
                        value: heap.wrapping_add(size),
                        timestamp: clk,
                        prev_value: heap,
                        prev_timestamp: prev_heap_ts,
                    };
                    (heap, vec![a3_record, heap_record], Some(heap_record))
                } else {
                    (arg1, vec![a3_record], None)
                };
                let local_mem_access = self.precompile_local_events(
                    [(Register::A3 as u32, MemoryRecordEnum::Write(a3_record))].into_iter().chain(
                        heap_touch.map(|r| (Register::HEAP as u32, MemoryRecordEnum::Write(r))),
                    ),
                );
                let event =
                    self.linux_event(clk, arg1, arg2, v0, syscall_id, vec![], write_records, local_mem_access);
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_CLONE => {
                let v0 = 1;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let a3_record = MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts };
                let local_mem_access =
                    self.precompile_local_events([(Register::A3 as u32, MemoryRecordEnum::Write(a3_record))]);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![a3_record],
                    local_mem_access,
                );
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_EXT_GROUP => {
                next_pc = 0;
                let v0 = 0;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let a3_record = MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts };
                let local_mem_access =
                    self.precompile_local_events([(Register::A3 as u32, MemoryRecordEnum::Write(a3_record))]);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![a3_record],
                    local_mem_access,
                );
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_FCNTL => {
                let (v0, a3) = fcntl_result(arg1, arg2);
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, a3);
                let a3_record = MemoryWriteRecord { value: a3, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts };
                let local_mem_access =
                    self.precompile_local_events([(Register::A3 as u32, MemoryRecordEnum::Write(a3_record))]);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![a3_record],
                    local_mem_access,
                );
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_READ => {
                let (v0, a3) = read_result(arg1);
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, a3);
                let a3_record = MemoryWriteRecord { value: a3, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts };
                let local_mem_access =
                    self.precompile_local_events([(Register::A3 as u32, MemoryRecordEnum::Write(a3_record))]);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![a3_record],
                    local_mem_access,
                );
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_WRITE => {
                let nbytes = self.core.reg(Register::A2);
                let nbytes_ts = self.core.reg_timestamp(Register::A2);
                self.core.set_reg_aux(Register::A2, nbytes);
                let a2_record = MemoryReadRecord { value: nbytes, timestamp: clk, prev_timestamp: nbytes_ts };
                for _ in 0..nbytes {
                    self.core.next_oracle_value(); // write preimage; see SHA_COMPRESS.
                }
                let v0 = nbytes;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let a3_record = MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts };
                let local_mem_access = self.precompile_local_events([
                    (Register::A2 as u32, MemoryRecordEnum::Read(a2_record)),
                    (Register::A3 as u32, MemoryRecordEnum::Write(a3_record)),
                ]);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id,
                    vec![a2_record],
                    vec![a3_record],
                    local_mem_access,
                );
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_OPEN
            | SyscallCode::SYS_CLOSE
            | SyscallCode::SYS_RT_SIGACTION
            | SyscallCode::SYS_RT_SIGPROCMASK
            | SyscallCode::SYS_MADVISE
            | SyscallCode::SYS_GETTID
            | SyscallCode::SYS_SCHED_GETAFFINITY
            | SyscallCode::SYS_CLOCK_GETTIME
            | SyscallCode::SYS_NANOSLEEP
            | SyscallCode::SYS_PRLIMIT64
            | SyscallCode::SYS_SIGALTSTACK
            | SyscallCode::SYS_OPENAT
            | SyscallCode::SYS_FSTAT64
            | SyscallCode::SYS_MUNMAP => {
                let v0 = 0;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let a3_record = MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts };
                let local_mem_access =
                    self.precompile_local_events([(Register::A3 as u32, MemoryRecordEnum::Write(a3_record))]);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![a3_record],
                    local_mem_access,
                );
                self.record.precompile_events.add_event(
                    SyscallCode::SYS_LINUX,
                    SyscallEvent {
                        pc, next_pc, clk,
                        a_record: MemoryWriteRecord {
                            value: v0,
                            timestamp: clk,
                            prev_value: syscall_id,
                            prev_timestamp: self.core.reg_timestamp(Register::V0),
                        },
                        a_record_is_real: true,
                        b_record: None, c_record: None,
                        syscall_id: code.syscall_id(), arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            _ => None,
        };

        let a0 = a0_result.unwrap_or(syscall_id);
        let a_record = match self.write_op_a(Register::V0, a0, clk) {
            MemoryRecordEnum::Write(r) => r,
            MemoryRecordEnum::Read(_) => unreachable!("write_op_a always returns a Write record"),
        };
        // SYSCALL's register operands (always V0/A0/A1) use the cheap register-access scheme
        // (see `Executor::emit_memory_bump_events`'s call site comment), so this is unconditional.
        self.maybe_bump(Register::V0, &MemoryRecordEnum::Write(a_record));
        self.maybe_bump(Register::A0, &b_record);
        self.maybe_bump(Register::A1, &c_record);
        self.core.advance_clk_extra(extra_cycles);
        self.record.syscall_events.push(SyscallEvent {
            pc,
            next_pc,
            clk,
            a_record,
            a_record_is_real: true,
            b_record: Some(b_record),
            c_record: Some(c_record),
            // `SyscallInstrsChip`'s AIR (`crates/core/machine/src/syscall/instructions/
            // air.rs`) requires the witnessed `syscall_id` column to read as
            // `EXIT_UNCONSTRAINED`'s id, not `ENTER_UNCONSTRAINED`'s own, on an
            // `ENTER_UNCONSTRAINED` dispatch. `code` (the dispatch match target above) stays
            // the real `ENTER_UNCONSTRAINED` throughout; only this witnessed column needs the
            // substitution.
            syscall_id: if code == SyscallCode::ENTER_UNCONSTRAINED {
                SyscallCode::EXIT_UNCONSTRAINED.syscall_id()
            } else {
                code.syscall_id()
            },
            arg1,
            arg2,
        });
        Ok(next_pc)
    }
}

fn sign_extend<const BITS: u32>(value: u32) -> u32 {
    let shift = 32 - BITS;
    (((value << shift) as i32) >> shift) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minimal::MinimalExecutor;
    use std::collections::BTreeMap;

    /// Deep, field-level check (not just event counts) that `AluEvent`'s `a_record`/`b_record`/
    /// `c_record` carry real, correctly-positioned timestamps -- `assert_matches_golden`'s count
    /// comparison can't catch a wrong `MemoryAccessPosition` offset or a stale prev_timestamp, so
    /// this hand-verifies the actual clk arithmetic against `Executor::rw_cpu`'s `clk + position`
    /// scheme for a small, fully-known register-dependency chain: `$t0 = 5`, `$t1 = 7`,
    /// `$t2 = $t0 + $t1` (register-register form, so `$t0`/`$t1` are real read records).
    #[test]
    fn alu_event_records_have_real_position_tagged_timestamps() {
        const T0: u8 = Register::T0 as u8;
        const T1: u8 = Register::T1 as u8;
        const T2: u8 = Register::T2 as u8;
        const ZERO: u8 = Register::ZERO as u8;

        let program = Program::new(
            vec![
                Instruction::new(Opcode::ADD, T0, ZERO as u32, 5, false, true),
                Instruction::new(Opcode::ADD, T1, ZERO as u32, 7, false, true),
                Instruction::new(Opcode::ADD, T2, T0 as u32, T1 as u32, false, false),
            ],
            0,
            0,
        );

        let program = Arc::new(program);
        let mut minimal = MinimalExecutor::new(program.clone(), u64::MAX / 2);
        let chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");
        let max_syscall_cycles = minimal.max_syscall_cycles();

        let mut record = ExecutionRecord::new(program.clone());
        let mut tracing_vm = TracingVM::new(&chunk, program, max_syscall_cycles, &mut record);
        assert_eq!(tracing_vm.execute().unwrap(), CoreVMStatus::Done);

        assert_eq!(record.add_events.len(), 1, "only the 3rd (register-register) ADD isn't Addi");
        let event = record.add_events[0];

        // Instruction 1 (`$t0 = 5`) retires at clk 1, so its own `a_record` (position A) is
        // timestamped `1 + 3 = 4`. Instruction 2 (`$t1 = 7`) retires at clk `1 + 5 = 6`, so its
        // `a_record` is timestamped `6 + 3 = 9`. Instruction 3 retires at clk `6 + 5 = 11`.
        let instr3_clk = 11;
        assert_eq!(event.clk, instr3_clk);

        match event.b_record {
            Some(MemoryRecordEnum::Read(r)) => {
                assert_eq!(r.value, 5, "b (=$t0) should read the value instruction 1 wrote");
                assert_eq!(r.timestamp, instr3_clk + MemoryAccessPosition::B as u64);
                assert_eq!(r.prev_timestamp, 1 + MemoryAccessPosition::A as u64, "$t0's own last-write timestamp, from instruction 1");
            }
            other => panic!("expected a real b_record, got {other:?}"),
        }

        match event.c_record {
            Some(MemoryRecordEnum::Read(r)) => {
                assert_eq!(r.value, 7, "c (=$t1) should read the value instruction 2 wrote");
                assert_eq!(r.timestamp, instr3_clk + MemoryAccessPosition::C as u64);
                assert_eq!(r.prev_timestamp, 6 + MemoryAccessPosition::A as u64, "$t1's own last-write timestamp, from instruction 2");
            }
            other => panic!("expected a real c_record, got {other:?}"),
        }

        match event.a_record {
            Some(MemoryRecordEnum::Write(r)) => {
                assert_eq!(r.value, 12, "a (=$t2) should be 5 + 7");
                assert_eq!(r.timestamp, instr3_clk + MemoryAccessPosition::A as u64);
                assert_eq!(r.prev_value, 0, "$t2 was never written before");
                assert_eq!(r.prev_timestamp, 0, "$t2 was never written before");
            }
            other => panic!("expected a real a_record, got {other:?}"),
        }
    }

    #[test]
    fn cpu_local_memory_access_tracks_first_and_last_touch_per_register() {
        const T0: u8 = Register::T0 as u8;
        const T1: u8 = Register::T1 as u8;
        const T2: u8 = Register::T2 as u8;
        const ZERO: u8 = Register::ZERO as u8;

        let program = Program::new(
            vec![
                Instruction::new(Opcode::ADD, T0, ZERO as u32, 5, false, true),
                Instruction::new(Opcode::ADD, T1, ZERO as u32, 7, false, true),
                Instruction::new(Opcode::ADD, T2, T0 as u32, T1 as u32, false, false),
            ],
            0,
            0,
        );

        let program = Arc::new(program);
        let mut minimal = MinimalExecutor::new(program.clone(), u64::MAX / 2);
        let chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");
        let max_syscall_cycles = minimal.max_syscall_cycles();

        let mut record = ExecutionRecord::new(program.clone());
        let mut tracing_vm = TracingVM::new(&chunk, program, max_syscall_cycles, &mut record);
        assert_eq!(tracing_vm.execute().unwrap(), CoreVMStatus::Done);

        let by_addr: BTreeMap<u32, MemoryLocalEvent> =
            record.cpu_local_memory_access.iter().map(|e| (e.addr, *e)).collect();

        // Instruction 3 retires at clk 11 (see the timing note in the test above).
        let instr3_clk = 11;

        let t0 = by_addr[&u32::from(T0)];
        assert_eq!(t0.initial_mem_access.value, 0, "$t0 was never touched before instruction 1");
        assert_eq!(t0.initial_mem_access.timestamp, 0);
        assert_eq!(t0.final_mem_access.value, 5, "instruction 3 re-reads the value instruction 1 wrote");
        assert_eq!(t0.final_mem_access.timestamp, instr3_clk + MemoryAccessPosition::B as u64);

        let t1 = by_addr[&u32::from(T1)];
        assert_eq!(t1.initial_mem_access.value, 0, "$t1 was never touched before instruction 2");
        assert_eq!(t1.initial_mem_access.timestamp, 0);
        assert_eq!(t1.final_mem_access.value, 7, "instruction 3 re-reads the value instruction 2 wrote");
        assert_eq!(t1.final_mem_access.timestamp, instr3_clk + MemoryAccessPosition::C as u64);

        let t2 = by_addr[&u32::from(T2)];
        assert_eq!(t2.initial_mem_access.value, 0, "$t2 was never touched before its own write");
        assert_eq!(t2.initial_mem_access.timestamp, 0);
        assert_eq!(t2.final_mem_access.value, 12, "5 + 7");
        assert_eq!(t2.final_mem_access.timestamp, instr3_clk + MemoryAccessPosition::A as u64);
    }

    /// Regression test for a real bug: `CoreVM::register_timestamps` was only updated by writes,
    /// not reads, so a register touched twice with no intervening write (including a single
    /// instruction reading the same register at both its B and C operand slots, e.g. `ADD $t1,
    /// $t0, $t0`) would give the second access a `prev_timestamp` chained to a stale earlier
    /// write instead of the first access's own timestamp -- breaking the memory-consistency
    /// argument. Also exercises the C-before-B read order `alu_operands` must use (mirroring
    /// `Executor::alu_rr`) for this aliasing case to chain correctly at all.
    #[test]
    fn read_op_b_and_c_chain_correctly_when_aliasing_the_same_register() {
        const T0: u8 = Register::T0 as u8;
        const T1: u8 = Register::T1 as u8;
        const ZERO: u8 = Register::ZERO as u8;

        let program = Program::new(
            vec![
                Instruction::new(Opcode::ADD, T0, ZERO as u32, 5, false, true),
                // $t1 = $t0 + $t0 -- reads $t0 at both B and C.
                Instruction::new(Opcode::ADD, T1, T0 as u32, T0 as u32, false, false),
                // $t0 = $t1 + $t1 -- reads $t1 at both B and C.
                Instruction::new(Opcode::ADD, T0, T1 as u32, T1 as u32, false, false),
            ],
            0,
            0,
        );

        let program = Arc::new(program);
        let mut minimal = MinimalExecutor::new(program.clone(), u64::MAX / 2);
        let chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");
        let max_syscall_cycles = minimal.max_syscall_cycles();

        let mut record = ExecutionRecord::new(program.clone());
        let mut tracing_vm = TracingVM::new(&chunk, program, max_syscall_cycles, &mut record);
        assert_eq!(tracing_vm.execute().unwrap(), CoreVMStatus::Done);

        assert_eq!(record.add_events.len(), 2, "instructions 2 and 3 are both register-register ADD");

        // Instruction 1 ($t0 = 5) retires at clk 1; instruction 2 at clk 6; instruction 3 at clk 11.
        let instr2_clk = 6;
        let instr2_c_ts = instr2_clk + MemoryAccessPosition::C as u64;
        let instr2_b_ts = instr2_clk + MemoryAccessPosition::B as u64;

        let event2 = record.add_events[0];
        assert_eq!(event2.a, 10, "$t1 = 5 + 5");
        match (event2.b_record, event2.c_record) {
            (Some(MemoryRecordEnum::Read(b)), Some(MemoryRecordEnum::Read(c))) => {
                assert_eq!(c.value, 5);
                assert_eq!(c.timestamp, instr2_c_ts);
                assert_eq!(c.prev_timestamp, 1 + MemoryAccessPosition::A as u64, "chains from instruction 1's write");
                assert_eq!(b.value, 5);
                assert_eq!(b.timestamp, instr2_b_ts);
                assert_eq!(
                    b.prev_timestamp, instr2_c_ts,
                    "B must chain from C's own timestamp (this instruction's read), not the stale pre-instruction write"
                );
            }
            other => panic!("expected real b_record/c_record, got {other:?}"),
        }

        let instr3_clk = 11;
        let instr3_c_ts = instr3_clk + MemoryAccessPosition::C as u64;
        let instr3_b_ts = instr3_clk + MemoryAccessPosition::B as u64;

        let event3 = record.add_events[1];
        assert_eq!(event3.a, 20, "$t0 = 10 + 10");
        match (event3.b_record, event3.c_record) {
            (Some(MemoryRecordEnum::Read(b)), Some(MemoryRecordEnum::Read(c))) => {
                assert_eq!(c.value, 10);
                assert_eq!(c.timestamp, instr3_c_ts);
                assert_eq!(
                    c.prev_timestamp,
                    instr2_clk + MemoryAccessPosition::A as u64,
                    "chains from instruction 2's write of $t1 (its own destination, the last touch of $t1)"
                );
                assert_eq!(b.value, 10);
                assert_eq!(b.timestamp, instr3_b_ts);
                assert_eq!(b.prev_timestamp, instr3_c_ts, "B must chain from C's own timestamp again");
            }
            other => panic!("expected real b_record/c_record, got {other:?}"),
        }
    }

    /// End-to-end unconstrained-mode test: a hand-crafted program mirroring exactly what the
    /// guest-side `unconstrained!{}` macro compiles to (`v0 = syscall_enter_unconstrained(); if
    /// v0 != 0 { <block>; syscall_exit_unconstrained(); }`), so it exercises the *real* mechanism
    /// (branch-on-return-value) rather than a simplified stand-in. `$t0` is set to 42 before the
    /// block, corrupted to 999 *inside* the block (real, uncommitted execution -- not skipped),
    /// and must read back as 42 afterward, proving the COW-backed rollback actually discarded it.
    ///
    /// Also verifies the replay side: `MinimalExecutor`'s real run and `TracingVM`'s independent
    /// replay must reach the exact same final register state, confirming `CoreVM`'s unconditional
    /// `a=0` for `ENTER_UNCONSTRAINED` correctly makes it skip the entire block (including the
    /// `EXIT_UNCONSTRAINED` syscall itself) via ordinary branch-not-taken control flow, with no
    /// oracle-log desync despite `MinimalExecutor` executing the block body for real.
    #[test]
    fn unconstrained_block_is_rolled_back_and_replay_matches() {
        const T0: u8 = Register::T0 as u8;
        const T1: u8 = Register::T1 as u8;
        const T2: u8 = Register::T2 as u8;
        const V0: u8 = Register::V0 as u8;
        const A0: u8 = Register::A0 as u8;
        const ZERO: u8 = Register::ZERO as u8;

        let syscall_instr = |a: u8, b: u32| Instruction::new(Opcode::SYSCALL, a, b, 5, false, false);

        let instructions = vec![
            /* 0, pc=0  */ Instruction::new(Opcode::ADD, T0, ZERO as u32, 42, false, true),
            /* 1, pc=4  */ Instruction::new(
                Opcode::ADD,
                V0,
                ZERO as u32,
                SyscallCode::ENTER_UNCONSTRAINED as u32,
                false,
                true,
            ),
            /* 2, pc=8  */ syscall_instr(V0, A0 as u32), // ENTER_UNCONSTRAINED
            /* 3, pc=12 */ Instruction::new(Opcode::BEQ, V0, ZERO as u32, 16, false, true),
            /* 4, pc=16 */ Instruction::new(Opcode::ADD, T2, ZERO as u32, 0, false, true), // delay slot
            /* 5, pc=20 */ Instruction::new(Opcode::ADD, T0, ZERO as u32, 999, false, true), // block body
            /* 6, pc=24 */ Instruction::new(
                Opcode::ADD,
                V0,
                ZERO as u32,
                SyscallCode::EXIT_UNCONSTRAINED as u32,
                false,
                true,
            ),
            /* 7, pc=28 */ syscall_instr(V0, A0 as u32), // EXIT_UNCONSTRAINED
            /* 8, pc=32 */ Instruction::new(Opcode::ADD, T1, T0 as u32, ZERO as u32, false, false),
            /* 9, pc=36 */ Instruction::new(Opcode::ADD, V0, ZERO as u32, 0, false, true), // HALT code
            /* 10,pc=40 */ Instruction::new(Opcode::ADD, A0, ZERO as u32, 0, false, true), // exit code 0
            /* 11,pc=44 */ syscall_instr(V0, A0 as u32), // HALT
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut minimal = MinimalExecutor::new(program.clone(), u64::MAX / 2);
        let chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");
        assert_eq!(
            minimal.registers()[T0 as usize],
            42,
            "MinimalExecutor: $t0 must read back as 42, not 999 -- the unconstrained write must \
             have been rolled back"
        );

        let max_syscall_cycles = minimal.max_syscall_cycles();
        let mut record = ExecutionRecord::new(program.clone());
        let mut tracing_vm = TracingVM::new(&chunk, program, max_syscall_cycles, &mut record);
        assert_eq!(tracing_vm.execute().unwrap(), CoreVMStatus::Done);

        assert_eq!(
            tracing_vm.registers()[T0 as usize],
            42,
            "TracingVM replay must independently reach the same rolled-back $t0"
        );
        assert_eq!(
            tracing_vm.registers(),
            minimal.registers(),
            "TracingVM's full final register state must match MinimalExecutor's real run"
        );
        assert_eq!(tracing_vm.pc(), minimal.pc());
    }
}
