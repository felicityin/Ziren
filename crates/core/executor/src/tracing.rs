//! `TracingVM` -- replays a [`SplicedMinimalTrace`]-shaped [`MinimalTrace`] shard and constructs
//! the `ExecutionRecord` events that describe it, using the same oracle-driven `CoreVM` primitives
//! `SplicingVM` uses for shard-cut accounting. Event *values* (opcode, operands, results) and
//! event *counts* (which `ExecutionRecord` vector each instruction routes to, including the
//! `op_a == 0` special cases the legacy executor's chips route separately) are built to match the
//! legacy `Executor` exactly.
//!
//! Per-operand `MemoryRecordEnum`/`MemoryWriteRecord` fields on each event (the register/memory
//! timestamp bookkeeping the AIR's memory-consistency argument needs for real proving) are left at
//! placeholder values (the correct *value*, a `0`/current-`clk` timestamp) rather than the
//! precise, chased-through-every-access values the legacy executor computes. Likewise
//! `bump_memory_events`/`cpu_local_memory_access`/`global_memory_initialize_events`/
//! `global_memory_finalize_events`/`global_lookup_events` are not populated at all yet. Real
//! proving needs all of this to be exact; it is not built here yet.

#![allow(dead_code)]

use std::sync::Arc;

use crate::{
    events::{
        AluEvent, BranchEvent, BumpClkHighEvent, CompAluEvent, EdDecompressEvent,
        EllipticCurveAddEvent, EllipticCurveDecompressEvent, EllipticCurveDoubleEvent,
        FieldOperation, Fp2AddSubEvent, Fp2MulEvent, FpOpEvent, JumpEvent, KeccakSpongeEvent,
        LinuxEvent, MemInstrEvent, MemoryAccessPosition, MemoryReadRecord, MemoryRecordEnum,
        MemoryWriteRecord, MiscEvent, MovCondEvent, Poseidon2PermuteEvent, PrecompileEvent,
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
}

impl<'a> TracingVM<'a> {
    #[must_use]
    pub(crate) fn new<T: MinimalTrace>(
        trace: &'a T,
        program: Arc<Program>,
        max_syscall_cycles: u32,
        record: &'a mut ExecutionRecord,
    ) -> Self {
        Self { core: CoreVM::new(trace, program, max_syscall_cycles), record }
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
        loop {
            self.execute_instruction()?;
            if self.core.is_halted() {
                return Ok(CoreVMStatus::Done);
            }
            if self.core.clk() >= self.core.clk_end() {
                return Ok(CoreVMStatus::TraceEnd);
            }
        }
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
    /// of instruction families (branches) ever *read* `op_a` rather than write it.
    fn read_op_a(&self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        MemoryRecordEnum::Read(MemoryReadRecord {
            value,
            timestamp: clk + MemoryAccessPosition::A as u64,
            prev_timestamp: self.core.reg_timestamp(r),
        })
    }

    /// Builds a real read record for `op_b`, given its already-resolved value.
    fn read_op_b(&self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        MemoryRecordEnum::Read(MemoryReadRecord {
            value,
            timestamp: clk + MemoryAccessPosition::B as u64,
            prev_timestamp: self.core.reg_timestamp(r),
        })
    }

    /// Builds a real read record for `op_c`, given its already-resolved value.
    fn read_op_c(&self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        MemoryRecordEnum::Read(MemoryReadRecord {
            value,
            timestamp: clk + MemoryAccessPosition::C as u64,
            prev_timestamp: self.core.reg_timestamp(r),
        })
    }

    /// Writes `value` to register `op_a` (`r`), returning the real write record for it.
    fn write_op_a(&mut self, r: Register, value: u32, clk: u64) -> MemoryRecordEnum {
        let prev_value = self.core.reg(r);
        let prev_timestamp = self.core.reg_timestamp(r);
        self.core.set_reg(r, value, MemoryAccessPosition::A);
        MemoryRecordEnum::Write(MemoryWriteRecord {
            value: self.core.reg(r),
            timestamp: clk + MemoryAccessPosition::A as u64,
            prev_value,
            prev_timestamp,
        })
    }

    /// Writes `value` to `Register::HI`, returning the real (non-`Option`) write record for it --
    /// `CompAluEvent`/`MiscEvent`'s `hi_record` field, unlike `a_record`/`b_record`/`c_record`,
    /// isn't wrapped in `MemoryRecordEnum`/`Option` since it's only ever a write.
    fn write_hi(&mut self, value: u32, clk: u64) -> MemoryWriteRecord {
        let prev_value = self.core.reg(Register::HI);
        let prev_timestamp = self.core.reg_timestamp(Register::HI);
        self.core.set_reg(Register::HI, value, MemoryAccessPosition::HI);
        MemoryWriteRecord {
            value: self.core.reg(Register::HI),
            timestamp: clk + MemoryAccessPosition::HI as u64,
            prev_value,
            prev_timestamp,
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
            // `Executor::branch_rr` exactly. The offset (`op_c`) is always a raw immediate for
            // this opcode family, never a register, so there is no `c_record`.
            let rs: Register = instruction.op_a.into();
            let src1 = self.core.reg(rs);
            let a_record = self.read_op_a(rs, src1, clk);
            let (src2, b_record) = if instruction.opcode.only_one_operand() {
                (0, None)
            } else {
                let rt: Register = (instruction.op_b as u8).into();
                let src2 = self.core.reg(rt);
                (src2, Some(self.read_op_b(rt, src2, clk)))
            };
            let offset = instruction.op_c;
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
            let b = self.core.reg(rs);
            let b_record = self.read_op_b(rs, b, clk);
            let c = self.core.reg(rt);
            let c_record = self.read_op_c(rt, c, clk);
            let a = crate::vm::condmov_result(instruction.opcode, prev_a, b, c);
            let a_record = self.write_op_a(rd, a, clk);
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
        &self,
        instruction: &Instruction,
        clk: u64,
    ) -> (Register, u32, Option<MemoryRecordEnum>, u32, Option<MemoryRecordEnum>) {
        if !instruction.imm_c {
            let rd = instruction.op_a.into();
            let b_reg: Register = (instruction.op_b as u8).into();
            let c_reg: Register = (instruction.op_c as u8).into();
            let b = self.core.reg(b_reg);
            let c = self.core.reg(c_reg);
            (rd, b, Some(self.read_op_b(b_reg, b, clk)), c, Some(self.read_op_c(c_reg, c, clk)))
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

        let mut event = MemInstrEvent::new(
            clk,
            pc,
            next_pc,
            instruction.opcode,
            val,
            rs_raw,
            offset,
            MemoryRecordEnum::Read(MemoryReadRecord {
                value: mem,
                timestamp: clk,
                prev_timestamp: mem_entry.clk,
            }),
            rt,
        );
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
        let mut event = MemInstrEvent::new(
            clk,
            pc,
            next_pc,
            instruction.opcode,
            a,
            rs,
            offset,
            MemoryRecordEnum::Write(MemoryWriteRecord {
                value: val,
                timestamp: clk,
                prev_value: mem,
                prev_timestamp: mem_entry.clk,
            }),
            rt,
        );
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
                self.push_misc_or_x0(op_a_is_zero, clk, pc, next_pc, instruction.opcode, a, b, c, 0, a_record, Some(b_record));
            }
            Opcode::EXT => {
                let b = self.core.reg(rt);
                let b_record = self.read_op_b(rt, b, clk);
                let a = crate::vm::ext(b, c)?;
                let a_record = self.write_op_a(rd, a, clk);
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
                self.push_misc_or_x0(op_a_is_zero, clk, pc, next_pc, instruction.opcode, a, b, c, prev_a, a_record, Some(b_record));
            }
            Opcode::TEQ => {
                let rs: Register = instruction.op_a.into();
                let rt: Register = (instruction.op_b as u8).into();
                let src2 = self.core.reg(rt);
                let src2_record = self.read_op_b(rt, src2, clk);
                let src1 = self.core.reg(rs);
                let src1_record = self.read_op_a(rs, src1, clk);
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
                let hi_record = self.write_hi(out_hi, clk);
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
    /// `minimal/ecall.rs::ec_add_dispatch`'s doc comment for why the oracle order is q-then-p
    /// even though `p` is conceptually read first).
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
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        EllipticCurveAddEvent { shard: 0, clk, p_ptr, p, q_ptr, q, p_memory_records, q_memory_records, local_mem_access: Vec::new() }
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
        EllipticCurveDoubleEvent { shard: 0, clk, p_ptr, p, p_memory_records, local_mem_access: Vec::new() }
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
        Ok(EllipticCurveDecompressEvent {
            shard: 0,
            clk,
            ptr,
            sign_bit: sign_bit != 0,
            x_bytes,
            decompressed_y_bytes,
            x_memory_records,
            y_memory_records,
            local_mem_access: Vec::new(),
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
        Ok(EdDecompressEvent {
            shard: 0,
            clk,
            ptr,
            sign: sign != 0,
            y_bytes,
            decompressed_x_bytes,
            x_memory_records,
            y_memory_records,
            local_mem_access: Vec::new(),
        })
    }

    /// Builds an `FpOpEvent`: pops `y`'s reads, then `x`'s write-preimage (the value actually
    /// needed as an input -- see `ec_add_event`'s doc comment for why the oracle order is
    /// y-then-x).
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
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        FpOpEvent { shard: 0, clk, x_ptr, x, y_ptr, y, op, x_memory_records, y_memory_records, local_mem_access: Vec::new() }
    }

    /// Builds an `Fp2AddSubEvent`: pops `y`'s reads, then `x`'s write-preimage.
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
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        Fp2AddSubEvent { shard: 0, clk, op, x_ptr, x, y_ptr, y, x_memory_records, y_memory_records, local_mem_access: Vec::new() }
    }

    /// Builds an `Fp2MulEvent`: pops `y`'s reads, then `x`'s write-preimage.
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
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        Fp2MulEvent { shard: 0, clk, x_ptr, x, y_ptr, y, x_memory_records, y_memory_records, local_mem_access: Vec::new() }
    }

    /// Builds a `Uint256MulEvent`: pops `y`'s reads, `modulus`'s reads, then `x`'s write-preimage
    /// (the value actually needed as an input).
    fn uint256_mul_event(&mut self, clk: u64, x_ptr: u32, y_ptr: u32) -> Uint256MulEvent {
        let y_entries = self.core.next_oracle_entries(8);
        let y: [u32; 8] = y_entries.iter().map(|e| e.value).collect::<Vec<_>>().try_into().unwrap();
        let y_memory_records: Vec<MemoryReadRecord> = y_entries
            .iter()
            .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
            .collect();
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
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
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
            local_mem_access: Vec::new(),
        }
    }

    /// Builds a `U256xU2048MulEvent`: reads `$a2`/`$a3` (live, unlogged) for `lo_ptr`/`hi_ptr`,
    /// pops `a`'s reads, `b`'s reads, then `lo`'s and `hi`'s write-preimages (discarded).
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
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();
        let hi_entries = self.core.next_oracle_entries(hi.len());
        let hi_memory_records: Vec<MemoryWriteRecord> = hi
            .iter()
            .zip(&hi_entries)
            .map(|(&value, e)| MemoryWriteRecord { value, timestamp: clk, prev_value: e.value, prev_timestamp: e.clk })
            .collect();

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
            local_mem_access: Vec::new(),
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
        Poseidon2PermuteEvent {
            shard: 0,
            clk,
            pre_state,
            post_state,
            state_records,
            state_addr: state_ptr,
            local_mem_access: Vec::new(),
        }
    }

    /// Builds a `LinuxEvent` for one of the Linux syscall shims (`SYS_BRK`/`SYS_MMAP`/etc, all
    /// bucketed under the synthetic `SyscallCode::SYS_LINUX` key like the legacy `Executor`
    /// itself does -- see `minimal/ecall.rs`'s corresponding dispatch arms for the compute logic
    /// each `read_records`/`write_records`/`v0` pairing mirrors).
    fn linux_event(
        &self,
        clk: u64,
        a0: u32,
        a1: u32,
        v0: u32,
        syscall_id: u32,
        read_records: Vec<MemoryReadRecord>,
        write_records: Vec<MemoryWriteRecord>,
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
            local_mem_access: Vec::new(),
        }
    }

    /// See `minimal/ecall.rs`'s module doc for scope (`HALT`/`WRITE`/`SYS_BRK` real, everything
    /// else a documented no-op). Returns `next_pc` (the caller still adds 4 for `next_next_pc`).
    fn execute_syscall(&mut self, clk: u64, pc: u32) -> Result<u32, ExecutionError> {
        let syscall_id = self.core.reg(Register::V0);
        let code = SyscallCode::from_u32(syscall_id);
        // Mirrors `Executor::execute_operation`'s `SYSCALL` branch exactly: `A0`/`A1` are read at
        // B/C (the syscall instruction's own op_b/op_c slots); `V0` is written at A once the
        // result is known, at the bottom of this function.
        let arg1 = self.core.reg(Register::A0);
        let b_record = self.read_op_b(Register::A0, arg1, clk);
        let arg2 = self.core.reg(Register::A1);
        let c_record = self.read_op_c(Register::A1, arg2, clk);

        let mut next_pc = pc.wrapping_add(4);
        let mut extra_cycles = 0u32;
        let a0_result: Option<u32> = match code {
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
            SyscallCode::SHA_COMPRESS => {
                let w_ptr = arg1;
                let h_ptr = arg2;
                let h_entries: [MemValue; 8] = std::array::from_fn(|_| self.core.next_oracle_entry());
                let h: [u32; 8] = h_entries.map(|e| e.value);
                let h_read_records: [MemoryReadRecord; 8] = h_entries
                    .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk });
                let w_entries: [MemValue; 64] = std::array::from_fn(|_| self.core.next_oracle_entry());
                let w: [u32; 64] = w_entries.map(|e| e.value);
                let w_i_read_records: Vec<MemoryReadRecord> = w_entries
                    .iter()
                    .map(|e| MemoryReadRecord { value: e.value, timestamp: clk, prev_timestamp: e.clk })
                    .collect();
                let out = sha256_compress(h, &w);
                let h_write_entries: [MemValue; 8] = std::array::from_fn(|_| self.core.next_oracle_entry());
                let h_write_records: [MemoryWriteRecord; 8] = std::array::from_fn(|i| MemoryWriteRecord {
                    value: out[i],
                    timestamp: clk,
                    prev_value: h_write_entries[i].value,
                    prev_timestamp: h_write_entries[i].clk,
                });
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
                        syscall_id,
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
                        local_mem_access: Vec::new(),
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
                for _ in 16..64u32 {
                    let e = self.core.next_oracle_entry();
                    let w_i_minus_15 = e.value;
                    w_i_minus_15_reads.push(MemoryReadRecord {
                        value: w_i_minus_15,
                        timestamp: clk,
                        prev_timestamp: e.clk,
                    });
                    let e = self.core.next_oracle_entry();
                    let w_i_minus_2 = e.value;
                    w_i_minus_2_reads.push(MemoryReadRecord {
                        value: w_i_minus_2,
                        timestamp: clk,
                        prev_timestamp: e.clk,
                    });
                    let e = self.core.next_oracle_entry();
                    let w_i_minus_16 = e.value;
                    w_i_minus_16_reads.push(MemoryReadRecord {
                        value: w_i_minus_16,
                        timestamp: clk,
                        prev_timestamp: e.clk,
                    });
                    let e = self.core.next_oracle_entry();
                    let w_i_minus_7 = e.value;
                    w_i_minus_7_reads.push(MemoryReadRecord {
                        value: w_i_minus_7,
                        timestamp: clk,
                        prev_timestamp: e.clk,
                    });
                    let w_i =
                        sha256_extend_word(w_i_minus_15, w_i_minus_2, w_i_minus_16, w_i_minus_7);
                    let e = self.core.next_oracle_entry();
                    w_i_writes.push(MemoryWriteRecord {
                        value: w_i,
                        timestamp: clk,
                        prev_value: e.value,
                        prev_timestamp: e.clk,
                    });
                }
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
                        syscall_id,
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
                        local_mem_access: Vec::new(),
                    }),
                );
                extra_cycles = 48;
                None
            }
            SyscallCode::KECCAK_SPONGE => {
                let input_ptr = arg1;
                let result_ptr = arg2;
                let input_len_entry = self.core.next_oracle_entry();
                let input_len_u32s = input_len_entry.value;
                let input_length_record = MemoryReadRecord {
                    value: input_len_u32s,
                    timestamp: clk,
                    prev_timestamp: input_len_entry.clk,
                };

                let mut input_values = Vec::with_capacity(input_len_u32s as usize);
                let mut input_read_records = Vec::with_capacity(input_len_u32s as usize);
                for _ in 0..input_len_u32s {
                    let e = self.core.next_oracle_entry();
                    input_values.push(e.value);
                    input_read_records.push(MemoryReadRecord {
                        value: e.value,
                        timestamp: clk,
                        prev_timestamp: e.clk,
                    });
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
                let output_write_entries = self.core.next_oracle_entries(values_to_write.len());
                let output_write_records: Vec<MemoryWriteRecord> = values_to_write
                    .iter()
                    .zip(&output_write_entries)
                    .map(|(&value, e)| MemoryWriteRecord {
                        value,
                        timestamp: clk,
                        prev_value: e.value,
                        prev_timestamp: e.clk,
                    })
                    .collect();

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
                        syscall_id,
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
                        local_mem_access: Vec::new(),
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
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
                        syscall_id, arg1, arg2,
                    },
                    PrecompileEvent::Poseidon2Permute(event),
                );
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
                let v0 = resolve_brk(initial_brk, initial_brk, arg1)?;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id,
                    vec![MemoryReadRecord { value: initial_brk, timestamp: clk, prev_timestamp: 0 }],
                    vec![MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts }],
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
                        syscall_id, arg1, arg2,
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
                let (v0, write_records) = if arg1 == 0 {
                    let heap = self.core.reg(Register::HEAP);
                    let prev_heap_ts = self.core.reg_timestamp(Register::HEAP);
                    self.core.set_reg_aux(Register::HEAP, heap.wrapping_add(size));
                    let heap_record = MemoryWriteRecord {
                        value: heap.wrapping_add(size),
                        timestamp: clk,
                        prev_value: heap,
                        prev_timestamp: prev_heap_ts,
                    };
                    (heap, vec![a3_record, heap_record])
                } else {
                    (arg1, vec![a3_record])
                };
                let event = self.linux_event(clk, arg1, arg2, v0, syscall_id, vec![], write_records);
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
                        syscall_id, arg1, arg2,
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
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts }],
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
                        syscall_id, arg1, arg2,
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
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts }],
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
                        syscall_id, arg1, arg2,
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
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![MemoryWriteRecord { value: a3, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts }],
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
                        syscall_id, arg1, arg2,
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
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![MemoryWriteRecord { value: a3, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts }],
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
                        syscall_id, arg1, arg2,
                    },
                    PrecompileEvent::Linux(event),
                );
                Some(v0)
            }
            SyscallCode::SYS_WRITE => {
                let nbytes = self.core.reg(Register::A2);
                let nbytes_ts = self.core.reg_timestamp(Register::A2);
                for _ in 0..nbytes {
                    self.core.next_oracle_value(); // write preimage; see SHA_COMPRESS.
                }
                let v0 = nbytes;
                let prev_a3 = self.core.reg(Register::A3);
                let prev_a3_ts = self.core.reg_timestamp(Register::A3);
                self.core.set_reg_aux(Register::A3, 0);
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id,
                    vec![MemoryReadRecord { value: nbytes, timestamp: clk, prev_timestamp: nbytes_ts }],
                    vec![MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts }],
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
                        syscall_id, arg1, arg2,
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
                let event = self.linux_event(
                    clk, arg1, arg2, v0, syscall_id, vec![],
                    vec![MemoryWriteRecord { value: 0, timestamp: clk, prev_value: prev_a3, prev_timestamp: prev_a3_ts }],
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
                        syscall_id, arg1, arg2,
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
        self.core.advance_clk_extra(extra_cycles);
        self.record.syscall_events.push(SyscallEvent {
            pc,
            next_pc,
            clk,
            a_record,
            a_record_is_real: true,
            b_record: Some(b_record),
            c_record: Some(c_record),
            syscall_id,
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
    use crate::{
        golden::run_golden,
        minimal::MinimalExecutor,
        programs::tests::{
            bls12381_add_program, bls12381_double_program, bls12381_fp2_addsub_program,
            bls12381_fp2_mul_program, bls12381_fp_program, bn254_add_program, bn254_double_program,
            bn254_fp2_addsub_program, bn254_fp2_mul_program, bn254_fp_program, ed_add_program,
            ed_decompress_program, fibonacci_program, halt_only_program, hello_world_program,
            poseidon2_permute_program, secp256k1_add_program, secp256k1_double_program,
            secp256r1_add_program, secp256r1_double_program, sha_compress_program,
            sha_extend_program, simple_program, ssz_withdrawals_program, u256xu2048_mul_program,
            uint256_mul_program,
        },
        register::NUM_REGISTERS,
    };
    use std::collections::BTreeMap;
    use zkm_hypercube::record::MachineRecord;

    /// Fields this scoped `TracingVM` doesn't populate yet (register/memory-timestamp
    /// bookkeeping the AIR's memory-consistency argument needs, and the postprocess-derived
    /// global-memory-init/finalize/lookup events) -- see the module doc. Excluded from the
    /// count comparison below; every other field must match the legacy `Executor` exactly.
    const DEFERRED_FIELDS: &[&str] = &[
        "bump_memory_events",
        "local_memory_access_events",
        "global_memory_initialize_events",
        "global_memory_finalize_events",
        "byte_lookups",
    ];

    fn run_tracing(program: Program) -> (BTreeMap<String, usize>, [u32; NUM_REGISTERS], u32, u64) {
        let program = Arc::new(program);
        let mut minimal = MinimalExecutor::new(program.clone(), u64::MAX / 2);
        let chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");
        let max_syscall_cycles = minimal.max_syscall_cycles();

        let mut record = ExecutionRecord::new(program.clone());
        let mut tracing = TracingVM::new(&chunk, program, max_syscall_cycles, &mut record);
        let status = tracing.execute().unwrap();
        assert_eq!(status, CoreVMStatus::Done, "expected the whole run to fit in one shard");
        let (registers, pc, clk) = (tracing.registers(), tracing.pc(), tracing.clk());

        let counts: BTreeMap<String, usize> = record.stats().into_iter().collect();
        (counts, registers, pc, clk)
    }

    fn assert_matches_golden(program: impl Fn() -> Program, name: &str) {
        let golden = run_golden(program());
        let (counts, registers, pc, clk) = run_tracing(program());

        let mut golden_counts = golden.event_counts;
        for field in DEFERRED_FIELDS {
            golden_counts.remove(*field);
        }
        assert_eq!(counts, golden_counts, "{name}: event counts mismatch");
        assert_eq!(registers, golden.final_registers, "{name}: final registers mismatch");
        assert_eq!(pc, golden.final_pc, "{name}: final pc mismatch");
        assert_eq!(clk, golden.final_clk, "{name}: final clk mismatch");
    }

    #[test]
    fn matches_golden_simple_program() {
        assert_matches_golden(simple_program, "simple_program");
    }

    #[test]
    fn matches_golden_halt_only_program() {
        assert_matches_golden(halt_only_program, "halt_only_program");
    }

    #[test]
    fn matches_golden_fibonacci_real_elf() {
        assert_matches_golden(fibonacci_program, "fibonacci_program");
    }

    #[test]
    fn matches_golden_hello_world_real_elf() {
        assert_matches_golden(hello_world_program, "hello_world_program");
    }

    #[test]
    fn matches_golden_sha_extend_real_elf() {
        assert_matches_golden(sha_extend_program, "sha_extend_program");
    }

    #[test]
    fn matches_golden_sha_compress_real_elf() {
        assert_matches_golden(sha_compress_program, "sha_compress_program");
    }

    #[test]
    fn matches_golden_keccak_sponge_real_elf() {
        assert_matches_golden(ssz_withdrawals_program, "ssz_withdrawals_program");
    }

    #[test]
    fn matches_golden_secp256r1_add_real_elf() {
        assert_matches_golden(secp256r1_add_program, "secp256r1_add_program");
    }

    #[test]
    fn matches_golden_secp256r1_double_real_elf() {
        assert_matches_golden(secp256r1_double_program, "secp256r1_double_program");
    }

    // No differential test for `*_decompress_program`: those ELFs read a compressed point from
    // stdin (see `zkm-core-machine`'s `weierstrass_decompress` tests, which generate one via
    // `k256`/`p256`), and this crate has no dependency on those key-generation crates or on the
    // stdin-plumbing helpers (`ZKMStdin`/`run_test_io`) used to feed it -- `run_golden` only runs
    // a program with an empty input stream. `ec_decompress_dispatch`/`ec_decompress_event` share
    // all their oracle-consumption-ordering machinery with `ec_add`/`ec_double` above, which *are*
    // covered.

    #[test]
    fn matches_golden_secp256k1_add_real_elf() {
        assert_matches_golden(secp256k1_add_program, "secp256k1_add_program");
    }

    #[test]
    fn matches_golden_secp256k1_double_real_elf() {
        assert_matches_golden(secp256k1_double_program, "secp256k1_double_program");
    }

    #[test]
    fn matches_golden_bn254_add_real_elf() {
        assert_matches_golden(bn254_add_program, "bn254_add_program");
    }

    #[test]
    fn matches_golden_bn254_double_real_elf() {
        assert_matches_golden(bn254_double_program, "bn254_double_program");
    }

    #[test]
    fn matches_golden_bls12381_add_real_elf() {
        assert_matches_golden(bls12381_add_program, "bls12381_add_program");
    }

    #[test]
    fn matches_golden_bls12381_double_real_elf() {
        assert_matches_golden(bls12381_double_program, "bls12381_double_program");
    }

    #[test]
    fn matches_golden_ed_add_real_elf() {
        assert_matches_golden(ed_add_program, "ed_add_program");
    }

    #[test]
    fn matches_golden_ed_decompress_real_elf() {
        assert_matches_golden(ed_decompress_program, "ed_decompress_program");
    }

    #[test]
    fn matches_golden_bn254_fp_real_elf() {
        assert_matches_golden(bn254_fp_program, "bn254_fp_program");
    }

    #[test]
    fn matches_golden_bn254_fp2_addsub_real_elf() {
        assert_matches_golden(bn254_fp2_addsub_program, "bn254_fp2_addsub_program");
    }

    #[test]
    fn matches_golden_bn254_fp2_mul_real_elf() {
        assert_matches_golden(bn254_fp2_mul_program, "bn254_fp2_mul_program");
    }

    #[test]
    fn matches_golden_bls12381_fp_real_elf() {
        assert_matches_golden(bls12381_fp_program, "bls12381_fp_program");
    }

    #[test]
    fn matches_golden_bls12381_fp2_addsub_real_elf() {
        assert_matches_golden(bls12381_fp2_addsub_program, "bls12381_fp2_addsub_program");
    }

    #[test]
    fn matches_golden_bls12381_fp2_mul_real_elf() {
        assert_matches_golden(bls12381_fp2_mul_program, "bls12381_fp2_mul_program");
    }

    #[test]
    fn matches_golden_uint256_mul_real_elf() {
        assert_matches_golden(uint256_mul_program, "uint256_mul_program");
    }

    #[test]
    fn matches_golden_u256xu2048_mul_real_elf() {
        assert_matches_golden(u256xu2048_mul_program, "u256xu2048_mul_program");
    }

    #[test]
    fn matches_golden_poseidon2_permute_real_elf() {
        assert_matches_golden(poseidon2_permute_program, "poseidon2_permute_program");
    }

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
}
