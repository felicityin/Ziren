//! `MinimalExecutor`: `MinimalRunner`'s own, independent instruction-execution engine.
//!
//! Runs the program for real -- real memory, real registers, real syscalls (side effects like
//! stdout/`COMMIT` happen for real, exactly once) -- and buffers the resulting `MemValue` stream.
//! Unlike `CoreVM<Oracle>` (shared by `SplicingVM`/`TracingVM`), it builds no `ExecutionRecord`
//! content and no typed AIR event: no `local_memory_access`, no `local_counts`, no
//! `MemoryAccessRecord`. `record_mut()` still returns a scratch `ExecutionRecord` (required by
//! `SyscallRuntime`'s signature, since syscalls like `COMMIT` write straight into
//! `record.public_values`), but nothing else ever inspects it.

use std::{collections::VecDeque, sync::Arc};

use hashbrown::HashMap;

use crate::{
    events::{
        MemoryAccessPosition, MemoryLocalEvent, MemoryReadRecord, MemoryRecordEnum,
        MemoryWriteRecord, SyscallEvent,
    },
    memory::{Entry, PagedMemory},
    program::MAX_MEMORY,
    record::ExecutionRecord,
    register::NUM_REGISTERS,
    syscalls::{default_syscall_map, Syscall, SyscallCode, SyscallContext, SyscallRuntime},
    utils::sign_extend as sign_extend_fn,
    vm::MemValue,
    ExecutionError, Instruction, Opcode, Program, Register,
};

/// Snapshot taken on entering an unconstrained block, restored on exit.
#[derive(Debug, Clone, Copy)]
struct UnconstrainedCtx {
    pc: u32,
    clk: u64,
    global_clk: u64,
}

pub struct MinimalExecutor {
    pub program: Arc<Program>,
    /// Registers -- kept out of `oracle_out` entirely, unlike `memory`: a register's value is
    /// always a deterministic function of the instruction stream and already-oracle-buffered
    /// memory reads, so a replay pass (`CoreVM<Oracle>`, see `vm.rs`) can recompute it locally
    /// instead of needing it pre-recorded.
    pub(crate) registers: [MemValue; NUM_REGISTERS],
    /// Whether each register has ever been set (via `load_image` or a real access), mirroring
    /// `memory`'s vacant/occupied distinction -- `registers` itself has no such concept, since
    /// every slot is always populated. Read by `tracing_chunk::emit_globals` to decide which
    /// registers need a global finalize event at all.
    pub(crate) registers_touched: [bool; NUM_REGISTERS],
    pub(crate) memory: PagedMemory<MemValue>,
    pub(crate) uninitialized_memory: crate::memory::Memory<u32>,
    /// Memory diff since the matching `enter_unconstrained`, reverted on exit.
    unconstrained_mem_diff: Option<HashMap<u32, Option<MemValue>>>,
    /// Register diff since the matching `enter_unconstrained`, reverted on exit. Registers are
    /// always populated (no vacant/occupied distinction), so unlike the memory diff this only
    /// ever holds the previous `MemValue`, never `None`.
    unconstrained_reg_diff: Option<HashMap<u32, MemValue>>,

    pub pc: u32,
    pub next_pc: u32,
    pub clk: u64,
    pub global_clk: u64,
    pub exited: bool,
    unconstrained: bool,
    unconstrained_ctx: Option<UnconstrainedCtx>,
    next_is_delayslot: bool,

    /// The value+timestamp stream being produced, for general memory only (see `registers`).
    /// `MinimalRunner` drains this into a `Chunk` once it's full or the program halts.
    pub oracle_out: Vec<MemValue>,

    syscall_map: HashMap<SyscallCode, Arc<dyn Syscall<Self>>>,
    pub input_stream: VecDeque<Vec<u8>>,
    pub public_values_stream: Vec<u8>,
    io_buf: HashMap<u32, String>,
    cycle_tracker: HashMap<String, (u64, u32)>,
    /// Scratch record -- see the module doc comment.
    record: ExecutionRecord,
}

impl MinimalExecutor {
    #[must_use]
    pub fn new(program: Arc<Program>) -> Self {
        let pc = program.pc_start;
        let next_pc = program.next_pc;
        let record = ExecutionRecord::new(program.clone());
        let syscall_map = default_syscall_map::<Self>();
        Self {
            program,
            registers: [MemValue::default(); NUM_REGISTERS],
            registers_touched: [false; NUM_REGISTERS],
            memory: PagedMemory::new_preallocated(),
            uninitialized_memory: crate::memory::Memory::new_preallocated(),
            unconstrained_mem_diff: None,
            unconstrained_reg_diff: None,
            pc,
            next_pc,
            // 1, not 0: `0` is the "never touched" sentinel `MemoryRecord::timestamp`, matching
            // `SplicingVM`'s/`TracingVM`'s own convention for a program's first shard.
            clk: 1,
            global_clk: 0,
            exited: false,
            unconstrained: false,
            unconstrained_ctx: None,
            next_is_delayslot: false,
            oracle_out: Vec::new(),
            syscall_map,
            input_stream: VecDeque::new(),
            public_values_stream: Vec::new(),
            io_buf: HashMap::new(),
            cycle_tracker: HashMap::new(),
            record,
        }
    }

    pub fn load_image(&mut self) {
        for (&addr, value) in &self.program.image.clone() {
            if addr < NUM_REGISTERS as u32 {
                self.registers[addr as usize] = MemValue { clk: 0, value: *value };
                self.registers_touched[addr as usize] = true;
            } else {
                self.memory.insert(addr, MemValue { clk: 0, value: *value });
            }
        }
    }

    #[must_use]
    const fn timestamp(&self, position: MemoryAccessPosition) -> u64 {
        self.clk + position as u64
    }

    fn fetch(&self) -> Instruction {
        self.program.fetch(self.pc)
    }

    // ---- primitive accessors ----

    /// General-memory access -- pushes into `oracle_out` (see the `registers` field's doc
    /// comment for why registers don't).
    fn mem_access(&mut self, addr: u32) -> MemValue {
        let entry = self.memory.entry(addr);
        if let Some(diff) = self.unconstrained_mem_diff.as_mut() {
            let existing = match &entry {
                Entry::Occupied(entry) => Some(*entry.get()),
                Entry::Vacant(_) => None,
            };
            diff.entry(addr).or_insert(existing);
        }
        let record: &mut MemValue = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let value = self.uninitialized_memory.get(addr).copied().unwrap_or(0);
                entry.insert(MemValue { clk: 0, value })
            }
        };
        let prev_record = *record;
        self.oracle_out.push(prev_record);
        prev_record
    }

    fn mem_commit(&mut self, addr: u32, record: MemValue) {
        self.memory.insert(addr, record);
    }

    fn mr(&mut self, addr: u32, timestamp: u64) -> u32 {
        let prev_record = self.mem_access(addr);
        self.mem_commit(addr, MemValue { clk: timestamp, value: prev_record.value });
        prev_record.value
    }

    fn mw(&mut self, addr: u32, value: u32, timestamp: u64) {
        let _prev_record = self.mem_access(addr);
        self.mem_commit(addr, MemValue { clk: timestamp, value });
    }

    /// Register access -- never pushed into `oracle_out`.
    fn reg_access(&mut self, register: Register) -> MemValue {
        let idx = register as u32 as usize;
        if let Some(diff) = self.unconstrained_reg_diff.as_mut() {
            diff.entry(idx as u32).or_insert(self.registers[idx]);
        }
        self.registers_touched[idx] = true;
        self.registers[idx]
    }

    fn reg_commit(&mut self, register: Register, record: MemValue) {
        self.registers[register as u32 as usize] = record;
    }

    fn rr(&mut self, register: Register, position: MemoryAccessPosition) -> u32 {
        let prev_record = self.reg_access(register);
        let timestamp = self.timestamp(position);
        self.reg_commit(register, MemValue { clk: timestamp, value: prev_record.value });
        prev_record.value
    }

    fn rw(&mut self, register: Register, value: u32, position: MemoryAccessPosition) {
        let value = if register == Register::ZERO { 0 } else { value };
        let _prev_record = self.reg_access(register);
        let timestamp = self.timestamp(position);
        self.reg_commit(register, MemValue { clk: timestamp, value });
    }

    /// Get the current value of a register, without creating an access record.
    fn register(&mut self, register: Register) -> u32 {
        self.reg_access(register).value
    }

    /// Get the current value of a word, without creating an access record.
    fn word(&mut self, addr: u32) -> u32 {
        self.mem_access(addr).value
    }

    /// Get the current value of a byte, without creating an access record.
    fn byte(&mut self, addr: u32) -> u8 {
        let word = self.word(addr - addr % 4);
        (word >> ((addr % 4) * 8)) as u8
    }

    fn mr_cpu(&mut self, addr: u32) -> u32 {
        self.mr(addr, self.timestamp(MemoryAccessPosition::Memory))
    }

    fn mw_cpu(&mut self, addr: u32, value: u32) {
        let timestamp = self.timestamp(MemoryAccessPosition::Memory);
        self.mw(addr, value, timestamp);
    }

    // ---- instruction-family helpers, ported from `CoreVM::step`'s helpers of the same name ----

    fn alu_rr(&mut self, instruction: &Instruction) -> (Register, u32, u32) {
        if !instruction.imm_c {
            let (rd, rs1, rs2) = (
                instruction.op_a.into(),
                (instruction.op_b as u8).into(),
                (instruction.op_c as u8).into(),
            );
            let c = self.rr(rs2, MemoryAccessPosition::C);
            let b = self.rr(rs1, MemoryAccessPosition::B);
            (rd, b, c)
        } else if !instruction.imm_b && instruction.imm_c {
            let (rd, rs1, imm) =
                (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
            (rd, self.rr(rs1, MemoryAccessPosition::B), imm)
        } else {
            debug_assert!(instruction.imm_b && instruction.imm_c);
            (instruction.op_a.into(), instruction.op_b, instruction.op_c)
        }
    }

    fn alu_rw(&mut self, op: &Instruction, rd: Register, hi: u32, a: u32) {
        if op.opcode.is_use_lo_hi_alu() {
            self.rw(Register::LO, a, MemoryAccessPosition::A);
            self.rw(Register::HI, hi, MemoryAccessPosition::HI);
        } else {
            self.rw(rd, a, MemoryAccessPosition::A);
        }
    }

    fn branch_rr(&mut self, instruction: &Instruction) -> (u32, u32, u32) {
        let (src1, src2, target) =
            (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = if instruction.opcode.only_one_operand() {
            0
        } else {
            self.rr(src2, MemoryAccessPosition::B)
        };
        let a = self.rr(src1, MemoryAccessPosition::A);
        (a, b, target)
    }

    fn execute_alu(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let (rd, b, c) = self.alu_rr(instruction);
        if matches!(instruction.opcode, Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU) && c == 0 {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        let (a, hi) = match instruction.opcode {
            Opcode::ADD => (b.overflowing_add(c).0, 0),
            Opcode::SUB => (b.overflowing_sub(c).0, 0),
            Opcode::SLL => (b << (c & 0x1f), 0),
            Opcode::SRL => (b >> (c & 0x1F), 0),
            Opcode::SRA => {
                let sin = b as i32;
                let sout = sin >> (c & 0x1f);
                (sout as u32, 0)
            }
            Opcode::ROR => {
                let sin = (b as u64) + ((b as u64) << 32);
                let sout = sin >> (c & 0x1f);
                (sout as u32, 0)
            }
            Opcode::MUL => (b.overflowing_mul(c).0, 0),
            Opcode::SLTU => {
                if b < c { (1, 0) } else { (0, 0) }
            }
            Opcode::SLT => {
                if (b as i32) < (c as i32) { (1, 0) } else { (0, 0) }
            }
            Opcode::MULT => {
                let out = (((b as i32) as i64) * ((c as i32) as i64)) as u64;
                (out as u32, (out >> 32) as u32)
            }
            Opcode::MULTU => {
                let out = b as u64 * c as u64;
                (out as u32, (out >> 32) as u32)
            }
            Opcode::DIV => (((b as i32) / (c as i32)) as u32, ((b as i32) % (c as i32)) as u32),
            Opcode::DIVU => (b / c, b % c),
            Opcode::MOD => (((b as i32) % (c as i32)) as u32, 0),
            Opcode::MODU => (b % c, 0),
            Opcode::AND => (b & c, 0),
            Opcode::OR => (b | c, 0),
            Opcode::XOR => (b ^ c, 0),
            Opcode::NOR => (!(b | c), 0),
            Opcode::CLZ => (b.leading_zeros(), 0),
            Opcode::CLO => (b.leading_ones(), 0),
            _ => unreachable!(),
        };
        self.alu_rw(instruction, rd, hi, a);
        Ok(())
    }

    fn execute_load(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let (rt_reg, rs_reg, offset_ext) =
            (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let rs_raw = self.rr(rs_reg, MemoryAccessPosition::B);
        let rt = self.register(rt_reg);

        let addr = rs_raw.wrapping_add(offset_ext);
        let aligned_addr = addr & 0xFFFF_FFFC;

        if aligned_addr + 3 > MAX_MEMORY as u32 {
            return Err(ExecutionError::MemoryOutOfBoundsAccess(addr as u64));
        }

        let mem = self.mr_cpu(aligned_addr);
        let rs = addr;

        let val = match instruction.opcode {
            Opcode::LH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LH, addr));
                }
                let mem_fc = |i: u32| -> u32 { sign_extend_fn::<16>((mem >> (i * 8)) & 0xffff) };
                mem_fc(rs & 2)
            }
            Opcode::LWL => {
                let out = |i: u32| -> u32 {
                    let val = mem << (24 - i * 8);
                    let mask: u32 = 0xFFFFFFFF_u32 << (24 - i * 8);
                    (rt & (!mask)) | val
                };
                out(rs & 3)
            }
            Opcode::LW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LW, addr));
                }
                mem
            }
            Opcode::LBU => {
                let out = |i: u32| -> u32 { (mem >> (i * 8)) & 0xff };
                out(rs & 3)
            }
            Opcode::LHU => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LHU, addr));
                }
                let mem_fc = |i: u32| -> u32 { (mem >> (i * 8)) & 0xffff };
                mem_fc(rs & 2)
            }
            Opcode::LWR => {
                let out = |i: u32| -> u32 {
                    let val = mem >> (i * 8);
                    let mask = 0xFFFFFFFF_u32 >> (i * 8);
                    (rt & (!mask)) | val
                };
                out(rs & 3)
            }
            Opcode::LL => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LL, addr));
                }
                mem
            }
            Opcode::LB => {
                let out = |i: u32| -> u32 { sign_extend_fn::<8>((mem >> (i * 8)) & 0xff) };
                out(rs & 3)
            }
            _ => unreachable!(),
        };
        self.rw(rt_reg, val, MemoryAccessPosition::A);
        Ok(())
    }

    fn execute_store(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let (rt_reg, rs_reg, offset_ext) =
            (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let rs = self.rr(rs_reg, MemoryAccessPosition::B);
        let rt = if instruction.opcode == Opcode::SC {
            self.register(rt_reg)
        } else {
            self.rr(rt_reg, MemoryAccessPosition::A)
        };

        let addr = rs.wrapping_add(offset_ext);
        let aligned_addr = addr & 0xFFFF_FFFC;

        let mem = self.word(aligned_addr);

        let val = match instruction.opcode {
            Opcode::SB => {
                let out = |i: u32| -> u32 {
                    let val = (rt & 0xff) << (i * 8);
                    let mask = 0xFFFFFFFF_u32 ^ (0xff << (i * 8));
                    (mem & mask) | val
                };
                out(addr & 3)
            }
            Opcode::SH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SH, addr));
                }
                let mem_fc = |i: u32| -> u32 {
                    let val = (rt & 0xffff) << (i * 8);
                    let mask = 0xFFFFFFFF_u32 ^ (0xffff << (i * 8));
                    (mem & mask) | val
                };
                mem_fc(addr & 2)
            }
            Opcode::SWL => {
                let out = |i: u32| -> u32 {
                    let val = rt >> (24 - i * 8);
                    let mask = 0xFFFFFFFF_u32 >> (24 - i * 8);
                    (mem & (!mask)) | val
                };
                out(addr & 3)
            }
            Opcode::SW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SW, addr));
                }
                rt
            }
            Opcode::SWR => {
                let out = |i: u32| -> u32 {
                    let val = rt << (i * 8);
                    let mask = 0xFFFFFFFF_u32 << (i * 8);
                    (mem & (!mask)) | val
                };
                out(addr & 3)
            }
            Opcode::SC => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SC, addr));
                }
                rt
            }
            _ => unreachable!("unexpected store opcode: {:?}", instruction.opcode),
        };

        if aligned_addr + 3 > MAX_MEMORY as u32 {
            return Err(ExecutionError::MemoryOutOfBoundsAccess(addr as u64));
        }

        self.mw_cpu(aligned_addr, val);
        if instruction.opcode == Opcode::SC {
            self.rw(rt_reg, 1, MemoryAccessPosition::A);
        }
        Ok(())
    }

    fn execute_branch(&mut self, instruction: &Instruction, next_pc: u32, mut next_next_pc: u32) -> u32 {
        let (src1, src2, offset) = self.branch_rr(instruction);
        let should_jump = match instruction.opcode {
            Opcode::BEQ => src1 == src2,
            Opcode::BNE => src1 != src2,
            Opcode::BGEZ => (src1 as i32) >= 0,
            Opcode::BLEZ => (src1 as i32) <= 0,
            Opcode::BGTZ => (src1 as i32) > 0,
            Opcode::BLTZ => (src1 as i32) < 0,
            _ => unreachable!(),
        };
        if should_jump {
            next_next_pc = offset.wrapping_add(next_pc);
        }
        next_next_pc
    }

    fn execute_jump(&mut self, instruction: &Instruction) -> u32 {
        let (link, target) = (instruction.op_a.into(), (instruction.op_b as u8).into());
        let target_pc = self.rr(target, MemoryAccessPosition::B);
        let return_pc = self.next_pc.wrapping_add(4);
        self.rw(link, return_pc, MemoryAccessPosition::A);
        target_pc
    }

    fn execute_jumpi(&mut self, instruction: &Instruction) -> u32 {
        let (link, target_pc) = (instruction.op_a.into(), instruction.op_b);
        let return_pc = self.next_pc.wrapping_add(4);
        self.rw(link, return_pc, MemoryAccessPosition::A);
        target_pc
    }

    fn execute_jump_direct(&mut self, instruction: &Instruction) -> u32 {
        let (link, offset) = (instruction.op_a.into(), instruction.op_b);
        let target_pc = offset.wrapping_add(self.next_pc);
        let return_pc = self.next_pc.wrapping_add(4);
        self.rw(link, return_pc, MemoryAccessPosition::A);
        target_pc
    }

    fn execute_condmov(&mut self, instruction: &Instruction) {
        let (rd, rs, rt) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let a = self.register(rd);
        let c = self.rr(rt, MemoryAccessPosition::C);
        let b = self.rr(rs, MemoryAccessPosition::B);
        let mov = match instruction.opcode {
            Opcode::MEQ => c == 0,
            Opcode::MNE => c != 0,
            _ => unreachable!(),
        };
        let a = if mov { b } else { a };
        self.rw(rd, a, MemoryAccessPosition::A);
    }

    fn execute_wsbh(&mut self, instruction: &Instruction) {
        let (rd, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());
        let b = self.rr(rt, MemoryAccessPosition::B);
        let a = (((b >> 16) & 0xFF) << 24)
            | (((b >> 24) & 0xFF) << 16)
            | ((b & 0xFF) << 8)
            | ((b >> 8) & 0xFF);
        self.rw(rd, a, MemoryAccessPosition::A);
    }

    fn execute_ext(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = self.rr(rt, MemoryAccessPosition::B);
        let msbd = c >> 5;
        let lsb = c & 0x1f;
        if msbd + lsb >= 32 {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        let mask_msb = if msbd + lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msbd + lsb + 1)) - 1 };
        let a = (b & mask_msb) >> lsb;
        self.rw(rd, a, MemoryAccessPosition::A);
        Ok(())
    }

    fn execute_ins(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = self.rr(rt, MemoryAccessPosition::B);
        let a = self.register(rd);
        let msb = c >> 5;
        let lsb = c & 0x1f;
        if msb < lsb {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        let mask = if msb - lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msb - lsb + 1)) - 1 };
        let mask_field = mask << lsb;
        let a = (a & !mask_field) | ((b << lsb) & mask_field);
        self.rw(rd, a, MemoryAccessPosition::A);
        Ok(())
    }

    fn execute_sext(&mut self, instruction: &Instruction) {
        let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
        let b = self.rr(rt, MemoryAccessPosition::B);
        let a = if c > 0 { (b & 0xffff) as i16 as i32 as u32 } else { (b & 0xff) as i8 as i32 as u32 };
        self.rw(rd, a, MemoryAccessPosition::A);
    }

    fn execute_teq(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let (rs, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());
        let src2 = self.rr(rt, MemoryAccessPosition::B);
        let src1 = self.rr(rs, MemoryAccessPosition::A);
        if src1 == src2 {
            return Err(ExecutionError::ExceptionOrTrap());
        }
        Ok(())
    }

    fn execute_maddu(&mut self, instruction: &Instruction) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr(rs, MemoryAccessPosition::C);
        let b = self.rr(rt, MemoryAccessPosition::B);
        let multiply = b as u64 * c as u64;
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = multiply.wrapping_add(addend);
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw(lo, out_lo, MemoryAccessPosition::A);
        self.rw(Register::HI, out_hi, MemoryAccessPosition::HI);
    }

    fn execute_msubu(&mut self, instruction: &Instruction) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr(rs, MemoryAccessPosition::C);
        let b = self.rr(rt, MemoryAccessPosition::B);
        let multiply = b as u64 * c as u64;
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = addend.wrapping_sub(multiply);
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw(lo, out_lo, MemoryAccessPosition::A);
        self.rw(Register::HI, out_hi, MemoryAccessPosition::HI);
    }

    fn execute_madd(&mut self, instruction: &Instruction) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr(rs, MemoryAccessPosition::C);
        let b = self.rr(rt, MemoryAccessPosition::B);
        let multiply = (b as i32 as i64) * (c as i32 as i64);
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = multiply.wrapping_add(addend as i64) as u64;
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw(lo, out_lo, MemoryAccessPosition::A);
        self.rw(Register::HI, out_hi, MemoryAccessPosition::HI);
    }

    fn execute_msub(&mut self, instruction: &Instruction) {
        let (lo, rt, rs) = (
            instruction.op_a.into(),
            (instruction.op_b as u8).into(),
            (instruction.op_c as u8).into(),
        );
        let c = self.rr(rs, MemoryAccessPosition::C);
        let b = self.rr(rt, MemoryAccessPosition::B);
        let multiply = (b as i32 as i64) * (c as i32 as i64);
        let lo_val = self.register(32.into());
        let hi_val = self.register(33.into());
        let addend = ((hi_val as u64) << 32) + lo_val as u64;
        let out = (addend as i64).wrapping_sub(multiply) as u64;
        let (out_lo, out_hi) = (out as u32, (out >> 32) as u32);
        self.rw(lo, out_lo, MemoryAccessPosition::A);
        self.rw(Register::HI, out_hi, MemoryAccessPosition::HI);
    }

    fn get_syscall(&self, code: SyscallCode) -> Option<Arc<dyn Syscall<Self>>> {
        self.syscall_map.get(&code).cloned()
    }

    /// Execute the next instruction at the current pc. Returns `true` once the program is done
    /// (halted, or ran off the end of the program).
    ///
    /// # Errors
    /// Returns an error if execution fails (invalid instruction, out-of-bounds access, etc).
    pub fn execute_instruction(&mut self) -> Result<bool, ExecutionError> {
        let instruction = self.fetch();

        let mut next_pc = self.next_pc;
        let mut next_next_pc = self.next_pc + 4;

        self.next_is_delayslot = false;

        if instruction.is_alu_instruction() {
            self.execute_alu(&instruction)?;
        } else if instruction.is_memory_load_instruction() {
            self.execute_load(&instruction)?;
        } else if instruction.is_memory_store_instruction() {
            self.execute_store(&instruction)?;
        } else if instruction.is_branch_instruction() {
            next_next_pc = self.execute_branch(&instruction, next_pc, next_next_pc);
            self.next_is_delayslot = true;
        } else if instruction.is_jump_instruction() {
            next_next_pc = if instruction.opcode == Opcode::Jump {
                self.execute_jump(&instruction)
            } else if instruction.opcode == Opcode::Jumpi {
                self.execute_jumpi(&instruction)
            } else {
                self.execute_jump_direct(&instruction)
            };
            self.next_is_delayslot = true;
        } else if instruction.is_mov_cond_instruction() {
            self.execute_condmov(&instruction);
        } else if instruction.is_misc_instruction() {
            if instruction.opcode == Opcode::WSBH {
                self.execute_wsbh(&instruction);
            } else if instruction.opcode == Opcode::EXT {
                self.execute_ext(&instruction)?;
            } else if instruction.opcode == Opcode::MADDU {
                self.execute_maddu(&instruction);
            } else if instruction.opcode == Opcode::INS {
                self.execute_ins(&instruction)?;
            } else if instruction.opcode == Opcode::SEXT {
                self.execute_sext(&instruction);
            } else if instruction.opcode == Opcode::TEQ {
                self.execute_teq(&instruction)?;
            } else if instruction.opcode == Opcode::MSUBU {
                self.execute_msubu(&instruction);
            } else if instruction.opcode == Opcode::MADD {
                self.execute_madd(&instruction);
            } else if instruction.opcode == Opcode::MSUB {
                self.execute_msub(&instruction);
            }
        } else if instruction.opcode == Opcode::SYSCALL {
            let syscall_id = self.register(Register::V0);
            let c = self.rr(Register::A1, MemoryAccessPosition::C);
            let b = self.rr(Register::A0, MemoryAccessPosition::B);
            let syscall = SyscallCode::from_u32(syscall_id);

            if self.unconstrained
                && (syscall != SyscallCode::EXIT_UNCONSTRAINED && syscall != SyscallCode::WRITE)
            {
                return Err(ExecutionError::InvalidSyscallUsage(syscall_id as u64));
            }

            let syscall_impl = self.get_syscall(syscall);
            let mut precompile_rt = SyscallContext::new(self);
            let (precompile_next_pc, precompile_cycles, returned_exit_code, prev_a) =
                if let Some(syscall_impl) = syscall_impl {
                    let res = syscall_impl.execute(&mut precompile_rt, syscall, b, c)?;
                    let a = res.unwrap_or(syscall_id);

                    if syscall == SyscallCode::HALT && precompile_rt.exit_code != 0 {
                        return Err(ExecutionError::HaltWithNonZeroExitCode(precompile_rt.exit_code));
                    }

                    (precompile_rt.next_pc, syscall_impl.num_extra_cycles(), precompile_rt.exit_code, a)
                } else {
                    return Err(ExecutionError::UnsupportedSyscall(syscall_id));
                };

            if syscall == SyscallCode::HALT && returned_exit_code == 0 {
                self.exited = true;
            }

            let a = if syscall == SyscallCode::EXIT_UNCONSTRAINED {
                self.register(Register::V0)
            } else {
                prev_a
            };

            self.rw(Register::V0, a, MemoryAccessPosition::A);
            next_pc = precompile_next_pc;
            next_next_pc = precompile_next_pc + 4;
            self.clk += u64::from(precompile_cycles);
        } else if instruction.opcode == Opcode::UNIMPL {
            return Err(ExecutionError::UnsupportedInstruction(instruction.op_c));
        } else {
            unreachable!()
        }

        if next_next_pc == 0 {
            return Err(ExecutionError::NullPointerReference());
        }

        self.pc = next_pc;
        self.next_pc = next_next_pc;
        self.clk += 5;
        self.global_clk += 1;

        let done = self.pc == 0
            || self.exited
            || self.pc.wrapping_sub(self.program.pc_base) >= (self.program.instructions.len() * 4) as u32;
        if done && self.unconstrained {
            return Err(ExecutionError::EndInUnconstrained());
        }

        Ok(done)
    }
}

impl SyscallRuntime for MinimalExecutor {
    fn shard(&self) -> u32 {
        1
    }

    fn clk(&self) -> u64 {
        self.clk
    }

    fn timestamp(&self) -> u64 {
        self.clk
    }

    fn pc(&self) -> u32 {
        self.pc
    }

    fn is_unconstrained(&self) -> bool {
        self.unconstrained
    }

    fn program(&self) -> &Program {
        &self.program
    }

    fn global_clk(&self) -> u64 {
        self.global_clk
    }

    fn mr(
        &mut self,
        addr: u32,
        _external: bool,
        clk: u64,
        _local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        // Syscalls always address data buffers, never registers directly, but branch here for
        // parity with `CoreVM::mr`'s equivalent guard rather than assuming that's never hit.
        if addr < NUM_REGISTERS as u32 {
            let prev_record = self.reg_access((addr as u8).into());
            self.reg_commit((addr as u8).into(), MemValue { clk, value: prev_record.value });
            return MemoryReadRecord::new(prev_record.value, clk, prev_record.clk);
        }
        let prev_record = self.mem_access(addr);
        self.mem_commit(addr, MemValue { clk, value: prev_record.value });
        MemoryReadRecord::new(prev_record.value, clk, prev_record.clk)
    }

    fn mw(
        &mut self,
        addr: u32,
        value: u32,
        _external: bool,
        clk: u64,
        _local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        if addr < NUM_REGISTERS as u32 {
            let prev_record = self.reg_access((addr as u8).into());
            self.reg_commit((addr as u8).into(), MemValue { clk, value });
            return MemoryWriteRecord::new(value, clk, prev_record.value, prev_record.clk);
        }
        let prev_record = self.mem_access(addr);
        self.mem_commit(addr, MemValue { clk, value });
        MemoryWriteRecord::new(value, clk, prev_record.value, prev_record.clk)
    }

    fn rr_traced(
        &mut self,
        register: Register,
        clk: u64,
        _local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        let prev_record = self.reg_access(register);
        self.reg_commit(register, MemValue { clk, value: prev_record.value });
        MemoryReadRecord::new(prev_record.value, clk, prev_record.clk)
    }

    fn rw_traced(
        &mut self,
        register: Register,
        value: u32,
        clk: u64,
        _local_memory_access: Option<&mut HashMap<u32, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        let value = if register == Register::ZERO { 0 } else { value };
        let prev_record = self.reg_access(register);
        self.reg_commit(register, MemValue { clk, value });
        MemoryWriteRecord::new(value, clk, prev_record.value, prev_record.clk)
    }

    fn register(&mut self, register: Register) -> u32 {
        MinimalExecutor::register(self, register)
    }

    fn word(&mut self, addr: u32) -> u32 {
        MinimalExecutor::word(self, addr)
    }

    fn byte(&mut self, addr: u32) -> u8 {
        MinimalExecutor::byte(self, addr)
    }

    fn record_mut(&mut self) -> &mut ExecutionRecord {
        &mut self.record
    }

    fn is_recording_events(&self) -> bool {
        false
    }

    fn outer_local_memory_access(&mut self) -> &mut HashMap<u32, MemoryLocalEvent> {
        // Never actually consulted: `SyscallContext::postprocess` only reads this when
        // `is_recording_events()` is true, which is never the case here. A real (empty, per-call)
        // map would work identically; this is only reachable if that invariant is ever violated.
        unreachable!("MinimalExecutor never records events, so postprocess never runs")
    }

    fn syscall_event(
        &self,
        clk: u64,
        a_record: Option<MemoryRecordEnum>,
        next_pc: u32,
        syscall_id: u32,
        arg1: u32,
        arg2: u32,
    ) -> SyscallEvent {
        let (write, is_real) = match a_record {
            Some(MemoryRecordEnum::Write(record)) => (record, true),
            _ => (MemoryWriteRecord::default(), false),
        };
        SyscallEvent {
            pc: self.pc,
            next_pc,
            shard: 1,
            clk,
            a_record: write,
            a_record_is_real: is_real,
            b_record: None,
            c_record: None,
            syscall_id,
            arg1,
            arg2,
        }
    }

    fn enter_unconstrained(&mut self) {
        assert!(!self.unconstrained, "Unconstrained block is already active.");
        self.unconstrained = true;
        self.unconstrained_ctx =
            Some(UnconstrainedCtx { pc: self.pc, clk: self.clk, global_clk: self.global_clk });
        assert!(
            self.unconstrained_mem_diff.is_none() && self.unconstrained_reg_diff.is_none(),
            "Unconstrained block is already active."
        );
        self.unconstrained_mem_diff = Some(HashMap::default());
        self.unconstrained_reg_diff = Some(HashMap::default());
    }

    fn exit_unconstrained(&mut self) {
        if let Some(ctx) = self.unconstrained_ctx.take() {
            self.pc = ctx.pc;
            self.clk = ctx.clk;
            self.global_clk = ctx.global_clk;
            if let Some(diff) = self.unconstrained_mem_diff.take() {
                for (addr, value) in diff {
                    match value {
                        Some(record) => {
                            self.memory.insert(addr, record);
                        }
                        None => {
                            self.memory.remove(addr);
                        }
                    }
                }
            }
            if let Some(diff) = self.unconstrained_reg_diff.take() {
                for (addr, record) in diff {
                    self.registers[addr as usize] = record;
                }
            }
            self.unconstrained = false;
        }
    }

    fn seed_uninitialized(&mut self, addr: u32, value: u32) -> Result<(), ExecutionError> {
        match self.uninitialized_memory.entry(addr) {
            Entry::Occupied(_) => {
                log::error!("hint read address is initialized already");
                Err(ExecutionError::InvalidSyscallArgs())
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
        }
    }

    /// `SYSHINTLEN`'s full behavior. Unlike `Executor`, also buffers the resolved length into
    /// `oracle_out` as a synthetic `MemValue` entry, interleaved in true chronological order with
    /// the real memory-access entries already pushed there from the same single-threaded
    /// execution loop -- this is what lets oracle-sourced runtimes (`CoreVM<Oracle>`, used by
    /// `SplicingVM`/`TracingVM`) resolve the same value deterministically without a live queue of
    /// their own (see `crate::vm::MemSource::next_raw`). `clk` on the entry is otherwise unused --
    /// there's no real address for a "previous vs new" comparison here -- but kept for
    /// consistency/debuggability with every other `oracle_out` entry.
    fn resolve_hint_len(&mut self) -> Result<u32, ExecutionError> {
        let Some(item) = self.input_stream.front() else {
            log::error!("failed reading stdin due to insufficient input data");
            return Err(ExecutionError::InvalidSyscallArgs());
        };
        let len = item.len() as u32;
        self.oracle_out.push(MemValue { clk: self.clk, value: len });
        Ok(len)
    }

    /// `SYSHINTREAD`'s full behavior -- the resolved bytes reach the oracle stream via the
    /// guest's own subsequent genuine load of `ptr` (an ordinary `mem_access`/`mem_commit` call),
    /// not through this method directly; `seed_uninitialized` only establishes the first-touch
    /// value that load sees.
    fn resolve_hint_read(&mut self, ptr: u32, len: u32) -> Result<(), ExecutionError> {
        let Some(vec) = self.input_stream.pop_front() else {
            log::error!("failed reading stdin due to insufficient input data");
            return Err(ExecutionError::InvalidSyscallArgs());
        };
        if self.unconstrained {
            log::error!("hint read should not be used in a unconstrained block");
            return Err(ExecutionError::ExceptionOrTrap());
        }
        if vec.len() as u32 != len || !ptr.is_multiple_of(4) {
            log::error!(
                "Invalid hint read syscall arguments: ptr={}, len={}, vec_len={}",
                ptr,
                len,
                vec.len()
            );
            return Err(ExecutionError::InvalidSyscallArgs());
        }
        for i in (0..len).step_by(4) {
            let b1 = vec[i as usize];
            let b2 = vec.get(i as usize + 1).copied().unwrap_or(0);
            let b3 = vec.get(i as usize + 2).copied().unwrap_or(0);
            let b4 = vec.get(i as usize + 3).copied().unwrap_or(0);
            let word = u32::from_le_bytes([b1, b2, b3, b4]);
            self.seed_uninitialized(ptr + i, word)?;
        }
        Ok(())
    }

    fn push_hint_input(&mut self, bytes: Vec<u8>) {
        self.input_stream.push_front(bytes);
    }

    fn write_public_values(&mut self, bytes: &[u8]) {
        self.public_values_stream.extend_from_slice(bytes);
    }

    fn stdout_line(&mut self, line: &str) {
        println!("stdout: {line}");
    }

    fn stderr_line(&mut self, line: &str) {
        println!("stderr: {line}");
    }

    fn io_buf_push(&mut self, fd: u32, s: &str) -> Vec<String> {
        let entry = self.io_buf.entry(fd).or_default();
        entry.push_str(s);
        if entry.contains('\n') {
            let prev_buf = std::mem::take(entry);
            let mut lines = prev_buf.split('\n').collect::<Vec<&str>>();
            let last = lines.pop().unwrap_or("");
            *entry = last.to_string();
            lines.into_iter().map(std::string::ToString::to_string).collect::<Vec<String>>()
        } else {
            vec![]
        }
    }

    fn invoke_hook(&mut self, _fd: u32, _buf: &[u8]) -> Result<Option<Vec<Vec<u8>>>, ExecutionError> {
        Ok(None)
    }

    fn cycle_tracker_start(&mut self, name: &str) {
        let depth = self.cycle_tracker.len() as u32;
        self.cycle_tracker.insert(name.to_string(), (self.global_clk, depth));
    }

    fn cycle_tracker_end(&mut self, name: &str) -> Option<u64> {
        self.cycle_tracker.remove(name).map(|(start, _depth)| self.global_clk - start)
    }

    fn cycle_tracker_report(&mut self, _name: &str, _total_cycles: u64) {}

    fn verify_deferred_proof(&mut self, _vkey: [u32; 8], _pv_digest: [u32; 8]) -> Result<(), ExecutionError> {
        // Deferred-proof verification only needs to happen once; `MinimalExecutor` doesn't carry
        // a proof stream (no `generate_records` caller currently feeds one in).
        Ok(())
    }
}
