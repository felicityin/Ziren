use std::{
    sync::Arc,
};

use crate::{
    DEFAULT_PC_INC, ExecutionError, Instruction, Opcode, Program, Register, vm::{memory::{GuestMemory, config::RV32_REGISTER_AS}, state::VmExecState}
};

/// An executor for the MIPS zkVM.
///
/// The executor is responsible for executing a user program and tracing important events which
/// occur during execution (i.e., memory reads, alu operations, etc).
pub struct SimpleExecutor {
    /// The program.
    pub program: Arc<Program>,

    /// The state of the execution.
    pub state: VmExecState,
}

impl SimpleExecutor {
    /// Create a new [`VmExecutor`] from a program and options.
    #[must_use]
    pub fn new(program: Program) -> Self {
        // Create a shared reference to the program.
        let program = Arc::new(program);

        let memory = GuestMemory::new(&program.image);
        let state = VmExecState::new(program.pc_start, program.next_pc, memory);

        Self {
            program,
            state,
        }
    }

    pub fn run(&mut self) -> Result<(), ExecutionError> {
        while !self.execute_cycle()? {}
        Ok(())
    }

    /// Fetch the instruction at the current program counter.
    #[inline]
    fn fetch(&self) -> Instruction {
        self.program.fetch(self.state.pc)
    }

    /// Executes one cycle of the program, returning whether the program has finished.
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn execute_cycle(&mut self) -> Result<bool, ExecutionError> {
        // Fetch the instruction at the current program counter.
        let instruction = self.fetch();

        // Execute the instruction.
        self.execute_operation(&instruction)?;

        let done = self.state.pc == 0
            || self.state.exited
            || self.state.pc.wrapping_sub(self.program.pc_base)
                >= (self.program.instructions.len() * 4) as u32;
        Ok(done)
    }

    /// Execute the given instruction over the current state of the runtime.
    #[allow(clippy::too_many_lines)]
    fn execute_operation(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        // let mut pc = self.state.pc;
        // let mut exit_code = 0u32; // use in halt code

        // let mut next_pc = self.state.next_pc;
        // let mut next_next_pc = self.state.next_pc + 4;

        if instruction.is_alu_instruction() {
            self.execute_alu(instruction)?;
        // } else if instruction.is_memory_load_instruction() {
        //    self.execute_load(instruction)?;
        // } else if instruction.is_memory_store_instruction() {
        //    self.execute_store(instruction)?;
        // } else if instruction.is_branch_instruction() {
        //    self.execute_branch(instruction, next_pc, next_next_pc);
        // } else if instruction.is_jump_instruction() {
        //     // Jump instructions.
        //     if instruction.opcode == Opcode::Jump {
        //         self.execute_jump(instruction)
        //     } else if instruction.opcode == Opcode::Jumpi {
        //         self.execute_jumpi(instruction)
        //     } else {
        //         self.execute_jump_direct(instruction)
        //     };
        //     // self.state.next_is_delayslot = true;
        // } else if instruction.is_mov_cond_instruction() {
        //     self.execute_condmov(instruction);
        // } else if instruction.is_misc_instruction() {
        //     if instruction.opcode == Opcode::WSBH {
        //         self.execute_wsbh(instruction);
        //     } else if instruction.opcode == Opcode::EXT {
        //         self.execute_ext(instruction);
        //     } else if instruction.opcode == Opcode::MADDU {
        //         self.execute_maddu(instruction);
        //     } else if instruction.opcode == Opcode::INS {
        //         self.execute_ins(instruction);
        //     } else if instruction.opcode == Opcode::SEXT {
        //         self.execute_sext(instruction);
        //     } else if instruction.opcode == Opcode::TEQ {
        //         self.execute_teq(instruction)?;
        //     } else if instruction.opcode == Opcode::MSUBU {
        //         self.execute_msubu(instruction);
        //     } else if instruction.opcode == Opcode::MADD {
        //         self.execute_madd(instruction);
        //     } else if instruction.opcode == Opcode::MSUB {
        //         self.execute_msub(instruction);
        //     }
        // } else if instruction.opcode == Opcode::SYSCALL {
        //     self.execute_syscall()?;
        } else if instruction.opcode == Opcode::UNIMPL {
            log::error!("{:X}: {:X}", self.state.pc, instruction.op_c);
            return Err(ExecutionError::UnsupportedInstruction(instruction.op_c));
        } else {
            unreachable!()
        }

        // if next_next_pc == 0 {
        //     log::error!("Null pointer reference {:X}: {:X}", self.state.pc, instruction.op_c);
        //     return Err(ExecutionError::NullPointerReference());
        // }

        // // Update the program counter.
        // self.state.pc = next_pc;
        // self.state.next_pc = next_next_pc;
        Ok(())
    }

    fn execute_alu(
        &mut self,
        instruction: &Instruction,
    ) -> Result<(), ExecutionError> {
        let (rd, b, c) = self.alu_rr(instruction);
        if matches!(instruction.opcode, Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU)
            && c == 0
        {
            return Err(ExecutionError::ExceptionOrTrap());
        }

        let (a, hi) = match instruction.opcode {
            Opcode::ADD => (b.overflowing_add(c).0, 0),
            Opcode::SUB => (b.overflowing_sub(c).0, 0),

            Opcode::SLL => (b << (c & 0x1f), 0),
            Opcode::SRL => (b >> (c & 0x1F), 0),
            Opcode::SRA => {
                // same as SRA
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
                if b < c {
                    (1, 0)
                } else {
                    (0, 0)
                }
            }
            Opcode::SLT => {
                if (b as i32) < (c as i32) {
                    (1, 0)
                } else {
                    (0, 0)
                }
            }

            Opcode::MULT => {
                let out = (((b as i32) as i64) * ((c as i32) as i64)) as u64;
                (out as u32, (out >> 32) as u32) // lo,hi
            }
            Opcode::MULTU => {
                let out = b as u64 * c as u64;
                (out as u32, (out >> 32) as u32) //lo,hi
            }
            Opcode::DIV => (
                ((b as i32) / (c as i32)) as u32, // lo
                ((b as i32) % (c as i32)) as u32, // hi
            ),
            Opcode::DIVU => (b / c, b % c), //lo,hi
            Opcode::MOD => (((b as i32) % (c as i32)) as u32, 0),
            Opcode::MODU => (b % c, 0), //lo,hi
            Opcode::AND => (b & c, 0),
            Opcode::OR => (b | c, 0),
            Opcode::XOR => (b ^ c, 0),
            Opcode::NOR => (!(b | c), 0),
            Opcode::CLZ => (b.leading_zeros(), 0),
            Opcode::CLO => (b.leading_ones(), 0),
            _ => {
                unreachable!()
            }
        };

        self.alu_rw(instruction, rd, hi, a);

        self.state.pc = self.state.next_pc;
        self.state.next_pc += DEFAULT_PC_INC;
        Ok(())
    }

    // fn execute_load(
    //     &mut self,
    //     instruction: &Instruction,
    // ) -> Result<(Option<u32>, u32, u32, u32), ExecutionError> {
    //     let (rt_reg, rs_reg, offset_ext) =
    //         (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    //     let rs_raw = self.rr_cpu(rs_reg);
    //     // We needn't the memory access record here, because we will write to rt_reg,
    //     // and we could use the `prev_value` of the MemoryWriteRecord in the circuit.
    //     let rt = self.register(rt_reg);

    //     let addr = rs_raw.wrapping_add(offset_ext);
    //     let aligned_addr = addr & 0xFFFF_FFFC;

    //     let mem = self.mr_cpu(aligned_addr);
    //     let rs = addr;

    //     if aligned_addr + 3 > MAX_MEMORY as u32 {
    //         return Err(ExecutionError::MemoryOutOfBoundsAccess(addr as u64));
    //     }

    //     let val = match instruction.opcode {
    //         Opcode::LH => {
    //             if addr & 1 != 0 {
    //                 return Err(ExecutionError::InvalidMemoryAccess(Opcode::LH, addr));
    //             }
    //             let mem_fc = |i: u32| -> u32 { sign_extend::<16>((mem >> (i * 8)) & 0xffff) };
    //             mem_fc(rs & 2)
    //         }
    //         Opcode::LWL => {
    //             let out = |i: u32| -> u32 {
    //                 let val = mem << (24 - i * 8);
    //                 let mask: u32 = 0xFFFFFFFF_u32 << (24 - i * 8);
    //                 (rt & (!mask)) | val
    //             };
    //             out(rs & 3)
    //         }
    //         Opcode::LW => {
    //             if addr & 3 != 0 {
    //                 return Err(ExecutionError::InvalidMemoryAccess(Opcode::LW, addr));
    //             }
    //             mem
    //         }
    //         Opcode::LBU => {
    //             let out = |i: u32| -> u32 { (mem >> (i * 8)) & 0xff };
    //             out(rs & 3)
    //         }
    //         Opcode::LHU => {
    //             if addr & 1 != 0 {
    //                 return Err(ExecutionError::InvalidMemoryAccess(Opcode::LHU, addr));
    //             }
    //             let mem_fc = |i: u32| -> u32 { (mem >> (i * 8)) & 0xffff };
    //             mem_fc(rs & 2)
    //         }
    //         Opcode::LWR => {
    //             let out = |i: u32| -> u32 {
    //                 let val = mem >> (i * 8);
    //                 let mask = 0xFFFFFFFF_u32 >> (i * 8);
    //                 (rt & (!mask)) | val
    //             };
    //             out(rs & 3)
    //         }
    //         Opcode::LL => {
    //             if addr & 3 != 0 {
    //                 return Err(ExecutionError::InvalidMemoryAccess(Opcode::LL, addr));
    //             }
    //             mem
    //         }
    //         Opcode::LB => {
    //             let out = |i: u32| -> u32 { sign_extend::<8>((mem >> (i * 8)) & 0xff) };
    //             out(rs & 3)
    //         }
    //         _ => unreachable!(),
    //     };
    //     self.rw_cpu(rt_reg, val, MemoryAccessPosition::A);

    //     Ok((Some(rt), val, rs_raw, offset_ext))
    // }

    /// Fetch the destination register and input operand values for an ALU instruction.
    fn alu_rr(&mut self, instruction: &Instruction) -> (Register, u32, u32) {
        if !instruction.imm_c {
            let (rd, rs1, rs2) = (
                instruction.op_a.into(),
                (instruction.op_b as u8).into(),
                (instruction.op_c as u8).into(),
            );
            let c = self.rr_cpu(rs2);
            let b = self.rr_cpu(rs1);
            (rd, b, c)
        } else if !instruction.imm_b && instruction.imm_c {
            let (rd, rs1, imm) =
                (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
            let (rd, b, c) = (rd, self.rr_cpu(rs1), imm);
            (rd, b, c)
        } else {
            debug_assert!(instruction.imm_b && instruction.imm_c);
            let (rd, b, c) = (instruction.op_a.into(), instruction.op_b, instruction.op_c);
            (rd, b, c)
        }
    }

    /// Read a register.
    #[inline]
    pub fn rr_cpu(&mut self, register: Register) -> u32 {
        let rs = self.state.vm_read::<u8, 4>(RV32_REGISTER_AS, register as u32);
        u32::from_le_bytes(rs)
    }

    /// Set the destination register with the result and emit an ALU event.
    fn alu_rw(
        &mut self,
        op: &Instruction,
        rd: Register,
        hi: u32,
        a: u32,
    ) -> Option<u32> {
        let hi = if op.opcode.is_use_lo_hi_alu() {
            self.rw_cpu(Register::LO, a);
            self.rw_cpu(Register::HI, hi);
            Some(hi)
        } else {
            self.rw_cpu(rd, a);
            None
        };

        hi
    }

    /// Write to a register.
    pub fn rw_cpu(&mut self, register: Register, value: u32) {
        // Register %x0 should always be 0.
        // We always write 0 to %x0.
        let value = if register == Register::ZERO { 0 } else { value };

        let rd = value.to_le_bytes();
        self.state.vm_write::<u8, 4>(RV32_REGISTER_AS, register as u32, &rd);
    }

    pub fn register(&self, offset: usize) -> u32 {
        let bytes = unsafe { self.state.memory.read::<u8, 4>(RV32_REGISTER_AS, offset as u32) };
        u32::from_le_bytes(bytes)
    }
}
