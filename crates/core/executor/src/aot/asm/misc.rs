use crate::aot::common::*;
use crate::aot::{AotCompiler, AotError};
use crate::events::MemoryAccessPosition;
use crate::{Executor, ExecutorMode, Instruction, Opcode, Register};

impl AotCompiler {
    pub fn generate_mov_cond_asm(
        &self,
        instruction: &Instruction,
        _pc: u32,
    ) -> Result<String, AotError> {
        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            // self.local_counts.event_counts[instruction.opcode as usize] += 1;
            asm += &Self::inc_events_count(vec![(instruction.opcode, 1)]);

            // self.local_counts.event_counts[Opcode::ADD as usize] += 1;
            asm += &Self::inc_events_count(vec![(Opcode::ADD, 1)]);

            asm += &Self::get_access_register_meta_addr();
            asm += &Self::set_access_register_meta(instruction.op_c, MemoryAccessPosition::C);
            asm += &Self::set_access_register_meta(instruction.op_b, MemoryAccessPosition::B);
            asm +=
                &Self::set_access_register_meta(instruction.op_a as u32, MemoryAccessPosition::A);
        }

        let extern_handler_ptr = format!("{:p}", execute_mov_cond as *const ());
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);

        asm += &Self::before_call();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::after_call();

        Ok(asm)
    }

    pub fn generate_misc_asm(
        &self,
        instruction: &Instruction,
        _pc: u32,
    ) -> Result<String, AotError> {
        let mut asm = String::new();

        if self.executor_mode == ExecutorMode::Checkpoint {
            // self.local_counts.event_counts[instruction.opcode as usize] += 1;
            asm += &Self::inc_events_count(vec![(instruction.opcode, 1)]);

            if instruction.opcode == Opcode::EXT {
                // self.local_counts.event_counts[Opcode::SLL as usize] += 1;
                // self.local_counts.event_counts[Opcode::SRL as usize] += 1;
                asm += &Self::inc_events_count(vec![(Opcode::SLL, 1), (Opcode::SRL, 1)]);
            } else if instruction.is_maddsubu_instruction() {
                // self.local_counts.event_counts[Opcode::MULTU as usize] += 1;
                asm += &Self::inc_events_count(vec![(Opcode::MULTU, 1)]);
            } else if instruction.opcode == Opcode::INS {
                // self.local_counts.event_counts[Opcode::ROR as usize] += 2;
                // self.local_counts.event_counts[Opcode::SLL as usize] += 1;
                // self.local_counts.event_counts[Opcode::SRL as usize] += 1;
                // self.local_counts.event_counts[Opcode::ADD as usize] += 1;
                asm += &Self::inc_events_count(vec![
                    (Opcode::ROR, 2),
                    (Opcode::SLL, 1),
                    (Opcode::SRL, 1),
                    (Opcode::ADD, 1),
                ]);
            }

            asm += &Self::get_access_register_meta_addr();
            match instruction.opcode {
                Opcode::MADDU | Opcode::MSUBU | Opcode::MADD | Opcode::MSUB => {
                    asm +=
                        &Self::set_access_register_meta(instruction.op_c, MemoryAccessPosition::C);
                    asm +=
                        &Self::set_access_register_meta(instruction.op_b, MemoryAccessPosition::B);
                    asm += &Self::set_access_register_meta(
                        Register::LO as u32,
                        MemoryAccessPosition::A,
                    );
                    asm += &Self::set_access_register_meta(
                        Register::HI as u32,
                        MemoryAccessPosition::HI,
                    );
                }
                Opcode::WSBH | Opcode::EXT | Opcode::SEXT | Opcode::INS | Opcode::TEQ => {
                    asm +=
                        &Self::set_access_register_meta(instruction.op_b, MemoryAccessPosition::B);
                    asm += &Self::set_access_register_meta(
                        instruction.op_a as u32,
                        MemoryAccessPosition::A,
                    );
                }
                _ => unreachable!(),
            };
        }

        let extern_handler_ptr = match instruction.opcode {
            Opcode::MADDU => format!("{:p}", execute_maddu as *const ()),
            Opcode::MSUBU => format!("{:p}", execute_msubu as *const ()),
            Opcode::MADD => format!("{:p}", execute_madd as *const ()),
            Opcode::MSUB => format!("{:p}", execute_msub as *const ()),
            Opcode::WSBH => format!("{:p}", execute_wsbh as *const ()),
            Opcode::EXT => format!("{:p}", execute_ext as *const ()),
            Opcode::SEXT => format!("{:p}", execute_sext as *const ()),
            Opcode::INS => format!("{:p}", execute_ins as *const ()),
            Opcode::TEQ => format!("{:p}", execute_teq as *const ()),
            _ => unreachable!(),
        };
        let instruction_ptr = format!("{:p}", instruction as *const Instruction);

        asm += &Self::before_call();
        asm += &format!("   mov {REG_FIRST_ARG}, {REG_EXECUTOR_PTR}\n");
        asm += &format!("   mov {REG_SECOND_ARG}, {instruction_ptr}\n");
        asm += &format!("   mov {REG_CALLER}, {extern_handler_ptr}\n");
        asm += &format!("   call {REG_CALLER}\n");
        asm += &Self::after_call();

        Ok(asm)
    }
}

extern "C" fn execute_mov_cond(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rs, rt) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let a = executor.state.read_register(rd);
    let c = executor.state.read_register(rt);
    let b = executor.state.read_register(rs);
    let mov = match instruction.opcode {
        Opcode::MEQ => c == 0,
        Opcode::MNE => c != 0,
        _ => {
            unreachable!()
        }
    };

    let a = if mov { b } else { a };
    executor.state.write_register(rd, a);
}

extern "C" fn execute_maddu(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = b as u64 * c as u64;
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = multiply.wrapping_add(addend);
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_msubu(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = b as u64 * c as u64;
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = addend.wrapping_sub(multiply);
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_madd(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = (b as i32 as i64) * (c as i32 as i64);
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = multiply.wrapping_add(addend as i64) as u64;
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_msub(executor: &mut Executor, instruction: &Instruction) {
    let (lo, rt, rs) =
        (instruction.op_a.into(), (instruction.op_b as u8).into(), (instruction.op_c as u8).into());
    let c = executor.state.read_register(rs);
    let b = executor.state.read_register(rt);
    let lo_val = executor.state.read_register(Register::LO as u32);
    let hi_val = executor.state.read_register(Register::HI as u32);

    let multiply = (b as i32 as i64) * (c as i32 as i64);
    let addend = ((hi_val as u64) << 32) + lo_val as u64;
    let out = (addend as i64).wrapping_sub(multiply) as u64;
    let out_lo = out as u32;
    let out_hi = (out >> 32) as u32;

    executor.state.write_register(lo, out_lo);
    executor.state.write_register(Register::HI as u32, out_hi);
}

extern "C" fn execute_wsbh(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());
    let b = executor.state.read_register(rt);
    let a = (((b >> 16) & 0xFF) << 24)
        | (((b >> 24) & 0xFF) << 16)
        | ((b & 0xFF) << 8)
        | ((b >> 8) & 0xFF);
    executor.state.write_register(rd, a);
}

extern "C" fn execute_ext(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let msbd = c >> 5;
    let lsb = c & 0x1f;
    let mask_msb = if msbd + lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msbd + lsb + 1)) - 1 };
    let a = (b & mask_msb) >> lsb;
    executor.state.write_register(rd, a);
}

extern "C" fn execute_sext(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let a = if c > 0 { (b & 0xffff) as i16 as i32 as u32 } else { (b & 0xff) as i8 as i32 as u32 };
    executor.state.write_register(rd, a);
}

extern "C" fn execute_ins(executor: &mut Executor, instruction: &Instruction) {
    let (rd, rt, c) = (instruction.op_a.into(), (instruction.op_b as u8).into(), instruction.op_c);
    let b = executor.state.read_register(rt);
    let a = executor.state.read_register(rd);

    let msb = c >> 5;
    let lsb = c & 0x1f;
    let mask = if msb - lsb + 1 == 32 { 0xFFFFFFFF } else { (1u32 << (msb - lsb + 1)) - 1 };
    let mask_field = mask << lsb;
    let a = (a & !mask_field) | ((b << lsb) & mask_field);

    executor.state.write_register(rd, a);
}

extern "C" fn execute_teq(executor: &mut Executor, instruction: &Instruction) {
    let (rs, rt) = (instruction.op_a.into(), (instruction.op_b as u8).into());

    let src2 = executor.state.read_register(rt);
    let src1 = executor.state.read_register(rs);

    if src1 == src2 {
        panic!("ExecutionError::ExceptionOrTrap()");
    }
}
