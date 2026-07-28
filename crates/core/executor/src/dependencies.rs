use crate::{
    events::{AluEvent, BranchEvent, MemInstrEvent, MemoryRecord, MiscEvent},
    utils::{get_quotient_and_remainder, is_signed_operation},
    Executor, Opcode, DEFAULT_PC_INC, UNUSED_PC,
};

/// Emits the dependencies for division and remainder operations.
#[allow(clippy::too_many_lines)]
pub fn emit_divrem_dependencies(executor: &mut Executor, event: AluEvent) {
    let (_, remainder) = get_quotient_and_remainder(event.b, event.c, event.opcode);
    let is_signed_operation = is_signed_operation(event.opcode);

    // `0 == c + abs_c` / `0 == remainder + abs_remainder` and `c * quotient` are now verified
    // locally by `DivRemChip` via embedded `AddOperation`/`MulOperation` (see `alu/divrem/mod.rs`),
    // so no dependency sends into `add_events`/`mul_events` are needed here at all.

    let lt_event = if is_signed_operation {
        AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SLTU,
            hi: 0,
            a: 1,
            b: (remainder as i32).unsigned_abs(),
            c: u32::max(1, (event.c as i32).unsigned_abs()),
            a_record: None,
            b_record: None,
            c_record: None,
        }
    } else {
        AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SLTU,
            hi: 0,
            a: 1,
            b: remainder,
            c: u32::max(1, event.c),
            a_record: None,
            b_record: None,
            c_record: None,
        }
    };

    if event.c != 0 {
        executor.record.lt_events.push(lt_event);
    }
}

/// Emits the dependencies for clo and clz operations.
#[allow(clippy::too_many_lines)]
pub fn emit_cloclz_dependencies(executor: &mut Executor, event: AluEvent) {
    let b = if event.opcode == Opcode::CLZ { event.b } else { !event.b };
    if b != 0 {
        let srl_event = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SRL,
            hi: 0,
            a: b >> (31 - event.a),
            b,
            c: 31 - event.a,
            a_record: None,
            b_record: None,
            c_record: None,
        };

        executor.record.shift_right_events.push(srl_event);
    }
}

/// Emit the dependencies for memory instructions.
pub fn emit_memory_dependencies(
    executor: &mut Executor,
    event: MemInstrEvent,
    memory_record: MemoryRecord,
) {
    // Every memory opcode now verifies `addr == b + c` locally via an embedded `AddOperation`
    // (see `WordAddressOperation`/`UnalignedWordAddressOperation`), so no dependency send is
    // needed here at all.
    let memory_addr = event.b.wrapping_add(event.c);
    let addr_offset = (memory_addr % 4_u32) as u8;
    let mem_value = memory_record.value;

    if matches!(event.opcode, Opcode::LB | Opcode::LH) {
        let (unsigned_mem_val, most_sig_mem_value_byte, sign_value) = match event.opcode {
            Opcode::LB => {
                let most_sig_mem_value_byte = mem_value.to_le_bytes()[addr_offset as usize];
                let sign_value = 256;
                (most_sig_mem_value_byte as u32, most_sig_mem_value_byte, sign_value)
            }
            Opcode::LH => {
                let sign_value = 65536;
                let unsigned_mem_val = match (addr_offset >> 1) % 2 {
                    0 => mem_value & 0x0000FFFF,
                    1 => (mem_value & 0xFFFF0000) >> 16,
                    _ => unreachable!(),
                };
                let most_sig_mem_value_byte = unsigned_mem_val.to_le_bytes()[1];
                (unsigned_mem_val, most_sig_mem_value_byte, sign_value)
            }
            _ => unreachable!(),
        };

        if most_sig_mem_value_byte >> 7 & 0x01 == 1 {
            let sub_event = AluEvent {
                clk: 0,
                pc: UNUSED_PC,
                next_pc: UNUSED_PC + DEFAULT_PC_INC,
                opcode: Opcode::SUB,
                hi: 0,
                a: event.a,
                b: unsigned_mem_val,
                c: sign_value,
                a_record: None,
                b_record: None,
                c_record: None,
            };
            executor.record.sub_events.push(sub_event);
        }
    }
}

/// Emit the dependencies for branch instructions.
pub fn emit_branch_dependencies(executor: &mut Executor, event: BranchEvent) {
    let a_lt_b = (event.a as i32) < (event.b as i32);
    let a_gt_b = (event.a as i32) > (event.b as i32);

    let lt_comp_event = AluEvent {
        clk: 0,
        pc: UNUSED_PC,
        next_pc: UNUSED_PC + DEFAULT_PC_INC,
        opcode: Opcode::SLT,
        hi: 0,
        a: a_lt_b as u32,
        b: event.a,
        c: event.b,
        a_record: None,
        b_record: None,
        c_record: None,
    };
    let gt_comp_event = AluEvent {
        clk: 0,
        pc: UNUSED_PC,
        next_pc: UNUSED_PC + DEFAULT_PC_INC,
        opcode: Opcode::SLT,
        hi: 0,
        a: a_gt_b as u32,
        b: event.b,
        c: event.a,
        a_record: None,
        b_record: None,
        c_record: None,
    };
    executor.record.lt_events.push(lt_comp_event);
    executor.record.lt_events.push(gt_comp_event);
    // The taken-branch `next_next_pc = next_pc + c` is now verified locally by `BranchChip` via an
    // embedded `AddOperation`, so no dependency send into `add_events` is needed here at all.
}

/// Emit the dependencies for misc instructions.
pub fn emit_misc_dependencies(executor: &mut Executor, event: MiscEvent) {
    // MADD/MADDU/MSUB/MSUBU's `b * c` is now verified locally by `MiscInstrsChip` via an
    // embedded `MulOperation` (see `misc/others/air.rs`'s `eval_maddsub`), so no dependency send
    // into `mul_events` is needed here at all.
    if matches!(event.opcode, Opcode::EXT) {
        let lsb = event.c & 0x1f;
        let msbd = event.c >> 5;
        // `execute_ext` rejects encodings with `lsb + msbd >= 32`, so the `31 - lsb - msbd`
        // shift amounts below cannot underflow.
        debug_assert!(
            lsb + msbd < 32,
            "EXT with lsb + msbd >= 32 must be rejected during execution"
        );
        let sll_val = event.b << (31 - lsb - msbd);
        let sll_event = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SLL,
            hi: 0,
            a: sll_val,
            b: event.b,
            c: 31 - lsb - msbd,
            a_record: None,
            b_record: None,
            c_record: None,
        };
        executor.record.shift_left_events.push(sll_event);
        let srl_event = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SRL,
            hi: 0,
            a: event.a,
            b: sll_val,
            c: 31 - msbd,
            a_record: None,
            b_record: None,
            c_record: None,
        };
        assert_eq!(event.a, sll_val >> (31 - msbd));
        executor.record.shift_right_events.push(srl_event);
    } else if matches!(event.opcode, Opcode::INS) {
        let lsb = event.c & 0x1f;
        let msb = event.c >> 5;
        let ror_val = event.prev_a.rotate_right(lsb);
        let ror_event = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::ROR,
            hi: 0,
            a: ror_val,
            b: event.prev_a,
            c: lsb,
            a_record: None,
            b_record: None,
            c_record: None,
        };
        executor.record.shift_right_events.push(ror_event);

        let srl1_val = ror_val >> 1;
        let srl1_event = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SRL,
            hi: 0,
            a: srl1_val,
            b: ror_val,
            c: 1,
            a_record: None,
            b_record: None,
            c_record: None,
        };
        executor.record.shift_right_events.push(srl1_event);

        let srl_val = srl1_val >> (msb - lsb);
        let srl_event = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SRL,
            hi: 0,
            a: srl_val,
            b: srl1_val,
            c: msb - lsb,
            a_record: None,
            b_record: None,
            c_record: None,
        };
        executor.record.shift_right_events.push(srl_event);

        let sll_val = event.b << (31 - msb + lsb);
        let sll_event = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::SLL,
            hi: 0,
            a: sll_val,
            b: event.b,
            c: 31 - msb + lsb,
            a_record: None,
            b_record: None,
            c_record: None,
        };
        executor.record.shift_left_events.push(sll_event);

        // `extra_shift = srl_val + sll_val` is now verified locally by `MiscInstrsChip` via an
        // embedded `AddOperation` (see `misc/others/air.rs`'s `eval_ins`), so no dependency send
        // into `add_events` is needed here at all.
        let extra_shift = srl_val + sll_val;

        let ror_event2 = AluEvent {
            clk: 0,
            pc: UNUSED_PC,
            next_pc: UNUSED_PC + DEFAULT_PC_INC,
            opcode: Opcode::ROR,
            hi: 0,
            a: event.a,
            b: extra_shift,
            c: 31 - msb,
            a_record: None,
            b_record: None,
            c_record: None,
        };
        assert_eq!(event.a, extra_shift.rotate_right(31 - msb));
        executor.record.shift_right_events.push(ror_event2);
    }
}
