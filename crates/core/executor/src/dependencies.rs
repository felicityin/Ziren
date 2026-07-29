use crate::{
    events::{AluEvent, MemInstrEvent, MemoryRecord},
    Executor, Opcode, DEFAULT_PC_INC, UNUSED_PC,
};

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

