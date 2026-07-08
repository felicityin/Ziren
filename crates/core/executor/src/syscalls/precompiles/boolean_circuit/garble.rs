use crate::events::{BooleanCircuitGarbleEvent, PrecompileEvent};
use crate::syscalls::{Syscall, SyscallCode, SyscallContext};
use crate::ExecutionError;

pub(crate) struct BooleanCircuitGarbleSyscall;

// number of bytes for each gate input info.
pub const GATE_INFO_BYTES: usize = 17;

impl Syscall for BooleanCircuitGarbleSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext,
        syscall_code: SyscallCode,
        arg1: u32,
        arg2: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let start_clk = ctx.clk;
        let input_ptr = arg1;
        let output_ptr = arg2;

        let mut result = true;

        // read number of gates
        let (num_gates_read_record, num_gates_u32) = ctx.mr(input_ptr);

        let (delta_read_records, delta_u32s) = ctx.mr_slice(input_ptr + 4, 4);
        let delta: [u32; 4] = delta_u32s.try_into().unwrap();

        let gate_input_size = GATE_INFO_BYTES as u32 * num_gates_u32;
        let gates_base_ptr = input_ptr + 20;
        let (gate_read_records, gates_info) =
            ctx.mr_slice(gates_base_ptr, gate_input_size as usize);

        // for each gate info
        for i in 0..num_gates_u32 {
            let base = i as usize * GATE_INFO_BYTES;

            let gate_type_u32 = gates_info[base];
            let h0 = &gates_info[base + 1..base + 5];
            let h1 = &gates_info[base + 5..base + 9];
            let label_b = &gates_info[base + 9..base + 13];
            let expected_ciphertext = &gates_info[base + 13..base + 17];

            let computed_ciphertext = h0
                .iter()
                .zip(h1.iter().zip(label_b.iter().zip(delta.iter())))
                .map(|(&h0_i, (&h1_i, (&label_b_i, &delta_i)))| {
                    if gate_type_u32 == 0 {
                        // AND gate
                        h0_i ^ h1_i ^ label_b_i
                    } else {
                        // OR gate
                        h0_i ^ h1_i ^ label_b_i ^ delta_i
                    }
                })
                .collect::<Vec<u32>>();

            let checked = computed_ciphertext.as_slice() == expected_ciphertext;
            result = result && checked;
        }

        // write result to output
        let write_record = ctx.mw(output_ptr, result as u32);
        let shard = ctx.current_shard();
        let event = BooleanCircuitGarbleEvent {
            shard,
            clk: start_clk,
            input_addr: input_ptr,
            output_addr: output_ptr,
            num_gates: num_gates_u32,
            delta,
            gates_info: gates_info.clone(),
            output: result as u32,
            num_gates_read_record,
            delta_read_records: delta_read_records.try_into().unwrap(),
            gates_read_records: gate_read_records,
            output_write_record: write_record,
            local_mem_access: ctx.postprocess(),
        };
        let syscall_event = ctx.rt.syscall_event(
            start_clk,
            None,
            ctx.next_pc,
            syscall_code.syscall_id(),
            arg1,
            arg2,
        );
        ctx.add_precompile_event(
            syscall_code,
            syscall_event,
            PrecompileEvent::BooleanCircuitGarble(event),
        );
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{events::PrecompileEvent, Executor, Program};
    use zkm_stark::ZKMCoreOpts;

    const INPUT_PTR: u32 = 0x1000;
    const OUTPUT_PTR: u32 = 0x2000;
    const OR_GATE_ID: u32 = 7;

    fn gate_info_words(gate_type: u32, delta: [u32; 4], valid: bool) -> [u32; GATE_INFO_BYTES] {
        let h0 = [11, 12, 13, 14];
        let h1 = [21, 22, 23, 24];
        let label_b = [31, 32, 33, 34];
        let mut expected = [0u32; 4];
        for i in 0..4 {
            expected[i] = h0[i] ^ h1[i] ^ label_b[i];
            if gate_type == OR_GATE_ID {
                expected[i] ^= delta[i];
            }
        }
        if !valid {
            expected[3] ^= 1;
        }

        let mut words = [0u32; GATE_INFO_BYTES];
        words[0] = gate_type;
        words[1..5].copy_from_slice(&h0);
        words[5..9].copy_from_slice(&h1);
        words[9..13].copy_from_slice(&label_b);
        words[13..17].copy_from_slice(&expected);
        words
    }

    fn write_input(
        runtime: &mut Executor<'_>,
        gate_infos: &[[u32; GATE_INFO_BYTES]],
        delta: [u32; 4],
    ) {
        let mut timestamp = 1;
        let shard = 1;
        runtime.mw(INPUT_PTR, gate_infos.len() as u32, shard, timestamp, None);
        timestamp += 1;
        for (i, value) in delta.into_iter().enumerate() {
            runtime.mw(INPUT_PTR + 4 + i as u32 * 4, value, shard, timestamp, None);
            timestamp += 1;
        }
        for (gate_idx, gate_info) in gate_infos.iter().enumerate() {
            let gate_base = INPUT_PTR + 20 + gate_idx as u32 * (GATE_INFO_BYTES as u32) * 4;
            for (word_idx, value) in gate_info.iter().enumerate() {
                runtime.mw(gate_base + word_idx as u32 * 4, *value, shard, timestamp, None);
                timestamp += 1;
            }
        }
        runtime.mw(OUTPUT_PTR, u32::MAX, shard, timestamp, None);
    }

    fn run_syscall(gate_infos: Vec<[u32; GATE_INFO_BYTES]>, delta: [u32; 4]) -> Executor<'static> {
        let mut runtime = Executor::new(Program::default(), ZKMCoreOpts::default());
        write_input(&mut runtime, &gate_infos, delta);
        runtime.state.current_shard = 2;
        runtime.state.clk = 1;

        let syscall = BooleanCircuitGarbleSyscall;
        let mut ctx = SyscallContext::new(&mut runtime);
        syscall
            .execute(&mut ctx, SyscallCode::BOOLEAN_CIRCUIT_GARBLE, INPUT_PTR, OUTPUT_PTR)
            .unwrap();
        runtime
    }

    #[test]
    fn basic_and_gate_verification_succeeds() {
        let delta = [101, 102, 103, 104];
        let mut runtime = run_syscall(vec![gate_info_words(0, delta, true)], delta);
        assert_eq!(runtime.word(OUTPUT_PTR), 1);

        let events = runtime.record.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE);
        assert_eq!(events.len(), 1);
        let (_, event) = &events[0];
        let event = match event {
            PrecompileEvent::BooleanCircuitGarble(event) => event,
            _ => unreachable!(),
        };
        assert_eq!(event.output, 1);
        assert_eq!(event.num_gates, 1);
        assert_eq!(event.gates_info.len(), GATE_INFO_BYTES);
    }

    #[test]
    fn basic_or_gate_verification_succeeds() {
        let delta = [201, 202, 203, 204];
        let mut runtime = run_syscall(vec![gate_info_words(OR_GATE_ID, delta, true)], delta);
        assert_eq!(runtime.word(OUTPUT_PTR), 1);
    }

    #[test]
    fn mixed_gates_with_bad_ciphertext_return_false() {
        let delta = [111, 222, 333, 444];
        let gate_infos = vec![
            gate_info_words(0, delta, true),
            gate_info_words(OR_GATE_ID, delta, true),
            gate_info_words(0, delta, false),
        ];
        let mut runtime = run_syscall(gate_infos, delta);
        assert_eq!(runtime.word(OUTPUT_PTR), 0);

        let events = runtime.record.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE);
        let (_, event) = &events[0];
        let event = match event {
            PrecompileEvent::BooleanCircuitGarble(event) => event,
            _ => unreachable!(),
        };
        let accessed_addrs = event
            .local_mem_access
            .iter()
            .map(|access| access.addr)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(accessed_addrs.contains(&INPUT_PTR));
        assert!(accessed_addrs.contains(&(INPUT_PTR + 20)));
        assert!(accessed_addrs.contains(&(INPUT_PTR + 20 + (GATE_INFO_BYTES as u32) * 4)));
        assert!(accessed_addrs.contains(&OUTPUT_PTR));
    }

    #[test]
    fn zero_gates_write_true() {
        let delta = [1, 2, 3, 4];
        let mut runtime = run_syscall(vec![], delta);
        assert_eq!(runtime.word(OUTPUT_PTR), 1);

        let events = runtime.record.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE);
        let (_, event) = &events[0];
        let event = match event {
            PrecompileEvent::BooleanCircuitGarble(event) => event,
            _ => unreachable!(),
        };
        assert_eq!(event.num_gates, 0);
        assert!(event.gates_info.is_empty());
        assert_eq!(event.output, 1);
    }
}
