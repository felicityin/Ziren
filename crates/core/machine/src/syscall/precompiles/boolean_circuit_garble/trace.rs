use crate::syscall::precompiles::boolean_circuit_garble::columns::{
    BooleanCircuitGarbleCols, NUM_BOOLEAN_CIRCUIT_GARBLE_COLS,
};
use crate::syscall::precompiles::boolean_circuit_garble::{
    BooleanCircuitGarbleChip, GATE_INFO_BYTES, OR_GATE_ID,
};
use crate::{utils::next_power_of_two, CoreChipError};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::ParallelSlice;
use std::borrow::BorrowMut;
use zkm_core_executor::events::{
    BooleanCircuitGarbleEvent, ByteLookupEvent, ByteRecord, PrecompileEvent,
};
use zkm_core_executor::syscalls::SyscallCode;
use zkm_core_executor::{ExecutionRecord, Program};
#[cfg(feature = "picus")]
use zkm_stark::air::PicusInfo;
use zkm_stark::MachineAir;

impl<F: PrimeField32> MachineAir<F> for BooleanCircuitGarbleChip {
    type Record = ExecutionRecord;
    type Program = Program;
    type Error = CoreChipError;

    fn name(&self) -> String {
        "BooleanCircuitGarble".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        BooleanCircuitGarbleCols::<u8>::picus_info()
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let events = input.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE);
        let chunk_size = std::cmp::max(events.len() / num_cpus::get(), 1);

        let blu_batches = events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(_, event)| {
                    let event = if let PrecompileEvent::BooleanCircuitGarble(event) = event {
                        event
                    } else {
                        unreachable!();
                    };

                    let _ = self.event_to_rows::<F>(event, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = input.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE);
        let mut rows: Vec<[F; NUM_BOOLEAN_CIRCUIT_GARBLE_COLS]> = events
            .par_iter()
            .flat_map(|(_, event)| {
                let event = if let PrecompileEvent::BooleanCircuitGarble(event) = event {
                    event
                } else {
                    unreachable!();
                };

                self.event_to_rows(event, &mut Vec::new())
            })
            .collect();

        let padded = next_power_of_two(
            rows.len(),
            input.fixed_log2_rows::<F, _>(self),
            <BooleanCircuitGarbleChip as MachineAir<F>>::name(self).as_str(),
        );
        rows.resize_with(padded, || [F::ZERO; NUM_BOOLEAN_CIRCUIT_GARBLE_COLS]);
        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_BOOLEAN_CIRCUIT_GARBLE_COLS,
        ))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::BOOLEAN_CIRCUIT_GARBLE).is_empty()
        }
    }
}

impl BooleanCircuitGarbleChip {
    pub fn event_to_rows<F: PrimeField32>(
        &self,
        event: &BooleanCircuitGarbleEvent,
        blu: &mut impl ByteRecord,
    ) -> Vec<[F; NUM_BOOLEAN_CIRCUIT_GARBLE_COLS]> {
        let gates_num = event.num_gates();
        let mut rows = Vec::new();

        let mut input_address = event.input_addr;
        let mut pre_check = true;

        // first row to read gates_num and delta
        // gates_num: gate_input_mem[0]
        // delta: gate_input_mem[1..5]
        {
            let mut row = [F::ZERO; NUM_BOOLEAN_CIRCUIT_GARBLE_COLS];
            let cols: &mut BooleanCircuitGarbleCols<F> = row.as_mut_slice().borrow_mut();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.is_real = F::ONE;
            cols.is_gate = F::ZERO;
            cols.is_first_row = F::ONE;
            cols.is_empty = F::from_bool(gates_num == 0);
            cols.input_address = F::from_canonical_u32(input_address);
            cols.output_address = F::from_canonical_u32(event.output_addr);
            cols.gates_num = F::from_canonical_u32(gates_num as u32);
            cols.checks_acc = F::ONE;
            for i in 0..4 {
                let delta_i_bytes = event.delta[i].to_le_bytes();
                cols.delta[i]
                    .0
                    .iter_mut()
                    .enumerate()
                    .for_each(|(id, x)| *x = F::from_canonical_u8(delta_i_bytes[id]));
            }
            // read number of gates
            cols.gates_input_mem[0].populate(event.num_gates_read_record, blu);
            // read delta
            for i in 0..4 {
                cols.gates_input_mem[1 + i].populate(event.delta_read_records[i], blu);
            }
            if gates_num == 0 {
                cols.result_mem.populate(event.output_write_record, blu);
            }
            rows.push(row);
        }

        input_address += 20;
        for gate_id in 0..gates_num {
            let mut row = [F::ZERO; NUM_BOOLEAN_CIRCUIT_GARBLE_COLS];
            let cols: &mut BooleanCircuitGarbleCols<F> = row.as_mut_slice().borrow_mut();
            cols.shard = F::from_canonical_u32(event.shard);
            cols.clk = F::from_canonical_u32(event.clk);
            cols.is_real = F::ONE;
            cols.is_gate = F::ONE;
            cols.input_address = F::from_canonical_u32(input_address);
            cols.output_address = F::from_canonical_u32(event.output_addr);
            cols.is_empty = F::ZERO;
            cols.is_first_gate = F::from_bool(gate_id == 0);
            cols.is_last_gate = F::from_bool(gate_id == gates_num - 1);
            cols.not_last_gate = F::from_bool(gate_id != gates_num - 1);
            cols.gate_id = F::from_canonical_u32(gate_id as u32);
            cols.gates_num = F::from_canonical_u32(gates_num as u32);

            for i in 0..4 {
                let delta_i_bytes = event.delta[i].to_le_bytes();
                cols.delta[i]
                    .0
                    .iter_mut()
                    .enumerate()
                    .for_each(|(id, x)| *x = F::from_canonical_u8(delta_i_bytes[id]));
            }

            // read gate info
            for i in 0..GATE_INFO_BYTES {
                cols.gates_input_mem[i]
                    .populate(event.gates_read_records[gate_id * GATE_INFO_BYTES + i], blu);
            }

            let gate_type = event.gates_info[gate_id * GATE_INFO_BYTES];
            assert!(gate_type == 0 || gate_type == OR_GATE_ID);
            cols.gate_type[(gate_type == OR_GATE_ID) as usize] = F::ONE;

            // XOR computation
            let mut check_u32s = [0u32; 4];
            for i in 0..4 {
                let h0_id = gate_id * GATE_INFO_BYTES + 1 + i;
                let h1_id = gate_id * GATE_INFO_BYTES + 5 + i;
                let label_b_id = gate_id * GATE_INFO_BYTES + 9 + i;
                let expected_id = gate_id * GATE_INFO_BYTES + 13 + i;

                let inter1 =
                    cols.aux1[i].populate(blu, event.gates_info[h0_id], event.gates_info[h1_id]);
                let inter2 = cols.aux2[i].populate(blu, inter1, event.gates_info[label_b_id]);
                let inter3 = cols.aux3[i].populate(blu, inter2, event.delta[i]);
                if i == 0 {
                    if gate_type == 0 {
                        // AND gate
                        check_u32s[i] =
                            cols.is_equal_words[i].populate(inter2, event.gates_info[expected_id]);
                    } else {
                        // OR gate
                        check_u32s[i] =
                            cols.is_equal_words[i].populate(inter3, event.gates_info[expected_id]);
                    }
                } else if gate_type == 0 {
                    // AND gate
                    check_u32s[i] = check_u32s[i - 1]
                        * cols.is_equal_words[i].populate(inter2, event.gates_info[expected_id]);
                } else {
                    // OR gate
                    check_u32s[i] = check_u32s[i - 1]
                        * cols.is_equal_words[i].populate(inter3, event.gates_info[expected_id]);
                }
            }
            // populate check results
            cols.checks[0] = F::from_canonical_u32(check_u32s[1]);
            cols.checks[1] = F::from_canonical_u32(check_u32s[2]);
            cols.checks[2] = F::from_canonical_u32(check_u32s[3]);
            cols.checks_acc = F::from_bool(pre_check);
            pre_check = pre_check && (check_u32s[3] == 1);

            // if this is the last gate, write result
            if gate_id == gates_num - 1 {
                cols.result_mem.populate(event.output_write_record, blu);
            }

            rows.push(row);
            input_address += 68;
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syscall::precompiles::boolean_circuit_garble::columns::BooleanCircuitGarbleCols;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
    use p3_matrix::stack::VerticalPair;
    use p3_matrix::Matrix;
    use std::{
        borrow::BorrowMut,
        panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe},
    };
    use zkm_core_executor::{
        events::{
            BooleanCircuitGarbleEvent, MemoryReadRecord, MemoryWriteRecord, PrecompileEvent,
            SyscallEvent,
        },
        syscalls::SyscallCode,
        ExecutionRecord,
    };
    use zkm_stark::air::{EmptyMessageBuilder, MachineAir};

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

    fn make_event(gate_types: &[(u32, bool)], output: u32) -> BooleanCircuitGarbleEvent {
        let shard = 1;
        let clk = 5;
        let input_addr = 0x1000;
        let output_addr = 0x2000;
        let delta = [101, 102, 103, 104];
        let mut gates_info = Vec::new();
        for &(gate_type, valid) in gate_types {
            gates_info.extend_from_slice(&gate_info_words(gate_type, delta, valid));
        }

        let mut timestamp = 1u32;
        let num_gates_read_record =
            MemoryReadRecord::new(gate_types.len() as u32, shard, timestamp, 0, 0);
        timestamp += 1;

        let delta_read_records = core::array::from_fn(|i| {
            let record = MemoryReadRecord::new(delta[i], shard, timestamp, 0, 0);
            timestamp += 1;
            record
        });

        let gates_read_records = gates_info
            .iter()
            .map(|&value| {
                let record = MemoryReadRecord::new(value, shard, timestamp, 0, 0);
                timestamp += 1;
                record
            })
            .collect();

        let output_write_record = MemoryWriteRecord::new(output, shard, timestamp, 0, 0, 0);

        BooleanCircuitGarbleEvent {
            shard,
            clk,
            input_addr,
            output_addr,
            num_gates: gate_types.len() as u32,
            delta,
            gates_info,
            output,
            num_gates_read_record,
            delta_read_records,
            gates_read_records,
            output_write_record,
            local_mem_access: vec![],
        }
    }

    fn trace_for_event(event: BooleanCircuitGarbleEvent) -> RowMajorMatrix<KoalaBear> {
        let mut record = ExecutionRecord::default();
        let syscall_code = SyscallCode::BOOLEAN_CIRCUIT_GARBLE;
        let syscall_event = SyscallEvent {
            pc: 32,
            next_pc: 36,
            shard: event.shard,
            clk: event.clk,
            a_record: MemoryWriteRecord::default(),
            a_record_is_real: false,
            syscall_id: syscall_code.syscall_id(),
            arg1: event.input_addr,
            arg2: event.output_addr,
        };
        record.precompile_events.add_event(
            syscall_code,
            syscall_event,
            PrecompileEvent::BooleanCircuitGarble(event),
        );

        BooleanCircuitGarbleChip.generate_trace(&record, &mut ExecutionRecord::default()).unwrap()
    }

    struct EvalBuilder<'a> {
        local: &'a [KoalaBear],
        next: &'a [KoalaBear],
        is_first_row: bool,
        is_last_row: bool,
    }

    impl<'a> AirBuilder for EvalBuilder<'a> {
        type F = KoalaBear;
        type Expr = KoalaBear;
        type Var = KoalaBear;
        type M = VerticalPair<RowMajorMatrixView<'a, KoalaBear>, RowMajorMatrixView<'a, KoalaBear>>;

        fn main(&self) -> Self::M {
            VerticalPair::new(
                RowMajorMatrixView::new_row(self.local),
                RowMajorMatrixView::new_row(self.next),
            )
        }

        fn is_first_row(&self) -> Self::Expr {
            KoalaBear::from_bool(self.is_first_row)
        }

        fn is_last_row(&self) -> Self::Expr {
            KoalaBear::from_bool(self.is_last_row)
        }

        fn is_transition_window(&self, size: usize) -> Self::Expr {
            assert_eq!(size, 2);
            KoalaBear::from_bool(!self.is_last_row)
        }

        fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
            assert_eq!(x.into(), KoalaBear::ZERO, "constraints had nonzero value");
        }
    }

    impl<'a> AirBuilderWithPublicValues for EvalBuilder<'a> {
        type PublicVar = KoalaBear;

        fn public_values(&self) -> &[Self::PublicVar] {
            &[]
        }
    }

    impl<'a> EmptyMessageBuilder for EvalBuilder<'a> {}

    fn check_trace(trace: &RowMajorMatrix<KoalaBear>) {
        let air = BooleanCircuitGarbleChip;
        let height = trace.height();
        for row_index in 0..height {
            let row_index_next = (row_index + 1) % height;
            let local = trace.row_slice(row_index);
            let next = trace.row_slice(row_index_next);
            let mut builder = EvalBuilder {
                local: &*local,
                next: &*next,
                is_first_row: row_index == 0,
                is_last_row: row_index == height - 1,
            };
            air.eval(&mut builder);
        }
    }
    fn assert_gate_row_encoding(row: &mut [KoalaBear], expected_gate_type: u32) {
        let cols: &mut BooleanCircuitGarbleCols<KoalaBear> = row.borrow_mut();
        assert_eq!(cols.gate_type[0], KoalaBear::from_bool(expected_gate_type == 0));
        assert_eq!(cols.gate_type[1], KoalaBear::from_bool(expected_gate_type == OR_GATE_ID));

        let encoded_gate_type = cols.gate_type[1] * KoalaBear::from_canonical_u32(OR_GATE_ID);
        let gate_type_word = cols.gates_input_mem[0].access.value[0];
        assert_eq!(encoded_gate_type, gate_type_word);
    }

    #[test]
    fn test_zero_gate_trace_writes_true_result() {
        let event = make_event(&[], 1);
        let chip = BooleanCircuitGarbleChip::default();
        let mut rows = chip.event_to_rows::<KoalaBear>(&event, &mut Vec::new());
        assert_eq!(rows.len(), 1);

        let cols: &mut BooleanCircuitGarbleCols<KoalaBear> = rows[0].as_mut_slice().borrow_mut();
        assert_eq!(cols.is_first_row, KoalaBear::ONE);
        assert_eq!(cols.is_empty, KoalaBear::ONE);
        assert_eq!(cols.result_mem.access.value[0], KoalaBear::ONE);
    }

    #[test]
    fn test_and_gate_type_encoding_matches_gate_word() {
        let event = make_event(&[(0, true)], 1);
        let chip = BooleanCircuitGarbleChip::default();
        let mut rows = chip.event_to_rows::<KoalaBear>(&event, &mut Vec::new());
        assert_eq!(rows.len(), 2);
        assert_gate_row_encoding(&mut rows[1], 0);
    }

    #[test]
    fn test_mixed_gate_type_encoding_matches_gate_words() {
        let event = make_event(&[(0, true), (OR_GATE_ID, true)], 1);
        let chip = BooleanCircuitGarbleChip::default();
        let mut rows = chip.event_to_rows::<KoalaBear>(&event, &mut Vec::new());
        assert_eq!(rows.len(), 3);
        assert_gate_row_encoding(&mut rows[1], 0);
        assert_gate_row_encoding(&mut rows[2], OR_GATE_ID);

        let second_gate: &mut BooleanCircuitGarbleCols<KoalaBear> =
            rows[2].as_mut_slice().borrow_mut();
        assert_eq!(second_gate.input_address, KoalaBear::from_canonical_u32(0x1000 + 20 + 68));
        assert_eq!(second_gate.result_mem.access.value[0], KoalaBear::ONE);
    }

    #[test]
    fn test_boolean_circuit_garble_air_accepts_valid_trace() {
        let trace = trace_for_event(make_event(&[(0, true), (OR_GATE_ID, true), (0, true)], 1));
        check_trace(&trace);
    }

    #[test]
    fn test_boolean_circuit_garble_air_accepts_false_result() {
        let trace = trace_for_event(make_event(&[(0, true), (OR_GATE_ID, false)], 0));
        check_trace(&trace);
    }

    #[test]
    fn test_boolean_circuit_garble_air_rejects_inconsistent_output() {
        let trace = trace_for_event(make_event(&[(0, true), (OR_GATE_ID, false)], 1));
        let prev_hook = take_hook();
        set_hook(Box::new(|_| {}));
        let result = catch_unwind(AssertUnwindSafe(|| check_trace(&trace)));
        set_hook(prev_hook);
        assert!(result.is_err());
    }
}
