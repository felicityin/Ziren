use std::borrow::BorrowMut;

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use zkm_core_executor::{
    events::{BranchEvent, ByteLookupEvent, ByteRecord},
    get_msb, ByteOpcode, ExecutionRecord, Opcode, Program,
};
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

use super::{BranchChip, BranchColumns, NUM_BRANCH_COLS};

impl<F: PrimeField32> MachineAir<F> for BranchChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Branch".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        BranchColumns::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.branch_events.len(),
            None,
            <BranchChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max((input.branch_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <BranchChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_BRANCH_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_BRANCH_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_BRANCH_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut BranchColumns<F> = row.borrow_mut();

                    if idx < input.branch_events.len() {
                        let event = &input.branch_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_BRANCH_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.branch_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl BranchChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &BranchEvent,
        cols: &mut BranchColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);

        // Every `branch_events` row is a real, retired instruction -- nothing ever produces a
        // synthetic dependency row here (see this chip's `Air::eval` doc comment).
        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);
        cols.reader.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
        );

        cols.is_beq = F::from_bool(matches!(event.opcode, Opcode::BEQ));
        cols.is_bne = F::from_bool(matches!(event.opcode, Opcode::BNE));
        cols.is_bltz = F::from_bool(matches!(event.opcode, Opcode::BLTZ));
        cols.is_bgtz = F::from_bool(matches!(event.opcode, Opcode::BGTZ));
        cols.is_blez = F::from_bool(matches!(event.opcode, Opcode::BLEZ));
        cols.is_bgez = F::from_bool(matches!(event.opcode, Opcode::BGEZ));

        // Computed locally (no cross-chip lookup into `LtChip`): `a_eq_b` covers BEQ/BNE's real
        // comparison and doubles as `op_a == 0` for BLTZ/BGEZ/BLEZ/BGTZ (whose `op_b` is always
        // the executor's own hardcoded-zero `event.b`); `msb_a` is `op_a`'s sign bit, the only
        // other primitive those four opcodes need.
        let a_eq_b = cols.a_eq_b.populate(event.a, event.b) == 1;
        let msb_a = get_msb(event.a);
        cols.msb_a = F::from_canonical_u8(msb_a);
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::MSB,
            a1: msb_a as u16,
            a2: 0,
            b: event.a.to_le_bytes()[3],
            c: 0,
        });

        let branching = match event.opcode {
            Opcode::BEQ => a_eq_b,
            Opcode::BNE => !a_eq_b,
            Opcode::BLTZ => msb_a == 1,
            Opcode::BLEZ => msb_a == 1 || a_eq_b,
            Opcode::BGTZ => msb_a == 0 && !a_eq_b,
            Opcode::BGEZ => msb_a == 0,
            _ => panic!("Invalid opcode: {}", event.opcode),
        };

        cols.next_pc = Word::from(event.next_pc);
        cols.next_next_pc = Word::from(event.next_next_pc);
        cols.next_pc_range_checker.populate(blu, event.next_pc);
        cols.next_next_pc_range_checker.populate(blu, event.next_next_pc);
        cols.is_branching = F::from_bool(branching);
        if branching {
            // `next_next_pc = next_pc + op_c`: only populated (and its byte-range-check
            // dependency events only recorded) when actually branching, matching the AIR's
            // `is_branching`-gated `AddOperation::eval` -- populating it unconditionally would
            // record BLU events with no matching send on non-branching rows, an interaction
            // imbalance.
            cols.add_operation.populate(blu, event.next_pc, event.c);
        } else {
            blu.add_u8_range_checks(&event.next_pc.to_le_bytes());
            blu.add_u8_range_checks(&event.next_next_pc.to_le_bytes());
        }
    }
}
