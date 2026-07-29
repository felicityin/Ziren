use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemoryRecordEnum, MiscEvent},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::InstructionCols,
    adapter::{
        clk_low_expr, eval_cpu_state, eval_register_reader, eval_state_chain, CpuState,
        RegisterReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::MemoryCols,
    operations::AddOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `InsChip`.
pub const NUM_INS_COLS: usize = size_of::<InsCols<u8>>();

/// A chip that implements the MIPS bit-field insert instruction INS.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `ins_events`. INS is a read-modify-write of `op_a` (it preserves the untouched bits of the
/// previous value), so it needs `prev_a_value`.
///
/// INS still sends its shift-family intermediate steps (`ror_val`/`srl1_val`/`srl_val`/`sll_val`)
/// to `ShiftLeft`/`ShiftRightChip` via `send_alu` rather than embedding them locally -- unlike the
/// ADD dependency (already embedded as `add_operation` below), embedding the 4 chained
/// shift/rotate steps is a separate, later step now that this chip's own width no longer inflates
/// SEXT/EXT/MADD/TEQ's rows too (the reason this split exists in the first place).
#[derive(Default)]
pub struct InsChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct InsCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The raw fetched instruction.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`/`c`.
    pub reader: RegisterReader<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The value of the first operand.
    pub op_a_value: Word<T>,
    pub prev_a_value: Word<T>,
    /// The value of the second operand.
    pub op_b_value: Word<T>,
    /// The value of the third operand.
    pub op_c_value: Word<T>,

    /// Lsb/Msb of the insert field.
    pub lsb: T,
    pub msb: T,

    /// Result value of intermediate operations.
    ///
    /// The INS decomposition extracts the upper bits of prev_a via a right shift by `width =
    /// msb - lsb + 1`. Since the ShiftRight chip only supports shift amounts 0-31, we split this
    /// into two steps: `>> 1` then `>> (msb - lsb)`, each of which is always in range [0, 31].
    pub ror_val: Word<T>,
    pub srl1_val: Word<T>,
    pub srl_val: Word<T>,
    pub sll_val: Word<T>,

    /// `add_val = srl_val + sll_val`, computed locally (no cross-chip lookup into `AddChip`).
    pub add_operation: AddOperation<T>,

    /// Whether this row is a real, retired INS instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for InsChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Ins".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        InsCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.ins_events.len(),
            None,
            <InsChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.ins_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <InsChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_INS_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_INS_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_INS_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut InsCols<F> = row.borrow_mut();

                    if idx < input.ins_events.len() {
                        let event = &input.ins_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    } else {
                        // Padding row: force the register reader's b/c memory-access
                        // multiplicities to zero (see cpuchip-migration-register-reader-gotchas
                        // memory).
                        cols.instruction.imm_b = F::ONE;
                        cols.instruction.imm_c = F::ONE;
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_INS_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.ins_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl InsChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MiscEvent,
        cols: &mut InsCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.is_real = F::ONE;

        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);
        cols.instruction.populate(&instruction);

        *cols.reader.op_a_access.value_mut() = event.a.into();
        *cols.reader.op_b_access.value_mut() = event.b.into();
        *cols.reader.op_c_access.value_mut() = event.c.into();

        if let Some(record) = event.a_record {
            cols.reader.op_a_access.populate(record, blu);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.b_record {
            cols.reader.op_b_access.populate(record, blu);
        }
        if let Some(MemoryRecordEnum::Read(record)) = event.c_record {
            cols.reader.op_c_access.populate(record, blu);
        }
        cols.reader.populate_op_a_range_checks(blu);

        cols.op_a_value = event.a.into();
        cols.op_b_value = event.b.into();
        cols.op_c_value = event.c.into();
        cols.prev_a_value = event.prev_a.into();

        let lsb = event.c & 0x1f;
        let msb = event.c >> 5;
        let ror_val = event.prev_a.rotate_right(lsb);
        let srl1_val = ror_val >> 1;
        let srl_val = srl1_val >> (msb - lsb);
        let sll_val = event.b << (31 - msb + lsb);
        cols.lsb = F::from_canonical_u32(lsb);
        cols.msb = F::from_canonical_u32(msb);
        cols.ror_val = Word::from(ror_val);
        cols.srl1_val = Word::from(srl1_val);
        cols.srl_val = Word::from(srl_val);
        cols.sll_val = Word::from(sll_val);
        cols.add_operation.populate(blu, srl_val, sll_val);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: lsb as u8,
            c: msb as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: 1,
            a2: 0,
            b: lsb as u8,
            c: (msb + 1) as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: 1,
            a2: 0,
            b: msb as u8,
            c: 32,
        });
    }
}

impl<F> BaseAir<F> for InsChip {
    fn width(&self) -> usize {
        NUM_INS_COLS
    }
}

impl<AB> Air<AB> for InsChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &InsCols<AB::Var> = (*local).borrow();

        let is_real = local.is_real;
        builder.assert_bool(is_real);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        builder.send_program(local.pc, local.instruction, is_real.into());

        // INS is a read-modify-write of `op_a` (`prev_a_value` cross-checked against the
        // register's real previous value).
        eval_register_reader(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            &local.instruction,
            local.op_a_value.map(Into::into),
            local.prev_a_value.map(Into::into),
            is_real.into(),
            AB::Expr::zero(),
            is_real.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.into());

        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk_high,
            clk_low,
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            is_real.into(),
        );

        builder
            .when(is_real)
            .assert_word_eq(local.reader.op_b_val(), local.op_b_value.map(Into::into));
        builder
            .when(is_real)
            .assert_word_eq(local.reader.op_c_val(), local.op_c_value.map(Into::into));

        // Bind to the row's actual fetched opcode, so a row can't claim to be INS while still
        // passing the program lookup with a different instruction.
        builder
            .when(is_real)
            .assert_eq(local.instruction.opcode, Opcode::INS.as_field::<AB::F>());
        builder.when(is_real).assert_zero(local.op_c_value[2]);
        builder.when(is_real).assert_zero(local.op_c_value[3]);

        // Ins is decomposed into 6 ALU sub-operations:
        //    ror_val  = rotate_right(prev_a, lsb)            [shift: lsb ∈ 0..31]
        //    srl1_val = ror_val >> 1                          [shift: 1]
        //    srl_val  = srl1_val >> (msb - lsb)               [shift: msb-lsb ∈ 0..31]
        //    sll_val  = op_b << (31 - msb + lsb)              [shift: ∈ 0..31]
        //    add_val  = srl_val + sll_val
        //    result   = rotate_right(add_val, 31 - msb)       [shift: ∈ 0..31]
        builder.send_alu(
            Opcode::ROR.as_field::<AB::F>(),
            local.ror_val,
            local.prev_a_value,
            Word([
                AB::Expr::from_canonical_u32(0) + local.lsb,
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            is_real,
        );

        // SRL step 1: shift right by 1 (always in range).
        builder.send_alu(
            Opcode::SRL.as_field::<AB::F>(),
            local.srl1_val,
            local.ror_val,
            Word([AB::Expr::one(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            is_real,
        );

        // SRL step 2: shift right by msb - lsb (range [0, 31]).
        builder.send_alu(
            Opcode::SRL.as_field::<AB::F>(),
            local.srl_val,
            local.srl1_val,
            Word([
                AB::Expr::from_canonical_u32(0) + local.msb - local.lsb,
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            is_real,
        );

        builder.send_alu(
            Opcode::SLL.as_field::<AB::F>(),
            local.sll_val,
            local.op_b_value,
            Word([
                AB::Expr::from_canonical_u32(31) - local.msb + local.lsb,
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            is_real,
        );

        // `add_val = srl_val + sll_val`, computed locally (no cross-chip lookup into `AddChip`).
        AddOperation::<AB::F>::eval(
            builder,
            local.srl_val,
            local.sll_val,
            local.add_operation,
            is_real.into(),
        );
        let add_val = local.add_operation.value;

        builder.send_alu(
            Opcode::ROR.as_field::<AB::F>(),
            local.op_a_value,
            add_val,
            Word([
                AB::Expr::from_canonical_u32(31) - local.msb,
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            is_real,
        );

        // op_c = (msb << 5) + lsb
        builder.when(is_real).assert_eq(
            local.op_c_value.reduce::<AB>(),
            local.lsb + local.msb * AB::Expr::from_canonical_u32(32),
        );

        // 32 > msb >= lsb >= 0.
        builder.send_byte(
            ByteOpcode::U8Range.as_field::<AB::F>(),
            AB::Expr::zero(),
            local.lsb,
            local.msb,
            is_real,
        );
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            local.lsb,
            local.msb + AB::Expr::one(),
            is_real,
        );
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            local.msb,
            AB::Expr::from_canonical_u32(32),
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::MiscEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::InsChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::INS,
                op_a: 5,
                op_b: 8,
                op_c: 0x21,
                imm_b: false,
                imm_c: true,
                raw: None,
            }],
            pc_start: 0,
            pc_base: 0,
            next_pc: 4,
            image: Default::default(),
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.ins_events = vec![MiscEvent::new(
            0,
            0,
            4,
            Opcode::INS,
            0,
            0xDEAD_BEEF,
            0x21,
            0x1234_5678,
            Default::default(),
        )];
        let chip = InsChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
