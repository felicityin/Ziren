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
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `ExtChip`.
pub const NUM_EXT_COLS: usize = size_of::<ExtCols<u8>>();

/// A chip that implements the MIPS bit-field extract instruction EXT.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `ext_events`. EXT is a fresh write of `op_a` (not read-modify-write), so it needs no
/// `prev_a_value`.
///
/// EXT still sends its two intermediate shift steps (`sll_val = op_b << (31 - lsb - msbd)` then
/// `op_a = sll_val >> (31 - msbd)`) to `ShiftLeft`/`ShiftRightChip` via `send_alu` rather than
/// embedding them locally -- unlike the ADD/MUL dependencies removed earlier this session,
/// embedding shift logic here is a separate, later step, now unblocked by this chip's split from
/// `SextChip`/`InsChip`/`MaddsubChip`/`TeqChip` (each opcode's width no longer inflates the
/// others').
#[derive(Default)]
pub struct ExtChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct ExtCols<T: Copy> {
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
    /// The value of the second operand.
    pub op_b_value: Word<T>,
    /// The value of the third operand.
    pub op_c_value: Word<T>,

    /// Lsb/Msb of the extracted field.
    pub lsb: T,
    pub msbd: T,

    /// Result value of the intermediate SLL operation.
    pub sll_val: Word<T>,

    /// Whether this row is a real, retired EXT instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for ExtChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Ext".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        ExtCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.ext_events.len(),
            None,
            <ExtChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.ext_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <ExtChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_EXT_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_EXT_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_EXT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut ExtCols<F> = row.borrow_mut();

                    if idx < input.ext_events.len() {
                        let event = &input.ext_events[idx];
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

        Ok(RowMajorMatrix::new(values, NUM_EXT_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.ext_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl ExtChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MiscEvent,
        cols: &mut ExtCols<F>,
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

        let lsb = event.c & 0x1f;
        let msbd = event.c >> 5;
        let shift_left = event.b << (31 - lsb - msbd);
        cols.lsb = F::from_canonical_u32(lsb);
        cols.msbd = F::from_canonical_u32(msbd);
        cols.sll_val = Word::from(shift_left);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: lsb as u8,
            c: msbd as u8,
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: 1,
            a2: 0,
            b: (lsb + msbd) as u8,
            c: 32,
        });
    }
}

impl<F> BaseAir<F> for ExtChip {
    fn width(&self) -> usize {
        NUM_EXT_COLS
    }
}

impl<AB> Air<AB> for ExtChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ExtCols<AB::Var> = (*local).borrow();

        let is_real = local.is_real;
        builder.assert_bool(is_real);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        builder.send_program(local.pc, local.instruction, is_real.into());

        // EXT is a fresh write of `op_a` (not read-modify-write).
        eval_register_reader(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            &local.instruction,
            local.op_a_value.map(Into::into),
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
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

        // Bind to the row's actual fetched opcode, so a row can't claim to be EXT while still
        // passing the program lookup with a different instruction.
        builder
            .when(is_real)
            .assert_eq(local.instruction.opcode, Opcode::EXT.as_field::<AB::F>());
        builder.when(is_real).assert_zero(local.op_c_value[2]);
        builder.when(is_real).assert_zero(local.op_c_value[3]);

        // Ext can be divided into 2 operations:
        //    sll_val = op_b << (31 - lsb - msbd)
        //    result = sll_val >> (31 - msbd)
        builder.send_alu(
            Opcode::SLL.as_field::<AB::F>(),
            local.sll_val,
            local.op_b_value,
            Word([
                AB::Expr::from_canonical_u32(31) - local.msbd - local.lsb,
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            is_real,
        );

        builder.send_alu(
            Opcode::SRL.as_field::<AB::F>(),
            local.op_a_value,
            local.sll_val,
            Word([
                AB::Expr::from_canonical_u32(31) - local.msbd,
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            is_real,
        );

        // op_c = (msbd << 5) + lsb
        builder.when(is_real).assert_eq(
            local.op_c_value.reduce::<AB>(),
            local.lsb + local.msbd * AB::Expr::from_canonical_u32(32),
        );

        // 0 <= lsb/msbd < 32, lsb + msbd < 32.
        builder.send_byte(
            ByteOpcode::U8Range.as_field::<AB::F>(),
            AB::Expr::zero(),
            local.lsb,
            local.msbd,
            is_real,
        );
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            local.lsb + local.msbd,
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

    use super::ExtChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::EXT,
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
        shard.ext_events = vec![MiscEvent::new(
            0,
            0,
            4,
            Opcode::EXT,
            0,
            0xDEAD_BEEF,
            0x21,
            0,
            Default::default(),
        )];
        let chip = ExtChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
