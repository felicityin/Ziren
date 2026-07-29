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

/// The number of main trace columns for `SextChip`.
pub const NUM_SEXT_COLS: usize = size_of::<SextCols<u8>>();

/// A chip that implements the MIPS sign-extend instructions SEB/SEH (both decoded as
/// `Opcode::SEXT`, distinguished by the encoded `c` immediate: 0 for SEB, 1 for SEH).
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `sext_events`. SEXT is a fresh write of `op_a` (not read-modify-write), so it needs no
/// `prev_a_value`.
#[derive(Default)]
pub struct SextChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct SextCols<T: Copy> {
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

    /// The most significant bit of the most significant byte.
    pub most_sig_bit: T,

    /// The most significant byte.
    pub sig_byte: T,

    /// SEB/SEH instruction selectors.
    pub is_seb: T,
    pub is_seh: T,

    /// Whether this row is a real, retired SEXT instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for SextChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Sext".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        SextCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.sext_events.len(),
            None,
            <SextChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.sext_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <SextChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SEXT_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_SEXT_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_SEXT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SextCols<F> = row.borrow_mut();

                    if idx < input.sext_events.len() {
                        let event = &input.sext_events[idx];
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

        Ok(RowMajorMatrix::new(values, NUM_SEXT_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.sext_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl SextChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MiscEvent,
        cols: &mut SextCols<F>,
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

        let (sig_bit, sig_byte) = if event.c > 0 {
            cols.is_seh = F::ONE;
            ((event.b as u16) >> 15, (event.b >> 8 & 0xff) as u8)
        } else {
            cols.is_seb = F::ONE;
            (((event.b as u8) >> 7) as u16, event.b as u8)
        };
        cols.most_sig_bit = F::from_canonical_u16(sig_bit);
        cols.sig_byte = F::from_canonical_u8(sig_byte);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::MSB,
            a1: sig_bit,
            a2: 0,
            b: sig_byte,
            c: 0,
        });
    }
}

impl<F> BaseAir<F> for SextChip {
    fn width(&self) -> usize {
        NUM_SEXT_COLS
    }
}

impl<AB> Air<AB> for SextChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SextCols<AB::Var> = (*local).borrow();

        let is_real = local.is_real;
        builder.assert_bool(is_real);
        builder.assert_bool(local.is_seb);
        builder.assert_bool(local.is_seh);
        builder.when(is_real).assert_one(local.is_seh + local.is_seb);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        builder.send_program(local.pc, local.instruction, is_real.into());

        // SEXT is a fresh write of `op_a` (not read-modify-write).
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

        // Bind to the row's actual fetched opcode, so a row can't claim to be SEXT while still
        // passing the program lookup with a different instruction.
        builder
            .when(is_real)
            .assert_eq(local.instruction.opcode, Opcode::SEXT.as_field::<AB::F>());

        // most_sig_bit is bit 7 of sig_byte.
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.most_sig_bit,
            local.sig_byte,
            AB::Expr::zero(),
            is_real,
        );

        // op_c can be 0 (for seb) and 1 (for seh).
        builder.when(is_real).assert_bool(local.op_c_value[0]);
        builder.when(is_real).when(local.is_seb).assert_zero(local.op_c_value[0]);
        builder.when(is_real).when(local.is_seh).assert_one(local.op_c_value[0]);

        // For seb, sig_byte is byte 0 of op_a. For seh, sig_byte is byte 1 of op_a.
        builder.when(is_real).when(local.is_seb).assert_eq(local.op_b_value[0], local.sig_byte);
        builder.when(is_real).when(local.is_seh).assert_eq(local.op_b_value[1], local.sig_byte);

        // Constraints for result value: for both seb and seh, bytes lower than sig_byte (contain)
        // equal op_b, bytes upper than sig_byte equal sign byte (0xff when sig_bit is 1,
        // otherwise 0).
        let sign_byte = AB::Expr::from_canonical_u8(0xFF) * local.most_sig_bit;

        builder.when(is_real).assert_eq(local.op_a_value[0], local.op_b_value[0]);
        builder.when(is_real).when(local.is_seb).assert_eq(local.op_a_value[1], sign_byte.clone());
        builder.when(is_real).when(local.is_seh).assert_eq(local.op_a_value[1], local.op_b_value[1]);
        builder.when(is_real).assert_eq(local.op_a_value[2], sign_byte.clone());
        builder.when(is_real).assert_eq(local.op_a_value[3], sign_byte);
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::MiscEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::SextChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SEXT,
                op_a: 5,
                op_b: 8,
                op_c: 0,
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
        shard.sext_events = vec![MiscEvent::new(
            0,
            0,
            4,
            Opcode::SEXT,
            0xFFFFFF80,
            0x80,
            0,
            0,
            Default::default(),
        )];
        let chip = SextChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
