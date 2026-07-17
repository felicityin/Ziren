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
    events::{ByteLookupEvent, ByteRecord, MemoryRecordEnum, MovCondEvent},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{
    air::{BaseAirBuilder, MachineAir, PublicValues, ZKMAirBuilder, ZKM_PROOF_NUM_PV_ELTS},
    word::Word,
};

use crate::{
    adapter::InstructionCols,
    adapter::{
        clk_expr, eval_cpu_state, eval_register_reader, eval_state_chain, CpuState, RegisterReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::MemoryCols,
    CoreChipError,
};

use crate::operations::IsZeroWordOperation;

use crate::utils::{next_power_of_two, zeroed_f_vec};

/// The number of main trace columns for `MovCondChip`.
pub const NUM_MOV_COND_COLS: usize = size_of::<MovCondCols<u8>>();

/// A chip that implements condition mov for the opcode MNE，MEQ (and the unrelated WSBH,
/// grouped here for chip-size reasons).
///
/// Nothing ever emits a synthetic dependency row into `movcond_events` and this chip never
/// produces one either -- every row here is a real, retired instruction.
#[derive(Default)]
pub struct MovCondChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct MovCondCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The raw fetched instruction.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`/`c`.
    pub reader: RegisterReader<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The value of the second operand.
    pub op_a_value: Word<T>,
    pub prev_a_value: Word<T>,
    /// The value of the second operand.
    pub op_b_value: Word<T>,
    /// The value of the third operand.
    pub op_c_value: Word<T>,

    /// Whether c equals 0.
    pub c_eq_0: IsZeroWordOperation<T>,

    /// Flag indicating whether the opcode is `MNE`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_mne: T,

    /// Flag indicating whether the opcode is `MEQ`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_meq: T,

    /// Flag indicating whether the opcode is `WSBH`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_wsbh: T,
}

impl<F: PrimeField32> MachineAir<F> for MovCondChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "MovCond".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        MovCondCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.movcond_events.len(),
            None,
            <MovCondChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.movcond_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <MovCondChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MOV_COND_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_MOV_COND_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_MOV_COND_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut MovCondCols<F> = row.borrow_mut();

                    if idx < input.movcond_events.len() {
                        let event = &input.movcond_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    } else {
                        // Padding row: force the register reader's b/c memory-access
                        // multiplicities to zero (see
                        // cpuchip-migration-register-reader-gotchas memory).
                        cols.instruction.imm_b = F::ONE;
                        cols.instruction.imm_c = F::ONE;
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_MOV_COND_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.movcond_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl MovCondChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MovCondEvent,
        cols: &mut MovCondCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        // Every `movcond_events` row is a real, retired instruction -- nothing ever produces a
        // synthetic dependency row here.
        cols.state.populate(blu, event.shard, event.clk);

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

        cols.c_eq_0.populate(event.c);

        cols.is_meq = F::from_bool(matches!(event.opcode, Opcode::MEQ));
        cols.is_mne = F::from_bool(matches!(event.opcode, Opcode::MNE));
        cols.is_wsbh = F::from_bool(matches!(event.opcode, Opcode::WSBH));
    }
}

impl<F> BaseAir<F> for MovCondChip {
    fn width(&self) -> usize {
        NUM_MOV_COND_COLS
    }
}

impl<AB> Air<AB> for MovCondChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MovCondCols<AB::Var> = (*local).borrow();
        let is_real = local.is_mne + local.is_meq + local.is_wsbh;

        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk = clk_expr::<AB>(&local.state);

        builder.send_program(local.pc, local.instruction, is_real.clone());

        // MEQ/MNE are read-modify-write of `op_a` (`is_rw_a = 1`, `prev_a_value` cross-checked
        // against the register's real previous value via `hi_or_prev_a`) -- this is what lets
        // `op_a` keep its old value when the move condition is false. WSBH is a fresh write.
        eval_register_reader(
            builder,
            &local.reader,
            local.state.shard,
            clk.clone(),
            &local.instruction,
            local.op_a_value.map(Into::into),
            local.prev_a_value.map(Into::into),
            local.is_mne + local.is_meq,
            AB::Expr::zero(),
            is_real.clone(),
        );

        eval_cpu_state(
            builder,
            &local.state,
            public_values.execution_shard,
            clk.clone(),
            is_real.clone(),
        );

        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk,
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            is_real.clone(),
        );

        builder
            .when(is_real.clone())
            .assert_word_eq(local.reader.op_b_val(), local.op_b_value.map(Into::into));
        builder
            .when(is_real.clone())
            .assert_word_eq(local.reader.op_c_val(), local.op_c_value.map(Into::into));

        // Bind each opcode flag to the row's actual fetched opcode, so a real-instruction row
        // can't claim the wrong mov-cond/WSBH variant while still passing the program lookup.
        builder
            .when(local.is_meq)
            .assert_eq(local.instruction.opcode, Opcode::MEQ.as_field::<AB::F>());
        builder
            .when(local.is_mne)
            .assert_eq(local.instruction.opcode, Opcode::MNE.as_field::<AB::F>());
        builder
            .when(local.is_wsbh)
            .assert_eq(local.instruction.opcode, Opcode::WSBH.as_field::<AB::F>());

        IsZeroWordOperation::<AB::F>::eval(
            builder,
            local.op_c_value.map(|x| x.into()),
            local.c_eq_0,
            is_real.clone(),
        );

        // Constraints for condition move result:
        // op_a = op_b, when condition is true.
        // Otherwise, op_a remains unchanged.
        {
            builder
                .when(local.is_meq)
                .when(local.c_eq_0.result)
                .assert_word_eq(local.op_a_value, local.op_b_value);

            builder
                .when(local.is_meq)
                .when_not(local.c_eq_0.result)
                .assert_word_eq(local.op_a_value, local.prev_a_value);

            builder
                .when(local.is_mne)
                .when_not(local.c_eq_0.result)
                .assert_word_eq(local.op_a_value, local.op_b_value);

            builder
                .when(local.is_mne)
                .when(local.c_eq_0.result)
                .assert_word_eq(local.op_a_value, local.prev_a_value);
        }

        self.eval_wsbh(builder, local);
        builder.when(local.is_wsbh).assert_word_zero(local.prev_a_value);
        builder.assert_bool(local.is_mne);
        builder.assert_bool(local.is_meq);
        builder.assert_bool(local.is_wsbh);
        builder.assert_bool(is_real);
    }
}

impl MovCondChip {
    pub(crate) fn eval_wsbh<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &MovCondCols<AB::Var>,
    ) {
        builder.when(local.is_wsbh).assert_eq(local.op_a_value[0], local.op_b_value[1]);

        builder.when(local.is_wsbh).assert_eq(local.op_a_value[1], local.op_b_value[0]);

        builder.when(local.is_wsbh).assert_eq(local.op_a_value[2], local.op_b_value[3]);

        builder.when(local.is_wsbh).assert_eq(local.op_a_value[3], local.op_b_value[2]);
    }
}

#[cfg(test)]
mod tests {

    use crate::utils::{run_test, setup_logger};

    use zkm_core_executor::{Instruction, Opcode, Program};

    #[test]
    fn test_mov_cond_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xf, false, true),
            Instruction::new(Opcode::ADD, 28, 0, 0x8F8F, false, true),
            Instruction::new(Opcode::MEQ, 30, 29, 0, false, false),
            Instruction::new(Opcode::MEQ, 30, 29, 28, false, false),
            Instruction::new(Opcode::MEQ, 0, 29, 0, false, false),
            Instruction::new(Opcode::MEQ, 0, 29, 29, false, false),
            Instruction::new(Opcode::MNE, 30, 29, 28, false, false),
            Instruction::new(Opcode::MNE, 0, 29, 0, false, false),
            Instruction::new(Opcode::WSBH, 32, 29, 0, false, true),
            Instruction::new(Opcode::WSBH, 32, 31, 0, false, true),
            Instruction::new(Opcode::WSBH, 0, 29, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }
}
