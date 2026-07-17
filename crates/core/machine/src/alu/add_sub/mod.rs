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
    events::{AluEvent, ByteLookupEvent, ByteRecord, MemoryRecordEnum},
    ExecutionRecord, Opcode, Program, UNUSED_PC,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{
    air::{MachineAir, PublicValues, ZKM_PROOF_NUM_PV_ELTS},
    word::Word,
};

use crate::{
    adapter::InstructionCols,
    adapter::{
        clk_expr, eval_cpu_state, eval_register_reader, eval_state_chain, CpuState, RegisterReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::MemoryCols,
    operations::AddOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `AddSubChip`.
pub const NUM_ADD_SUB_COLS: usize = size_of::<AddSubCols<u8>>();

/// A chip that implements addition for the opcode ADD, ADDU, ADDI, ADDIU, SUB and SUBU.
///
/// SUB is basically an ADD with a re-arrangement of the operands and result.
/// E.g. given the standard ALU op variable name and positioning of `a` = `b` OP `c`,
/// `a` = `b` + `c` should be verified for ADD, and `b` = `a` + `c` (e.g. `a` = `b` - `c`)
/// should be verified for SUB.
///
/// Not every row corresponds to a real retired instruction: some ADD/SUB rows are internal
/// dependency checks emitted by other chips (e.g. DivRem verifying `quotient*divisor+remainder`,
/// or Branch/Jump verifying a target-address addition) that reuse this chip's arithmetic circuit
/// instead of duplicating it. Those rows carry the sentinel `pc == UNUSED_PC` and hand off via
/// the chip-to-chip `send_alu`/`receive_instruction` pair on `LookupKind::Instruction`, bypassing
/// program lookup and register access entirely. `is_real_add`/`is_real_sub` distinguish the two
/// (kept separate from `is_add`/`is_sub` rather than as a single product, since interaction
/// values/multiplicities in this lookup argument must stay affine in the trace columns).
#[derive(Default)]
pub struct AddSubChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct AddSubCols<T: Copy> {
    /// The current shard and clk. Only meaningful when `is_real_add + is_real_sub == 1`.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The raw fetched instruction, used for the program lookup and register resolution. Only
    /// meaningful when `is_real_add + is_real_sub == 1`.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`/`c`. Only meaningful when this row is a real
    /// instruction (`is_real_add + is_real_sub == 1`).
    pub reader: RegisterReader<T>,

    /// Whether this row is a real, retired ADD instruction (as opposed to an internal
    /// dependency check from another chip, or padding).
    pub is_real_add: T,

    /// Whether this row is a real, retired SUB instruction.
    pub is_real_sub: T,

    /// Instance of `AddOperation` to handle addition logic in `AddSubChip`'s ALU operations.
    /// It's result will be `a` for the add operation and `b` for the sub operation.
    pub add_operation: AddOperation<T>,

    /// The first input operand.  This will be `b` for add operations and `a` for sub operations.
    pub operand_1: Word<T>,

    /// The second input operand.  This will be `c` for both operations.
    pub operand_2: Word<T>,

    /// Flag indicating whether the opcode is `ADD`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_add: T,

    /// Flag indicating whether the opcode is `SUB`.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_sub: T,
}

impl<F: PrimeField32> MachineAir<F> for AddSubChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "AddSub".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.add_sub_events.len(),
            None,
            <AddSubChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        AddSubCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the rows for the trace.
        let chunk_size = std::cmp::max(input.add_sub_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <AddSubChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADD_SUB_COLS);

        values.chunks_mut(chunk_size * NUM_ADD_SUB_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADD_SUB_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddSubCols<F> = row.borrow_mut();

                    if idx < input.add_sub_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.add_sub_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events, &input.program);
                    } else {
                        // Padding row: force the register reader's b/c memory-access
                        // multiplicities to zero (mirrors a real, non-immediate-operand row's
                        // `is_real_add`/`is_real_sub` also being zero, but without needing an
                        // extra degree of freedom on the interaction multiplicity itself).
                        cols.instruction.imm_b = F::ONE;
                        cols.instruction.imm_c = F::ONE;
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_ADD_SUB_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.add_sub_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .add_sub_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_ADD_SUB_COLS];
                    let cols: &mut AddSubCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.add_sub_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddSubChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut AddSubCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        cols.is_add = F::from_bool(event.opcode == Opcode::ADD);
        cols.is_sub = F::from_bool(event.opcode == Opcode::SUB);

        let is_add = event.opcode == Opcode::ADD;
        let operand_1 = if is_add { event.b } else { event.a };
        let operand_2 = event.c;

        cols.add_operation.populate(blu, operand_1, operand_2);
        cols.operand_1 = Word::from(operand_1);
        cols.operand_2 = Word::from(operand_2);

        // Default: not a real instruction (matches the padding-row convention below), so the
        // register reader's b/c memory accesses have zero multiplicity unless overwritten by a
        // real fetched instruction's actual immediate flags just below.
        cols.instruction.imm_b = F::ONE;
        cols.instruction.imm_c = F::ONE;

        let is_real_instruction = event.pc != UNUSED_PC;
        if is_real_instruction {
            cols.is_real_add = F::from_bool(is_add);
            cols.is_real_sub = F::from_bool(!is_add);

            cols.state.populate(blu, event.shard, event.clk);

            let instruction = program.fetch(event.pc);
            cols.instruction.populate(&instruction);

            // Immediate operands have no memory record to `.populate()` from; seed the raw
            // event value first so those columns hold the right value either way (`.populate()`
            // below then overwrites it with the full read/write access metadata for real
            // register reads).
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
        }
    }
}

impl<F> BaseAir<F> for AddSubChip {
    fn width(&self) -> usize {
        NUM_ADD_SUB_COLS
    }
}

impl<AB> Air<AB> for AddSubChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddSubCols<AB::Var> = (*local).borrow();

        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        let is_real = local.is_add + local.is_sub;
        builder.assert_bool(local.is_add);
        builder.assert_bool(local.is_sub);
        builder.assert_bool(is_real.clone());
        builder.assert_bool(local.is_real_add);
        builder.assert_bool(local.is_real_sub);
        // `is_real_add`/`is_real_sub` can only be set alongside the matching `is_add`/`is_sub`
        // selector -- kept as their own witnessed columns (not the product `is_real * is_add`)
        // because interaction values/multiplicities in this lookup argument must stay affine in
        // the trace columns, and this mux is used below both in assertions (any degree is fine)
        // and as an interaction multiplicity (degree 1 required).
        builder.when_not(local.is_add).assert_zero(local.is_real_add);
        builder.when_not(local.is_sub).assert_zero(local.is_real_sub);
        let is_real_instruction = local.is_real_add + local.is_real_sub;

        // Evaluate the addition operation.
        AddOperation::<AB::F>::eval(
            builder,
            local.operand_1,
            local.operand_2,
            local.add_operation,
            is_real.clone(),
        );

        // Role mux, restricted to real-instruction rows (zero for padding/synthetic rows, where
        // the register reader's own columns are also left at zero): for ADD, register `a` is
        // written `add_operation.value` and register `b` holds `operand_1`; for SUB, register
        // `a` is written `operand_1` and register `b` holds `add_operation.value` (since
        // `a = b - c` is verified as `b = a + c`). Register `c` always holds `operand_2`.
        let op_a_value: Word<AB::Expr> = Word(core::array::from_fn(|i| {
            local.is_real_add.into() * local.add_operation.value[i].into()
                + local.is_real_sub.into() * local.operand_1[i].into()
        }));
        let op_b_role: Word<AB::Expr> = Word(core::array::from_fn(|i| {
            local.is_real_add.into() * local.operand_1[i].into()
                + local.is_real_sub.into() * local.add_operation.value[i].into()
        }));
        let op_c_role: Word<AB::Expr> = Word(core::array::from_fn(|i| local.operand_2[i].into()));

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk = clk_expr::<AB>(&local.state);

        builder.send_program(local.pc, local.instruction, is_real_instruction.clone());

        eval_register_reader(
            builder,
            &local.reader,
            local.state.shard,
            clk.clone(),
            &local.instruction,
            op_a_value,
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real_instruction.clone(),
        );

        eval_cpu_state(
            builder,
            &local.state,
            public_values.execution_shard,
            clk.clone(),
            is_real_instruction.clone(),
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
            is_real_instruction.clone(),
        );

        builder
            .when(is_real_instruction.clone())
            .assert_word_eq(local.reader.op_b_val(), op_b_role);
        builder.when(is_real_instruction).assert_word_eq(local.reader.op_c_val(), op_c_role);

        // Bind `is_real_add`/`is_real_sub` to the row's actual fetched opcode, so a
        // real-instruction row can't claim the wrong arithmetic variant while still passing the
        // program lookup.
        builder
            .when(local.is_real_add)
            .assert_eq(local.instruction.opcode, Opcode::ADD.as_field::<AB::F>());
        builder
            .when(local.is_real_sub)
            .assert_eq(local.instruction.opcode, Opcode::SUB.as_field::<AB::F>());

        // ---- Synthetic dependency path: matches whichever chip generated this internal check via
        // `send_alu`/`send_alu_with_hi` (always at the `UNUSED_PC` sentinel, shard/clk zero).
        // `is_add - is_real_add` (resp. `is_sub - is_real_sub`) is 1 exactly when this is a real
        // ADD/SUB row that is *not* a real instruction, i.e. a synthetic dependency row -- and
        // stays affine (degree 1), unlike the product `is_add * (1 - is_real_instruction)`. ----
        let zero_word =
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]);

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.next_pc,
            local.next_pc + AB::Expr::from_canonical_u32(4),
            AB::Expr::zero(),
            Opcode::ADD.as_field::<AB::F>(),
            local.add_operation.value,
            local.operand_1,
            local.operand_2,
            zero_word.clone(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_add - local.is_real_add,
        );

        // For sub, `operand_1` is `a`, `add_operation.value` is `b`, and `operand_2` is `c`.
        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.next_pc,
            local.next_pc + AB::Expr::from_canonical_u32(4),
            AB::Expr::zero(),
            Opcode::SUB.as_field::<AB::F>(),
            local.operand_1,
            local.add_operation.value,
            local.operand_2,
            zero_word,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_sub - local.is_real_sub,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode, UNUSED_PC};
    use zkm_hypercube::air::MachineAir;

    use super::AddSubChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        // `UNUSED_PC` keeps this a synthetic-dependency-style row, so trace generation doesn't
        // need a real `Program` to fetch an instruction from.
        shard.add_sub_events = vec![AluEvent::new(UNUSED_PC, Opcode::ADD, 14, 8, 6)];
        let chip = AddSubChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
