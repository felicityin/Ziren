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
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, UNUSED_PC,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::InstructionCols,
    adapter::{clk_low_expr, eval_cpu_state, eval_r_type_reader, eval_state_chain, CpuState, RTypeReader},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    operations::AddOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `AddChip`.
pub const NUM_ADD_COLS: usize = size_of::<AddCols<u8>>();

/// A chip that implements addition for the register-form opcodes ADD and ADDU.
///
/// Not every row corresponds to a real retired instruction: some rows are internal dependency
/// checks emitted by other chips (e.g. DivRem verifying `quotient*divisor+remainder`, or
/// Branch/Jump verifying a target-address addition) that reuse this chip's arithmetic circuit
/// instead of duplicating it. Those rows carry the sentinel `pc == UNUSED_PC` and hand off via
/// the chip-to-chip `send_alu`/`receive_instruction` pair on `LookupKind::Instruction`, bypassing
/// program lookup and register access entirely. `is_retired` distinguishes the two (kept as its
/// own witnessed column rather than folded into a product, since interaction
/// values/multiplicities in this lookup argument must stay affine in the trace columns).
///
/// `Opcode::ADD` also covers two other shapes, both split into their own chips: ADDI/ADDIU
/// (register `b` + an encoded immediate `c`, see `AddiChip`) and SYNC/Pref (fully-immediate,
/// `op_a=0` constant, no register read for `b`/`c` at all, see `AddNoopChip`). Additionally, a
/// real register-register ADD/ADDU whose destination happens to be register 0 (`add $zero, ...`)
/// is routed to `AluX0Chip` instead, since its result is unobservable and discarding it soundly
/// requires a different (cheaper) register-write scheme than a real result does -- see
/// `RTypeReader`'s doc comment. Every real row reaching *this* chip therefore has a genuine,
/// non-zero destination register, which is what lets it use the narrow `RTypeReader` (see its doc
/// comment) instead of the generic `InstructionCols`+`RegisterReader` pair.
#[derive(Default)]
pub struct AddChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct AddCols<T: Copy> {
    /// The current shard and clk. Only meaningful when `is_retired == 1`.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`. Only meaningful when this row is a real
    /// instruction (`is_retired == 1`).
    pub adapter: RTypeReader<T>,

    /// Whether this row is a real, retired ADD instruction (as opposed to an internal
    /// dependency check from another chip, or padding).
    pub is_retired: T,

    /// Instance of `AddOperation` to handle addition logic: `value = operand_1 + operand_2`.
    pub add_operation: AddOperation<T>,

    /// The first input operand (register `b` for a real instruction).
    pub operand_1: Word<T>,

    /// The second input operand (register `c` for a real instruction).
    pub operand_2: Word<T>,

    /// Flag indicating whether this row is a real ADD-shaped row (retired instruction or
    /// synthetic dependency check) as opposed to padding.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for AddChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Add".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.add_events.len(),
            None,
            <AddChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        AddCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the rows for the trace.
        let chunk_size = std::cmp::max(input.add_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <AddChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADD_COLS);

        values.chunks_mut(chunk_size * NUM_ADD_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADD_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddCols<F> = row.borrow_mut();

                    if idx < input.add_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.add_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events, &input.program);
                    }
                    // A padding row is left all-zero: `is_real`/`is_retired` default to 0, which
                    // gates every interaction below to zero multiplicity on its own -- unlike the
                    // generic `RegisterReader`, `RTypeReader` needs no separate "force immediate
                    // flags" workaround for this.
                });
            },
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_ADD_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.add_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .add_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_ADD_COLS];
                    let cols: &mut AddCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.add_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut AddCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        cols.is_real = F::ONE;

        let operand_1 = event.b;
        let operand_2 = event.c;

        cols.add_operation.populate(blu, operand_1, operand_2);
        cols.operand_1 = Word::from(operand_1);
        cols.operand_2 = Word::from(operand_2);

        let is_real_instruction = event.pc != UNUSED_PC;
        if is_real_instruction {
            cols.is_retired = F::ONE;

            cols.state.populate(blu, event.clk);

            let instruction = program.fetch(event.pc);
            cols.adapter.populate(
                blu,
                instruction.op_a,
                event.a_record,
                instruction.op_b,
                event.b_record,
                instruction.op_c,
                event.c_record,
            );
        }
    }
}

impl<F> BaseAir<F> for AddChip {
    fn width(&self) -> usize {
        NUM_ADD_COLS
    }
}

impl<AB> Air<AB> for AddChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        builder.assert_bool(local.is_retired);
        // `is_retired` can only be set alongside `is_real` -- kept as its own witnessed column
        // (not the product `is_real * is_retired`) because interaction values/multiplicities in
        // this lookup argument must stay affine in the trace columns.
        builder.when_not(local.is_real).assert_zero(local.is_retired);

        // Evaluate the addition operation.
        AddOperation::<AB::F>::eval(
            builder,
            local.operand_1,
            local.operand_2,
            local.add_operation,
            local.is_real.into(),
        );

        // Register `a` is written `add_operation.value`, register `b` holds `operand_1`, and
        // register `c` holds `operand_2` -- a direct (not rearranged) mapping, unlike Sub's
        // `a = b - c` verified as `b = a + c`.
        let op_b_role = local.operand_1;
        let op_c_role = local.operand_2;

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: opcode/`op_a_0`/`imm_b`/
        // `imm_c` are compile-time constants (this chip only ever sees register-register ADD/ADDU
        // with `op_a != 0` -- any `op_a==0` row is routed to `AluX0Chip` instead, see
        // `RTypeReader`'s doc comment), and `op_b`/`op_c` are zero-extended from the adapter's
        // register-index columns. `send_program`'s lookup against `ProgramChip`'s preprocessed ROM
        // is what makes `op_a`/`op_b`/`op_c` trustworthy -- there's no separate opcode-binding
        // check needed, since the opcode here is never a variable a malicious prover could
        // substitute.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::ADD.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: Word::extend_var::<AB>(local.adapter.op_c),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::zero(),
        };
        builder.send_program(local.pc, instruction, local.is_retired.into());

        eval_r_type_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.add_operation.value.map(Into::into),
            local.is_retired.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_retired.into());

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
            local.is_retired.into(),
        );

        builder
            .when(local.is_retired)
            .assert_word_eq(local.adapter.op_b_val(), op_b_role.map(Into::into));
        builder
            .when(local.is_retired)
            .assert_word_eq(local.adapter.op_c_val(), op_c_role.map(Into::into));

        // ---- Synthetic dependency path: matches whichever chip generated this internal check via
        // `send_alu`/`send_alu_with_hi` (always at the `UNUSED_PC` sentinel, shard/clk zero).
        // `is_real - is_retired` is 1 exactly when this is a real ADD row that is *not* a real
        // instruction, i.e. a synthetic dependency row -- and stays affine (degree 1), unlike the
        // product `is_real * (1 - is_retired)`. ----
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
            zero_word,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_real - local.is_retired,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode, UNUSED_PC};
    use zkm_hypercube::air::MachineAir;

    use super::AddChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        // `UNUSED_PC` keeps this a synthetic-dependency-style row, so trace generation doesn't
        // need a real `Program` to fetch an instruction from.
        shard.add_events = vec![AluEvent::new(UNUSED_PC, Opcode::ADD, 14, 8, 6)];
        let chip = AddChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
