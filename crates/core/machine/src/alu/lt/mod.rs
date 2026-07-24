use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
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
    adapter::{
        clk_low_expr, eval_cpu_state, eval_r_type_reader, eval_state_chain, CpuState,
        InstructionCols, RTypeReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    operations::LtOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

mod slti;
pub use slti::*;

/// The number of main trace columns for `LtChip`.
pub const NUM_LT_COLS: usize = size_of::<LtCols<u8>>();

/// A chip that implements the register-form opcodes SLT and SLTU.
///
/// Not every row corresponds to a real retired instruction: some rows are internal dependency
/// checks emitted by `Branch`/`DivRem` (reusing this chip's comparison circuit for internal
/// SLT/SLTU checks) at the `UNUSED_PC` sentinel. `is_retired` distinguishes the two (kept as its
/// own witnessed column rather than folded into a product, since interaction
/// values/multiplicities in this lookup argument must stay affine in the trace columns).
///
/// `Opcode::SLT`/`Opcode::SLTU` also cover the immediate-form SLTI/SLTIU (register `b` + an
/// encoded immediate `c`, see `SltiChip`) -- every real row reaching *this* chip is therefore
/// genuine register-register SLT/SLTU with a non-zero destination (any `op_a==0` row is routed to
/// `AluX0Chip` instead), which is what lets it use the narrow `RTypeReader` (see its doc comment)
/// instead of the generic `InstructionCols`+`RegisterReader` pair.
#[derive(Default)]
pub struct LtChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct LtCols<T: Copy> {
    /// The current shard and clk. Only meaningful when `is_retired == 1`.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`. Only meaningful when this row is a real
    /// instruction (`is_retired == 1`).
    pub adapter: RTypeReader<T>,

    /// Whether this row is a real, retired SLT/SLTU instruction (as opposed to an internal
    /// dependency check from another chip, or padding).
    pub is_retired: T,

    /// The first input operand (register `b` for a real instruction).
    pub b: Word<T>,

    /// The second input operand (register `c` for a real instruction).
    pub c: Word<T>,

    /// The SLT/SLTU comparison circuit (shared with `SltiChip`).
    pub lt_operation: LtOperation<T>,
}

impl LtCols<u32> {
    pub fn from_trace_row<F: PrimeField32>(row: &[F]) -> Self {
        let sized: [u32; NUM_LT_COLS] =
            row.iter().map(|x| x.as_canonical_u32()).collect::<Vec<u32>>().try_into().unwrap();
        *sized.as_slice().borrow()
    }
}

impl<F: PrimeField32> MachineAir<F> for LtChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Lt".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.lt_events.len(),
            None,
            <LtChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        LtCols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let padded_nb_rows = <LtChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LT_COLS);
        let chunk_size = std::cmp::max((input.lt_events.len() + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_LT_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_LT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LtCols<F> = row.borrow_mut();

                    if idx < input.lt_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.lt_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events, &input.program);
                    }
                    // A padding row is left all-zero: `is_retired` defaults to 0, which gates
                    // every interaction below to zero multiplicity on its own -- unlike the
                    // generic `RegisterReader`, `RTypeReader` needs no separate "force immediate
                    // flags" workaround for this.
                });
            },
        );

        // Convert the trace to a row major matrix.

        Ok(RowMajorMatrix::new(values, NUM_LT_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.lt_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .lt_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_LT_COLS];
                    let cols: &mut LtCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.lt_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LtChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut LtCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.b = Word::from(event.b);
        cols.c = Word::from(event.c);

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

        let result = cols.lt_operation.populate(blu, event.opcode, event.b, event.c);
        assert_eq!(result, event.a);
    }
}

impl<F> BaseAir<F> for LtChip {
    fn width(&self) -> usize {
        NUM_LT_COLS
    }
}

impl<AB> Air<AB> for LtChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LtCols<AB::Var> = (*local).borrow();


        builder.assert_bool(local.is_retired);
        let is_real = local.lt_operation.is_slt + local.lt_operation.is_sltu;
        // `is_retired` can only be set alongside `is_real` -- kept as its own witnessed column
        // (not the product `is_real * is_retired`) because interaction values/multiplicities in
        // this lookup argument must stay affine in the trace columns.
        builder.when_not(is_real.clone()).assert_zero(local.is_retired);

        LtOperation::<AB::F>::eval(
            builder,
            local.b.map(Into::into),
            local.c.map(Into::into),
            local.lt_operation,
            is_real.clone(),
        );

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: opcode is selected by
        // `LtOperation`'s own `is_slt`/`is_sltu`, `op_a`/`op_b`/`op_c` are zero-extended from the
        // adapter's register-index columns, and `op_a_0`/`imm_b`/`imm_c` are compile-time
        // constants (this chip only ever sees register-register SLT/SLTU with `op_a != 0` -- any
        // `op_a==0` row is routed to `AluX0Chip` instead, see `RTypeReader`'s doc comment).
        let opcode = local.lt_operation.is_slt * Opcode::SLT.as_field::<AB::F>()
            + local.lt_operation.is_sltu * Opcode::SLTU.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: opcode.clone(),
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
            local.lt_operation.a.map(Into::into),
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
            .assert_word_eq(local.adapter.op_b_val(), local.b.map(Into::into));
        builder
            .when(local.is_retired)
            .assert_word_eq(local.adapter.op_c_val(), local.c.map(Into::into));

        // ---- Synthetic dependency path: matches whichever chip generated this internal check via
        // `send_alu` (always at the `UNUSED_PC` sentinel, shard/clk zero). `is_real - is_retired`
        // is 1 exactly when this is a real SLT/SLTU row that is *not* a real instruction, i.e. a
        // synthetic dependency row -- and stays affine (degree 1), unlike the product
        // `is_real * (1 - is_retired)`. ----
        let zero_word =
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]);

        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.next_pc,
            local.next_pc + AB::Expr::from_canonical_u32(4),
            AB::Expr::zero(),
            opcode,
            local.lt_operation.a,
            local.b,
            local.c,
            zero_word,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            is_real - local.is_retired,
        );
    }
}

#[cfg(test)]
mod tests {

    // use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode, UNUSED_PC};
    use zkm_hypercube::air::MachineAir;
    // use zkm_stark::{
    //     air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig,
    // };

    use super::LtChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.lt_events = vec![AluEvent::new(UNUSED_PC, Opcode::SLT, 0, 3, 2)];
        let chip = LtChip::default();
        let generate_trace = chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        let trace: RowMajorMatrix<KoalaBear> = generate_trace;
        println!("{:?}", trace.values)
    }

    // fn prove_koalabear_template(shard: &mut ExecutionRecord) {
    //     let config = KoalaBearPoseidon2::new();
    //     let mut challenger = config.challenger();
    //
    //     let chip = LtChip::default();
    //     let trace: RowMajorMatrix<KoalaBear> =
    //         chip.generate_trace(shard, &mut ExecutionRecord::default()).unwrap();
    //     let proof = prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
    //
    //     let mut challenger = config.challenger();
    //     verify(&config, &chip, &mut challenger, &proof).unwrap();
    // }

    #[test]
    #[ignore = "no zkm-hypercube single-chip prove/verify utility yet (old FRI-backed uni_stark_prove/verify removed)"]
    fn prove_koalabear_slt() {
        // let mut shard = ExecutionRecord::default();
        //
        // const NEG_3: u32 = 0b11111111111111111111111111111101;
        // const NEG_4: u32 = 0b11111111111111111111111111111100;
        // shard.lt_events = vec![
        //     // 0 == 3 < 2
        //     AluEvent::new(0, Opcode::SLT, 0, 3, 2),
        //     // 1 == 2 < 3
        //     AluEvent::new(0, Opcode::SLT, 1, 2, 3),
        //     // 0 == 5 < -3
        //     AluEvent::new(0, Opcode::SLT, 0, 5, NEG_3),
        //     // 1 == -3 < 5
        //     AluEvent::new(0, Opcode::SLT, 1, NEG_3, 5),
        //     // 0 == -3 < -4
        //     AluEvent::new(0, Opcode::SLT, 0, NEG_3, NEG_4),
        //     // 1 == -4 < -3
        //     AluEvent::new(0, Opcode::SLT, 1, NEG_4, NEG_3),
        //     // 0 == 3 < 3
        //     AluEvent::new(0, Opcode::SLT, 0, 3, 3),
        //     // 0 == -3 < -3
        //     AluEvent::new(0, Opcode::SLT, 0, NEG_3, NEG_3),
        // ];
        //
        // prove_koalabear_template(&mut shard);
    }

    #[test]
    #[ignore = "no zkm-hypercube single-chip prove/verify utility yet (old FRI-backed uni_stark_prove/verify removed)"]
    fn prove_koalabear_sltu() {
        // let mut shard = ExecutionRecord::default();
        //
        // const LARGE: u32 = 0b11111111111111111111111111111101;
        // shard.lt_events = vec![
        //     // 0 == 3 < 2
        //     AluEvent::new(0, Opcode::SLTU, 0, 3, 2),
        //     // 1 == 2 < 3
        //     AluEvent::new(0, Opcode::SLTU, 1, 2, 3),
        //     // 0 == LARGE < 5
        //     AluEvent::new(0, Opcode::SLTU, 0, LARGE, 5),
        //     // 1 == 5 < LARGE
        //     AluEvent::new(0, Opcode::SLTU, 1, 5, LARGE),
        //     // 0 == 0 < 0
        //     AluEvent::new(0, Opcode::SLTU, 0, 0, 0),
        //     // 0 == LARGE < LARGE
        //     AluEvent::new(0, Opcode::SLTU, 0, LARGE, LARGE),
        // ];
        //
        // prove_koalabear_template(&mut shard);
    }
}
