//! CLO and CLZ verification.
//!
//! This module implements the verification logic for clz and clo operations. It ensures
//! that for any given input b and outputs the leading zero/one count.
//!
//! First, we prove the CLZ.
//! if b == 0, then clz(b) = 32
//! if b > 0, then b >> (32 - (result + 1)) == 1 && b >> (32 - result) == 0
//!
//! Second, we prove the CLO.
//! we use clo(b) = clz(0xffffffff - b)

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemoryRecordEnum},
    ByteOpcode, ExecutionRecord, Opcode, Program, UNUSED_PC,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{
    air::MachineAir,
    word::Word,
};

use crate::{
    adapter::InstructionCols,
    adapter::{
        clk_low_expr, eval_cpu_state, eval_register_reader, eval_state_chain, CpuState,
        RegisterReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::MemoryCols,
    operations::ShiftRightOperation,
    utils::{next_power_of_two, pad_rows_fixed},
    CoreChipError,
};

/// The number of main trace columns for `CloClzChip`.
pub const NUM_CLOCLZ_COLS: usize = size_of::<CloClzCols<u8>>();

/// A chip that implements addition for the opcodes CLO/CLZ.
///
/// As with `AddChip`, not every row is a real retired instruction -- though as of this
/// writing no other chip emits a synthetic dependency row into `cloclz_events`, the
/// `is_real_instruction` flag is kept for consistency with the other opcode-family chips.
///
/// `bb >> (31 - result) == 1` is verified locally via an embedded `ShiftRightOperation` (no
/// cross-chip lookup into `ShiftRightChip`).
#[derive(Default)]
pub struct CloClzChip;

/// The column layout for the chip.
///
/// Optimized: `sr1` removed (hardcoded as 1 in SRL lookup since we always verify sr1 == 1),
/// `is_clo` removed (derived as `is_real - is_clz`).
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct CloClzCols<T: Copy> {
    /// The current shard and clk. Only meaningful when `is_real_instruction` is set.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The raw fetched instruction. Only meaningful when `is_real_instruction` is set.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`. Only meaningful when `is_real_instruction` is set.
    pub reader: RegisterReader<T>,

    /// Whether this row is a real, retired CLZ/CLO instruction (as opposed to an internal
    /// dependency check from another chip, or padding).
    pub is_real_instruction: T,

    /// The result
    pub a: Word<T>,

    /// The input operand.
    pub b: Word<T>,

    /// if clo, bb == 0xffffffff - b
    /// if clz, bb == b
    pub bb: Word<T>,

    /// whether the `bb` is zero.
    pub is_bb_zero: T,

    /// `31 - a[0]`, the shift amount used to verify `bb >> shift_amount == 1` below. Only
    /// meaningful when `is_bb_zero` is unset (when `is_bb_zero` is set, `a[0] == 32` and this
    /// would underflow, but the shift check is gated off in that case anyway).
    pub shift_amount: T,

    /// `bb >> shift_amount`, computed locally (no cross-chip lookup into `ShiftRightChip`).
    pub shift_right_operation: ShiftRightOperation<T>,

    /// Flag to indicate whether the opcode is CLZ.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_clz: T,

    /// Selector to know whether this row is enabled.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for CloClzChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "CloClz".to_string()
    }

    fn local_only(&self) -> bool {
        true
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        CloClzCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.cloclz_events.len(),
            None,
            <CloClzChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let mut rows: Vec<[F; NUM_CLOCLZ_COLS]> = vec![];
        let cloclz_events = input.cloclz_events.clone();
        for event in cloclz_events.iter() {
            assert!(event.opcode == Opcode::CLZ || event.opcode == Opcode::CLO);
            let mut row = [F::ZERO; NUM_CLOCLZ_COLS];
            let cols: &mut CloClzCols<F> = row.as_mut_slice().borrow_mut();

            cols.a = Word::from(event.a);
            cols.b = Word::from(event.b);
            cols.pc = F::from_canonical_u32(event.pc);
            cols.next_pc = F::from_canonical_u32(event.next_pc);
            cols.is_real = F::ONE;
            cols.is_clz = F::from_bool(event.opcode == Opcode::CLZ);

            // Default: not a real instruction, so the register reader's b/c memory accesses
            // have zero multiplicity unless overwritten by a real fetched instruction's actual
            // immediate flags just below.
            cols.instruction.imm_b = F::ONE;
            cols.instruction.imm_c = F::ONE;

            let is_real_instruction = event.pc != UNUSED_PC;
            cols.is_real_instruction = F::from_bool(is_real_instruction);
            if is_real_instruction {
                cols.state.populate(output, event.clk);

                let instruction = input.program.fetch(event.pc);
                cols.instruction.populate(&instruction);

                *cols.reader.op_a_access.value_mut() = event.a.into();
                *cols.reader.op_b_access.value_mut() = event.b.into();
                *cols.reader.op_c_access.value_mut() = event.c.into();

                if let Some(record) = event.a_record {
                    cols.reader.op_a_access.populate(record, output);
                }
                if let Some(MemoryRecordEnum::Read(record)) = event.b_record {
                    cols.reader.op_b_access.populate(record, output);
                }
                if let Some(MemoryRecordEnum::Read(record)) = event.c_record {
                    cols.reader.op_c_access.populate(record, output);
                }
                cols.reader.populate_op_a_range_checks(output);
            }

            let bb = if event.opcode == Opcode::CLZ { event.b } else { 0xffffffff - event.b };
            cols.bb = Word::from(bb);

            // if bb == 0, then result is 32.
            let is_bb_zero = bb == 0;
            cols.is_bb_zero = F::from_bool(is_bb_zero);

            // Range check.
            output.add_u8_range_checks(&bb.to_le_bytes());
            output.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: event.a as u8,
                c: 33,
            });

            // Verify `bb >> shift_amount == 1` locally (no cross-chip lookup into
            // `ShiftRightChip`). Only meaningful when `bb != 0`; default to 0 otherwise since
            // `31 - 32` would underflow. `populate` is only called when `!is_bb_zero`, matching
            // `eval`'s gating exactly -- otherwise its (unconditionally emitted) byte-lookup
            // events would have no matching send, unbalancing the lookup argument.
            if !is_bb_zero {
                let shift_amount = 31 - event.a;
                cols.shift_amount = F::from_canonical_u32(shift_amount);
                cols.shift_right_operation.populate(output, bb, shift_amount, false);
            }

            rows.push(row);
        }

        // Pad the trace to a power of two depending on the proof shape in `input`.
        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_CLOCLZ_COLS],
            None,
            <CloClzChip as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_CLOCLZ_COLS);

        // Create the template for the padded rows. These are fake rows that don't fail on some
        // sanity checks.
        let padded_row_template = {
            let mut row = [F::ZERO; NUM_CLOCLZ_COLS];
            let cols: &mut CloClzCols<F> = row.as_mut_slice().borrow_mut();
            // Padding rows: is_real=0, is_clz=0, is_bb_zero=1, a=32.
            // is_bb_zero=1 gates off the embedded `ShiftRightOperation`'s constraints.
            cols.a = Word::from(32);
            cols.is_bb_zero = F::ONE;
            cols.instruction.imm_b = F::ONE;
            cols.instruction.imm_c = F::ONE;

            row
        };
        debug_assert!(padded_row_template.len() == NUM_CLOCLZ_COLS);
        for i in input.cloclz_events.len() * NUM_CLOCLZ_COLS..trace.values.len() {
            trace.values[i] = padded_row_template[i % NUM_CLOCLZ_COLS];
        }

        Ok(trace)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.cloclz_events.is_empty()
    }
}

impl<F> BaseAir<F> for CloClzChip {
    fn width(&self) -> usize {
        NUM_CLOCLZ_COLS
    }
}

impl<AB> Air<AB> for CloClzChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &CloClzCols<AB::Var> = (*local).borrow();
        let one: AB::Expr = AB::F::ONE.into();
        let zero: AB::Expr = AB::F::ZERO.into();


        // Derive is_clo from is_real and is_clz.
        let is_clo: AB::Expr = local.is_real.into() - local.is_clz.into();

        // if clz, bb == b, else bb = !b
        {
            local.b.0.iter().zip_eq(local.bb.0.iter()).for_each(|(a, b)| {
                builder.when(is_clo.clone()).assert_eq(*a + *b, AB::Expr::from_canonical_u32(255));
                builder.when(local.is_clz).assert_eq(*a, *b);
            });

            builder.slice_range_check_u8(&local.bb.0, local.is_real);
        }

        // ensure result < 33
        // Send the comparison lookup.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::F::ONE,
            local.a[0],
            AB::Expr::from_canonical_u8(33),
            local.is_real,
        );

        builder.when(local.is_real).assert_zero(local.a[1]);
        builder.when(local.is_real).assert_zero(local.a[2]);
        builder.when(local.is_real).assert_zero(local.a[3]);

        // Get the opcode for the operation.
        // is_clo = is_real - is_clz, so:
        //   opcode = (is_real - is_clz) * CLO + is_clz * CLZ
        //          = is_real * CLO + is_clz * (CLZ - CLO)
        let cpu_opcode = is_clo.clone() * Opcode::CLO.as_field::<AB::F>()
            + local.is_clz * Opcode::CLZ.as_field::<AB::F>();

        builder.assert_bool(local.is_real_instruction);
        builder.when_not(local.is_real).assert_zero(local.is_real_instruction);

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        builder.send_program(local.pc, local.instruction, local.is_real_instruction);

        eval_register_reader(
            builder,
            &local.reader,
            clk_high.clone(),
            clk_low.clone(),
            &local.instruction,
            // Gated by `is_real_instruction`: `register.rs`'s `assert_word_eq(op_a_value,
            // reader.op_a_val())` fires unconditionally whenever `op_a_0` is unset, which it is
            // by default on synthetic rows (their `instruction` column is never populated) --
            // an ungated `op_a_value` would then have to equal `reader.op_a_val()` (always zero
            // on synthetic rows) even when the true result is nonzero.
            local.a.map(|x| local.is_real_instruction.into() * Into::<AB::Expr>::into(x)),
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real_instruction.into(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real_instruction.into());

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
            local.is_real_instruction.into(),
        );

        builder
            .when(local.is_real_instruction)
            .assert_word_eq(local.reader.op_b_val(), local.b.map(Into::into));

        // Bind the row's actual fetched opcode to whichever variant is real, so a
        // real-instruction row can't claim the wrong CLZ/CLO variant while still passing the
        // program lookup. These gates are plain assertions (not interaction
        // values/multiplicities), so the product `is_real_instruction * is_X` (degree 2) is
        // fine here.
        builder
            .when(Into::<AB::Expr>::into(local.is_real_instruction) * local.is_clz)
            .assert_eq(local.instruction.opcode, Opcode::CLZ.as_field::<AB::F>());
        builder
            .when(Into::<AB::Expr>::into(local.is_real_instruction) * is_clo.clone())
            .assert_eq(local.instruction.opcode, Opcode::CLO.as_field::<AB::F>());

        // ---- Synthetic dependency path: matches whichever chip generated this internal check via
        // `send_alu` (always at the `UNUSED_PC` sentinel, shard/clk zero). No current producer
        // targets `cloclz_events`, so this multiplicity is always zero today, but the wiring is
        // kept for consistency and to stay sound if a future dependency producer is added. ----
        builder.receive_instruction(
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.pc,
            local.next_pc,
            local.next_pc + AB::Expr::from_canonical_u32(4),
            AB::Expr::zero(),
            cpu_opcode,
            local.a,
            local.b,
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            local.is_real - local.is_real_instruction,
        );

        // if is_bb_zero == 1, bb == 0, and result is 32
        {
            builder.assert_bool(local.is_bb_zero);

            builder.when(local.is_bb_zero).assert_zero(local.bb.reduce::<AB>());
            builder.when(local.is_bb_zero).assert_zero(local.bb[3]);

            builder.when(local.is_bb_zero).assert_eq(local.a[0], AB::Expr::from_canonical_u32(32));
        }

        {
            // Verify bb >> (31 - result) == 1 locally, embedding `ShiftRightOperation` rather
            // than a cross-chip lookup into `ShiftRightChip`. Since the result is always 1 when
            // bb != 0, we hardcode the expected value as Word([1, 0, 0, 0]) directly, eliminating
            // 4 witness columns. `a[1]`/`a[2]`/`a[3]` are already proven zero whenever `is_real`
            // (below), so they're reused as the upper 3 (zero) bytes of the shift amount.
            let shift_is_real = one.clone() - local.is_bb_zero;
            builder
                .when(shift_is_real.clone())
                .assert_eq(local.shift_amount, AB::Expr::from_canonical_u32(31) - local.a[0]);

            let shift_result = ShiftRightOperation::<AB::F>::eval(
                builder,
                local.bb,
                Word([local.shift_amount, local.a[1], local.a[2], local.a[3]]),
                local.shift_right_operation,
                false,
                shift_is_real.clone(),
            );
            builder.when(shift_is_real).assert_word_eq(
                shift_result,
                Word([one.clone(), zero.clone(), zero.clone(), zero.clone()]),
            );
        }

        // is_clz and is_real are boolean; is_clo = is_real - is_clz must also be boolean,
        // which is equivalent to: is_clz = 1 implies is_real = 1.
        builder.assert_bool(local.is_clz);
        builder.assert_bool(local.is_real);
        builder.when(local.is_clz).assert_one(local.is_real);
    }
}

#[cfg(test)]
mod tests {
    // use crate::utils::{uni_stark_prove, uni_stark_verify};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode, UNUSED_PC};
    use zkm_hypercube::air::MachineAir;
    // use zkm_stark::{
    //     air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig,
    // };

    use super::CloClzChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.cloclz_events = vec![
            AluEvent::new(UNUSED_PC, Opcode::CLZ, 32, 0, 0),
            AluEvent::new(UNUSED_PC, Opcode::CLZ, 8, 0x00800000, 0),
            AluEvent::new(UNUSED_PC, Opcode::CLZ, 0, 0xffffffff, 0),
            AluEvent::new(UNUSED_PC, Opcode::CLO, 32, 0xffffffff, 0),
            AluEvent::new(UNUSED_PC, Opcode::CLO, 8, 0xff7fffff, 0),
            AluEvent::new(UNUSED_PC, Opcode::CLO, 0, 0, 0),
        ];
        let chip = CloClzChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    #[ignore = "no zkm-hypercube single-chip prove/verify utility yet (old FRI-backed uni_stark_prove/verify removed)"]
    fn prove_koalabear() {
        // let config = KoalaBearPoseidon2::new();
        // let mut challenger = config.challenger();
        //
        // let mut cloclz_events: Vec<AluEvent> = Vec::new();
        //
        // let clo_clzs: Vec<(Opcode, u32, u32, u32)> = vec![
        //     (Opcode::CLZ, 32, 0, 0),
        //     (Opcode::CLZ, 8, 0x00800000, 0),
        //     (Opcode::CLZ, 0, 0xffffffff, 0),
        //     (Opcode::CLO, 32, 0xffffffff, 0),
        //     (Opcode::CLO, 8, 0xff7fffff, 0),
        //     (Opcode::CLO, 0, 0, 0),
        // ];
        // for t in clo_clzs.iter() {
        //     cloclz_events.push(AluEvent::new(0, t.0, t.1, t.2, t.3));
        // }
        //
        // // Append more events until we have 1000 tests.
        // for _ in 0..(1000 - clo_clzs.len()) {
        //     cloclz_events.push(AluEvent::new(0, Opcode::CLZ, 32, 0, 0));
        // }
        //
        // let mut shard = ExecutionRecord::default();
        // shard.cloclz_events = cloclz_events;
        // let chip = CloClzChip::default();
        // let trace: RowMajorMatrix<KoalaBear> =
        //     chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        // let proof =
        //     uni_stark_prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        //
        // let mut challenger = config.challenger();
        // uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
