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
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{
        clk_low_expr, eval_cpu_state, eval_i_type_reader_non_zero, eval_state_chain, CpuState,
        ITypeReaderNonZero, InstructionCols,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    operations::ShiftRightOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `CloClzChip`.
pub const NUM_CLOCLZ_COLS: usize = size_of::<CloClzCols<u8>>();

/// A chip that implements addition for the opcodes CLO/CLZ.
///
/// `op_a` is written and may be any register (including register 0 -- routed to `AluX0Chip`
/// instead, see its doc comment); `op_b` is always a register and `op_c` is always the
/// instruction's own encoded immediate (always exactly `0`, never read) -- the same shape
/// `ITypeReaderNonZero` covers. Nothing ever emits a synthetic dependency row into `cloclz_events`
/// and this chip never produces one either -- every row here is a real, retired instruction.
///
/// `bb >> (31 - result) == 1` is verified locally via an embedded `ShiftRightOperation` (no
/// cross-chip lookup into `ShiftRightChip`).
#[derive(Default)]
pub struct CloClzChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct CloClzCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReaderNonZero<T>,

    /// The result.
    pub a: Word<T>,

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

    /// Flag to indicate whether the opcode is CLO.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_clo: T,
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
        let chunk_size = std::cmp::max(input.cloclz_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <CloClzChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_CLOCLZ_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_CLOCLZ_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_CLOCLZ_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut CloClzCols<F> = row.borrow_mut();

                    if idx < input.cloclz_events.len() {
                        let event = &input.cloclz_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    } else {
                        // Padding row: `is_clz`/`is_clo` default to 0, which gates the program
                        // lookup/register-access interactions to zero multiplicity on its own --
                        // unlike the generic `RegisterReader`, this needs no separate "force
                        // immediate flags" workaround. But the `is_bb_zero`/shift-right
                        // verification block below is *not* gated by `is_clz`/`is_clo` at all (it
                        // holds unconditionally on every row, real or padding), so `is_bb_zero=1`/
                        // `a[0]=32` must be set here to make it trivially true, gating off the
                        // embedded `ShiftRightOperation`'s constraints.
                        cols.a = Word::from(32u32);
                        cols.is_bb_zero = F::ONE;
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_CLOCLZ_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.cloclz_events.is_empty()
    }
}

impl CloClzChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut CloClzCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        // Every `cloclz_events` row is a real, retired instruction -- nothing ever produces a
        // synthetic dependency row here.
        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);
        cols.adapter.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
        );

        cols.a = Word::from(event.a);
        cols.is_clz = F::from_bool(event.opcode == Opcode::CLZ);
        cols.is_clo = F::from_bool(event.opcode == Opcode::CLO);

        let bb = if event.opcode == Opcode::CLZ { event.b } else { 0xffffffff - event.b };
        cols.bb = Word::from(bb);

        // if bb == 0, then result is 32.
        let is_bb_zero = bb == 0;
        cols.is_bb_zero = F::from_bool(is_bb_zero);

        // Range check.
        blu.add_u8_range_checks(&bb.to_le_bytes());
        blu.add_byte_lookup_event(ByteLookupEvent {
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
            cols.shift_right_operation.populate(blu, bb, shift_amount, false);
        }
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

        builder.assert_bool(local.is_clz);
        builder.assert_bool(local.is_clo);
        let is_real = local.is_clz + local.is_clo;
        builder.assert_bool(is_real.clone());

        let op_b_val = local.adapter.op_b_val();

        // if clz, bb == b, else bb = !b
        {
            op_b_val.0.iter().zip_eq(local.bb.0.iter()).for_each(|(a, b)| {
                builder
                    .when(local.is_clo)
                    .assert_eq(*a + *b, AB::Expr::from_canonical_u32(255));
                builder.when(local.is_clz).assert_eq(*a, *b);
            });

            builder.slice_range_check_u8(&local.bb.0, is_real.clone());
        }

        // ensure result < 33
        // Send the comparison lookup.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::F::ONE,
            local.a[0],
            AB::Expr::from_canonical_u8(33),
            is_real.clone(),
        );

        builder.when(is_real.clone()).assert_zero(local.a[1]);
        builder.when(is_real.clone()).assert_zero(local.a[2]);
        builder.when(is_real.clone()).assert_zero(local.a[3]);

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode` is a degree-1
        // linear combination of the one-hot selectors above, `op_a_0`/`imm_b` are compile-time
        // constants (this chip only ever sees a non-zero destination and a register `op_b` --
        // see this chip's doc comment), `imm_c` is always set (CLZ/CLO's `op_c` is always the
        // immediate 0), and `op_b`/`op_c` are the adapter's own columns.
        let opcode = local.is_clz * Opcode::CLZ.as_field::<AB::F>()
            + local.is_clo * Opcode::CLO.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        eval_i_type_reader_non_zero(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            local.a.map(Into::into),
            is_real.clone(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.clone());

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
            is_real.clone(),
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
            // (above), so they're reused as the upper 3 (zero) bytes of the shift amount.
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
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::CloClzChip;

    #[test]
    fn generate_trace() {
        // Every `cloclz_events` row is a real, retired instruction (see this chip's doc
        // comment), so trace-gen always does a real program lookup, unlike the old
        // `UNUSED_PC`-based synthetic-dependency test this replaced.
        let program = Program {
            instructions: vec![Instruction::new(Opcode::CLZ, 30, 29, 0, false, true)],
            pc_start: 0,
            pc_base: 0,
            ..Default::default()
        };
        let shard = ExecutionRecord {
            program: program.into(),
            cloclz_events: vec![
                AluEvent::new(0, Opcode::CLZ, 32, 0, 0),
                AluEvent::new(0, Opcode::CLZ, 8, 0x00800000, 0),
                AluEvent::new(0, Opcode::CLZ, 0, 0xffffffff, 0),
                AluEvent::new(0, Opcode::CLO, 32, 0xffffffff, 0),
                AluEvent::new(0, Opcode::CLO, 8, 0xff7fffff, 0),
                AluEvent::new(0, Opcode::CLO, 0, 0, 0),
            ],
            ..Default::default()
        };
        let chip = CloClzChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
