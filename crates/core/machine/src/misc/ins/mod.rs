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
    events::{ByteLookupEvent, ByteRecord, MiscEvent},
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
    air::ZKMCoreAirBuilder,
    operations::{AddOperation, FixedShiftRightOperation, ShiftLeftOperation, ShiftRightOperation},
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `InsChip`.
pub const NUM_INS_COLS: usize = size_of::<InsCols<u8>>();

/// A chip that implements the MIPS bit-field insert instruction INS.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `ins_events`. `op_a` may be any register (including register 0 -- routed to `AluX0Chip`
/// instead, see its doc comment, since the bit-inserted result is then unobservable and doesn't
/// need the shift/rotate chain verified at all); `op_b` is always a register and `op_c` is always
/// the instruction's own encoded immediate (`msb << 5 | lsb`) -- the same shape
/// `ITypeReaderNonZero` covers. INS is a read-modify-write of `op_a` (it preserves the untouched
/// bits of the previous value, read via the adapter's own `op_a_access.prev_value`), but its
/// final written value is fed directly from the shift/rotate/add chain's own output (an affine
/// expression, like `ShiftLeftChip`'s own migrated adapter feed), so no separate
/// `RegisterWriteAccessCols`-style masking is needed once `op_a==0` is routed away.
///
/// INS's shift-family intermediate steps (`ror_val`/`srl1_val`/`srl_val`/`sll_val`) are all
/// verified locally via embedded shift operations (no cross-chip lookup into
/// `ShiftLeft`/`ShiftRightChip`). `srl1_val`'s shift amount is the compile-time constant `1`, so
/// it uses the cheaper `FixedShiftRightOperation`; the other three use a witnessed (runtime)
/// shift amount, so they use `ShiftLeftOperation`/`ShiftRightOperation`.
#[derive(Default)]
pub struct InsChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct InsCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReaderNonZero<T>,

    /// Lsb/Msb of the insert field.
    pub lsb: T,
    pub msb: T,

    /// The INS decomposition extracts the upper bits of prev_a via a right shift by `width =
    /// msb - lsb + 1`. Since `ShiftRightOperation` only supports shift amounts 0-31, we split
    /// this into two steps: `>> 1` then `>> (msb - lsb)`, each of which is always in range
    /// [0, 31]. All shift/rotate sub-steps are computed locally (no cross-chip lookup into
    /// `ShiftLeft`/`ShiftRightChip`).

    /// `ror_val = rotate_right(prev_a, lsb)`.
    pub ror_operation: ShiftRightOperation<T>,
    /// `srl1_val = ror_val >> 1` (compile-time-constant shift amount).
    pub srl1_operation: FixedShiftRightOperation<T>,

    /// `msb - lsb`, the shift amount for the `srl_val` step below.
    pub msb_minus_lsb: T,
    /// `srl_val = srl1_val >> (msb - lsb)`.
    pub srl_operation: ShiftRightOperation<T>,

    /// `31 - msb + lsb`, the shift amount for the `sll_val` step below.
    pub sll_shift: T,
    /// `sll_val = op_b << (31 - msb + lsb)`.
    pub sll_operation: ShiftLeftOperation<T>,

    /// `add_val = srl_val + sll_val`, computed locally (no cross-chip lookup into `AddChip`).
    pub add_operation: AddOperation<T>,

    /// `31 - msb`, the shift amount for the final rotate step below.
    pub final_shift: T,
    /// `op_a`'s written value = rotate_right(add_val, 31 - msb)`.
    pub final_ror_operation: ShiftRightOperation<T>,

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
        cols.adapter.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
        );

        let lsb = event.c & 0x1f;
        let msb = event.c >> 5;
        cols.lsb = F::from_canonical_u32(lsb);
        cols.msb = F::from_canonical_u32(msb);

        let ror_val = cols.ror_operation.populate(blu, event.prev_a, lsb, true);
        let srl1_val = cols.srl1_operation.populate(blu, ror_val, 1);

        let msb_minus_lsb = msb - lsb;
        cols.msb_minus_lsb = F::from_canonical_u32(msb_minus_lsb);
        let srl_val = cols.srl_operation.populate(blu, srl1_val, msb_minus_lsb, false);

        let sll_shift = 31 - msb + lsb;
        cols.sll_shift = F::from_canonical_u32(sll_shift);
        let sll_val = cols.sll_operation.populate(blu, event.b, sll_shift);

        let add_val = cols.add_operation.populate(blu, srl_val, sll_val);

        let final_shift = 31 - msb;
        cols.final_shift = F::from_canonical_u32(final_shift);
        cols.final_ror_operation.populate(blu, add_val, final_shift, true);

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

        let prev_a_val = local.adapter.op_a_access.prev_value;
        let op_b_val = local.adapter.op_b_val();
        let op_c_val = local.adapter.op_c;

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode`/`op_a_0`/
        // `imm_b` are compile-time constants (this chip only ever sees a non-zero destination
        // and a register `op_b` -- see this chip's doc comment), `imm_c` is always set (INS's
        // `op_c` is always the immediate `msb << 5 | lsb`), and `op_b`/`op_c` are the adapter's
        // own columns.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::INS.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: op_c_val.map(Into::into),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.into());

        builder.when(is_real).assert_zero(op_c_val[2]);
        builder.when(is_real).assert_zero(op_c_val[3]);

        // Ins is decomposed into 6 sub-operations, each verified locally (no cross-chip lookup
        // into `ShiftLeft`/`ShiftRightChip`/`AddChip`):
        //    ror_val  = rotate_right(prev_a, lsb)            [shift: lsb ∈ 0..31]
        //    srl1_val = ror_val >> 1                          [shift: 1, fixed]
        //    srl_val  = srl1_val >> (msb - lsb)               [shift: msb-lsb ∈ 0..31]
        //    sll_val  = op_b << (31 - msb + lsb)              [shift: ∈ 0..31]
        //    add_val  = srl_val + sll_val
        //    result   = rotate_right(add_val, 31 - msb)       [shift: ∈ 0..31]
        let ror_val = ShiftRightOperation::<AB::F>::eval(
            builder,
            prev_a_val,
            // Only byte 0 of the shift-amount word is read by `eval`; the rest is unused padding.
            Word([local.lsb; 4]),
            local.ror_operation,
            true,
            is_real.into(),
        );

        FixedShiftRightOperation::<AB::F>::eval(
            builder,
            ror_val,
            1,
            local.srl1_operation,
            is_real.into(),
        );
        let srl1_val = local.srl1_operation.value;

        builder
            .when(is_real)
            .assert_eq(local.msb_minus_lsb, Into::<AB::Expr>::into(local.msb) - local.lsb);
        let srl_val = ShiftRightOperation::<AB::F>::eval(
            builder,
            srl1_val,
            Word([local.msb_minus_lsb; 4]),
            local.srl_operation,
            false,
            is_real.into(),
        );

        builder.when(is_real).assert_eq(
            local.sll_shift,
            AB::Expr::from_canonical_u32(31) - local.msb + local.lsb,
        );
        let sll_val = ShiftLeftOperation::<AB::F>::eval(
            builder,
            op_b_val,
            Word([local.sll_shift; 4]),
            local.sll_operation,
            is_real.into(),
        );

        // `add_val = srl_val + sll_val`, computed locally (no cross-chip lookup into `AddChip`).
        AddOperation::<AB::F>::eval(builder, srl_val, sll_val, local.add_operation, is_real.into());
        let add_val = local.add_operation.value;

        builder
            .when(is_real)
            .assert_eq(local.final_shift, AB::Expr::from_canonical_u32(31) - local.msb);
        let final_result = ShiftRightOperation::<AB::F>::eval(
            builder,
            add_val,
            Word([local.final_shift; 4]),
            local.final_ror_operation,
            true,
            is_real.into(),
        );

        eval_i_type_reader_non_zero(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            final_result.map(Into::into),
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

        // op_c = (msb << 5) + lsb
        builder.when(is_real).assert_eq(
            op_c_val.reduce::<AB>(),
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
