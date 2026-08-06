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
    operations::{ShiftLeftOperation, ShiftRightOperation},
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `ExtChip`.
pub const NUM_EXT_COLS: usize = size_of::<ExtCols<u8>>();

/// A chip that implements the MIPS bit-field extract instruction EXT.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `ext_events`. `op_a` may be any register (including register 0 -- routed to `AluX0Chip`
/// instead, see its doc comment, since the extracted result is then unobservable); `op_b` is
/// always a register and `op_c` is always the instruction's own encoded immediate (`msbd << 5 |
/// lsb`) -- the same shape `ITypeReaderNonZero` covers. EXT is a fresh write of `op_a` (not
/// read-modify-write), so no `prev_a_value` is needed; its final written value is fed directly
/// from the shift chain's own output (an affine expression, like `ShiftLeftChip`'s own migrated
/// adapter feed), so no separate masking is needed once `op_a==0` is routed away.
///
/// EXT's two intermediate shift steps (`sll_val = op_b << (31 - lsb - msbd)` then `op_a = sll_val
/// >> (31 - msbd)`) are verified locally via embedded `ShiftLeftOperation`/`ShiftRightOperation`
/// (no cross-chip lookup into `ShiftLeft`/`ShiftRightChip`).
#[derive(Default)]
pub struct ExtChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct ExtCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeReaderNonZero<T>,

    /// Lsb/Msb of the extracted field.
    pub lsb: T,
    pub msbd: T,

    /// `31 - lsb - msbd`, the shift amount for the intermediate SLL step below.
    pub sll_shift: T,
    /// `sll_val = op_b << sll_shift`, computed locally (no cross-chip lookup into `ShiftLeft`).
    pub sll_operation: ShiftLeftOperation<T>,

    /// `31 - msbd`, the shift amount for the final SRL step below.
    pub srl_shift: T,
    /// `op_a`'s written value = sll_val >> srl_shift`, computed locally (no cross-chip lookup
    /// into `ShiftRightChip`).
    pub srl_operation: ShiftRightOperation<T>,

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
        let nb_rows = next_multiple_of_32(
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
        cols.adapter.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
        );

        let lsb = event.c & 0x1f;
        let msbd = event.c >> 5;
        cols.lsb = F::from_canonical_u32(lsb);
        cols.msbd = F::from_canonical_u32(msbd);

        let sll_shift = 31 - lsb - msbd;
        cols.sll_shift = F::from_canonical_u32(sll_shift);
        let shift_left = cols.sll_operation.populate(blu, event.b, sll_shift);

        let srl_shift = 31 - msbd;
        cols.srl_shift = F::from_canonical_u32(srl_shift);
        cols.srl_operation.populate(blu, shift_left, srl_shift, false);

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

        let op_b_val = local.adapter.op_b_val();
        let op_c_val = local.adapter.op_c;

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode`/`op_a_0`/
        // `imm_b` are compile-time constants (this chip only ever sees a non-zero destination
        // and a register `op_b` -- see this chip's doc comment), `imm_c` is always set (EXT's
        // `op_c` is always the immediate `msbd << 5 | lsb`), and `op_b`/`op_c` are the adapter's
        // own columns.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::EXT.as_field::<AB::F>().into(),
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

        // Ext can be divided into 2 operations, each verified locally (no cross-chip lookup):
        //    sll_val = op_b << (31 - lsb - msbd)
        //    result = sll_val >> (31 - msbd)
        builder
            .when(is_real)
            .assert_eq(local.sll_shift, AB::Expr::from_canonical_u32(31) - local.msbd - local.lsb);
        let sll_val = ShiftLeftOperation::<AB::F>::eval(
            builder,
            op_b_val,
            // Only byte 0 of the shift-amount word is read by `eval`; the rest is unused padding.
            Word([local.sll_shift; 4]),
            local.sll_operation,
            is_real.into(),
        );

        builder.when(is_real).assert_eq(local.srl_shift, AB::Expr::from_canonical_u32(31) - local.msbd);
        let srl_result = ShiftRightOperation::<AB::F>::eval(
            builder,
            sll_val,
            Word([local.srl_shift; 4]),
            local.srl_operation,
            false,
            is_real.into(),
        );

        eval_i_type_reader_non_zero(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
            srl_result.map(Into::into),
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

        // op_c = (msbd << 5) + lsb
        builder.when(is_real).assert_eq(
            op_c_val.reduce::<AB>(),
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
