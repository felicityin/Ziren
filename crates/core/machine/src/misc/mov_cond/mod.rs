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
    events::{ByteLookupEvent, ByteRecord, MemoryAccessPosition, MovCondEvent},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{
    air::{BaseAirBuilder, MachineAir, ZKMAirBuilder},
    word::Word,
};

use crate::{
    adapter::InstructionCols,
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{RegisterAccessCols, RegisterWriteAccessCols},
    CoreChipError,
};

use crate::operations::IsZeroWordOperation;

use crate::utils::{next_multiple_of_32, zeroed_f_vec};

/// The number of main trace columns for `MovCondChip`.
pub const NUM_MOV_COND_COLS: usize = size_of::<MovCondCols<u8>>();

/// A chip that implements condition mov for the opcode MNE，MEQ (and the unrelated WSBH,
/// grouped here for chip-size reasons).
///
/// `op_b` is always a register and `op_c` is either a register (MEQ/MNE) or the instruction's own
/// immediate (WSBH, whose `op_c` is always the constant 0) -- the same shape `AluTypeReader`
/// covers, except `op_a` here is a genuine read-modify-write (MEQ/MNE keep `op_a`'s old value
/// when the move condition is false), so its write value is a masked/conditional expression, not
/// a chip-computed value that can be sent directly -- this needs `RegisterWriteAccessCols`'s own
/// witnessed `value` column (see its doc comment), unlike `AluTypeReader`'s plain
/// `RegisterAccessCols`. `op_a==0` rows are routed to `AluX0Chip` instead (see its doc comment),
/// so `op_a` here is guaranteed never register 0 and needs no masking for that case.
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

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The register index of `op_a` (read-modify-write, never register 0).
    pub op_a: T,
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// The register index of `op_b` (read; always a register for MEQ/MNE/WSBH).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// Either the register index of `op_c` (byte 0, for MEQ/MNE) or the constant 0 (for WSBH,
    /// whose `op_c` is always the immediate 0 -- `is_wsbh` doubles as `imm_c` here, since WSBH is
    /// the only opcode this chip supports with an immediate `op_c`).
    pub op_c: Word<T>,
    pub op_c_access: RegisterAccessCols<T>,

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
        let nb_rows = next_multiple_of_32(
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
        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);

        cols.op_a = F::from_canonical_u8(instruction.op_a);
        if let Some(record) = event.a_record {
            cols.op_a_access.populate(record, blu);
        }

        cols.op_b = F::from_canonical_u32(instruction.op_b);
        if let Some(record) = event.b_record {
            cols.op_b_access.populate(record, blu);
        }

        cols.is_meq = F::from_bool(matches!(event.opcode, Opcode::MEQ));
        cols.is_mne = F::from_bool(matches!(event.opcode, Opcode::MNE));
        cols.is_wsbh = F::from_bool(matches!(event.opcode, Opcode::WSBH));

        // `is_wsbh` doubles as `imm_c` (see `MovCondCols::op_c`'s doc comment): WSBH always
        // decodes with `imm_c=true`/`op_c=0`, MEQ/MNE always with `imm_c=false`.
        cols.op_c = Word::from(instruction.op_c);
        if instruction.imm_c {
            cols.op_c_access.prev_value = cols.op_c;
        } else if let Some(record) = event.c_record {
            cols.op_c_access.populate(record, blu);
        }

        cols.c_eq_0.populate(event.c);
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

        builder.assert_bool(local.is_mne);
        builder.assert_bool(local.is_meq);
        builder.assert_bool(local.is_wsbh);
        let is_real = local.is_mne + local.is_meq + local.is_wsbh;
        builder.assert_bool(is_real.clone());

        // `imm_c` is `is_wsbh` itself (see `MovCondCols::op_c`'s doc comment) -- already forced
        // to 0 outside real rows by `is_wsbh`'s own booleanity + the `is_real` sum above, so
        // `is_real - imm_c` below stays an affine lookup multiplicity landing on 0 or 1.
        let imm_c: AB::Expr = local.is_wsbh.into();

        // If `op_c` is an immediate, assert its value is copied into `op_c_access.prev_value` (so
        // `op_c_access.prev_value` below returns it either way).
        builder
            .when(is_real.clone() * imm_c.clone())
            .assert_word_eq(local.op_c_access.prev_value, local.op_c);

        let op_b_val: Word<AB::Expr> = local.op_b_access.prev_value.map(Into::into);
        let op_c_val: Word<AB::Expr> = local.op_c_access.prev_value.map(Into::into);
        let prev_a_val: Word<AB::Expr> = local.op_a_access.prev_value.map(Into::into);
        let written_value: Word<AB::Expr> = local.op_a_access.value.map(Into::into);

        IsZeroWordOperation::<AB::F>::eval(builder, op_c_val, local.c_eq_0, is_real.clone());

        // Constraints for condition move result:
        // op_a = op_b, when condition is true.
        // Otherwise, op_a remains unchanged (its previous value, read via `op_a_access`).
        builder
            .when(local.is_meq)
            .when(local.c_eq_0.result)
            .assert_word_eq(written_value.clone(), op_b_val.clone());
        builder
            .when(local.is_meq)
            .when_not(local.c_eq_0.result)
            .assert_word_eq(written_value.clone(), prev_a_val.clone());
        builder
            .when(local.is_mne)
            .when_not(local.c_eq_0.result)
            .assert_word_eq(written_value.clone(), op_b_val.clone());
        builder
            .when(local.is_mne)
            .when(local.c_eq_0.result)
            .assert_word_eq(written_value.clone(), prev_a_val);

        self.eval_wsbh(builder, &written_value, &op_b_val, local.is_wsbh);

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        let opcode = local.is_meq * Opcode::MEQ.as_field::<AB::F>()
            + local.is_mne * Opcode::MNE.as_field::<AB::F>()
            + local.is_wsbh * Opcode::WSBH.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: local.op_a.into(),
            op_b: Word::extend_var::<AB>(local.op_b),
            op_c: local.op_c.map(Into::into),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: imm_c.clone(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        // Register positions must be read/written in the order C, B, A (see
        // `MemoryAccessPosition`'s doc comment); `op_c`'s access is skipped (zero multiplicity)
        // when it's an immediate -- `is_real - imm_c` (not `is_real * (1 - imm_c)`) to keep this
        // an affine lookup multiplicity.
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
            local.op_c[0].into(),
            &local.op_c_access,
            is_real.clone() - imm_c,
        );
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            local.op_b.into(),
            &local.op_b_access,
            is_real.clone(),
        );
        builder.eval_register_access_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            local.op_a.into(),
            &local.op_a_access,
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
            is_real,
        );
    }
}

impl MovCondChip {
    pub(crate) fn eval_wsbh<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        written_value: &Word<AB::Expr>,
        op_b_val: &Word<AB::Expr>,
        is_wsbh: AB::Var,
    ) {
        builder.when(is_wsbh).assert_eq(written_value[0].clone(), op_b_val[1].clone());
        builder.when(is_wsbh).assert_eq(written_value[1].clone(), op_b_val[0].clone());
        builder.when(is_wsbh).assert_eq(written_value[2].clone(), op_b_val[3].clone());
        builder.when(is_wsbh).assert_eq(written_value[3].clone(), op_b_val[2].clone());
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
