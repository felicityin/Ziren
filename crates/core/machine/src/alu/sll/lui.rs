use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord, MemoryAccessPosition},
    ExecutionRecord, Opcode, Program, Register,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::BaseAirBuilder, air::MachineAir, word::Word};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState, InstructionCols},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::RegisterWriteAccessCols,
    utils::{next_power_of_two, pad_rows_fixed},
    CoreChipError,
};

/// The number of main trace columns for `LuiChip`.
pub const NUM_LUI_COLS: usize = size_of::<LuiCols<u8>>();

/// A chip that implements the MIPS load-upper-immediate instruction LUI.
///
/// LUI decodes to `Opcode::SLL` with `imm_b=true` (`op_b` is the instruction's own encoded
/// immediate, `rt = imm << 16`) and `imm_c=true`, `op_c` always the constant `16` -- a distinct
/// shape from register-form/immediate-shift-amount-form SLL/SLLV (`imm_b` always false, `op_b`
/// always a register), which `ShiftLeftChip` handles instead. Since the shift amount is always
/// exactly 16 (byte-aligned), the result is pure byte wiring, not arithmetic: `op_a = Word([0, 0,
/// op_b[0], op_b[1]])` (`op_b`'s upper two bytes are simply the bits shifted out of the 32-bit
/// result, discarded regardless of their value).
///
/// `op_b`/`op_c` are never registers, so unlike `RTypeReader`/`AluTypeReader`'s chips, `op_a==0`
/// isn't routed to `AluX0Chip` (its `op_b`/`op_c` columns assume a register-index shape that
/// doesn't apply here) -- this chip masks it inline instead, the same way `ITypeReader` does.
#[derive(Default)]
pub struct LuiChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct LuiCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The register index of `op_a` (written).
    pub op_a: T,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,
    pub op_a_access: RegisterWriteAccessCols<T>,

    /// The instruction's own encoded immediate (never a register).
    pub op_b: Word<T>,

    /// Whether this row is a real, retired LUI instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for LuiChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Lui".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        LuiCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.lui_events.len(),
            None,
            <LuiChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows = input
            .lui_events
            .par_iter()
            .map(|event| {
                let mut row = [F::ZERO; NUM_LUI_COLS];
                let cols: &mut LuiCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(event, cols, &mut blu, &input.program);
                row
            })
            .collect::<Vec<_>>();

        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_LUI_COLS],
            None,
            <LuiChip as MachineAir<F>>::name(self).as_str(),
        );

        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_LUI_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.lui_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .lui_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_LUI_COLS];
                    let cols: &mut LuiCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.lui_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LuiChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut LuiCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.is_real = F::ONE;

        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);
        cols.op_a = F::from_canonical_u8(instruction.op_a);
        cols.op_a_0 = F::from_bool(instruction.op_a == Register::ZERO as u8);
        cols.op_b = Word::from(event.b);

        if let Some(record) = event.a_record {
            cols.op_a_access.populate(record, blu);
        }
    }
}

impl<F> BaseAir<F> for LuiChip {
    fn width(&self) -> usize {
        NUM_LUI_COLS
    }
}

impl<AB> Air<AB> for LuiChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LuiCols<AB::Var> = (*local).borrow();

        let is_real = local.is_real;
        builder.assert_bool(is_real);
        builder.when(is_real).assert_bool(local.op_a_0);

        // `op_a = op_b << 16`: byte-aligned, so this is pure wiring -- `op_b`'s upper two bytes
        // are simply discarded (the bits shifted out of the 32-bit result), regardless of value.
        let computed_value = Word([
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.op_b[0].into(),
            local.op_b[1].into(),
        ]);

        let written_value: Word<AB::Expr> = local.op_a_access.value.map(Into::into);
        builder.when(is_real).when(local.op_a_0).assert_word_zero(written_value.clone());
        builder
            .when(is_real)
            .when_not(local.op_a_0)
            .assert_word_eq(computed_value, written_value);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode`/`imm_b`/
        // `imm_c` are compile-time constants (LUI always decodes to `Opcode::SLL` with both
        // immediate flags set), and `op_c` is always the constant `16` (the byte-aligned shift
        // amount) -- the program lookup against `ProgramChip`'s preprocessed ROM is what makes
        // `op_a`/`op_b` trustworthy.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SLL.as_field::<AB::F>().into(),
            op_a: local.op_a.into(),
            op_b: local.op_b.map(Into::into),
            op_c: Word([
                AB::Expr::from_canonical_u32(16),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            op_a_0: local.op_a_0.into(),
            imm_b: AB::Expr::one(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.into());

        builder.eval_register_access_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            local.op_a.into(),
            &local.op_a_access,
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
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::LuiChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SLL,
                op_a: 5,
                op_b: 0x1234,
                op_c: 16,
                imm_b: true,
                imm_c: true,
                raw: None,
            }],
            pc_start: 0,
            pc_base: 0,
            next_pc: 4,
            image: Default::default(),
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.lui_events = vec![AluEvent::new(0, Opcode::SLL, 0x1234_0000, 0x1234, 16)];
        let chip = LuiChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
