use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
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
        clk_low_expr, eval_alu_type_reader, eval_cpu_state, eval_state_chain, AluTypeReader,
        CpuState, InstructionCols,
    },
    air::ZKMCoreAirBuilder,
    utils::{next_power_of_two, pad_rows_fixed},
    CoreChipError,
};

/// The number of main trace columns for `BitwiseChip`.
pub const NUM_BITWISE_COLS: usize = size_of::<BitwiseCols<u8>>();

/// A chip that implements bitwise operations for the opcodes XOR, OR, AND, and NOR.
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `bitwise_events`. A real register-form or immediate-form XOR/OR/AND/NOR whose destination is
/// register 0 is routed to `AluX0Chip` instead (see its doc comment), since its result is
/// unobservable and discarding it soundly requires a different (cheaper) register-write scheme
/// than a real result does -- see `AluTypeReader`'s doc comment. Every real row reaching *this*
/// chip therefore has a genuine, non-zero destination register, which is what lets it use the
/// narrow `AluTypeReader` (see its doc comment) instead of the generic
/// `InstructionCols`+`RegisterReader` pair.
#[derive(Default)]
pub struct BitwiseChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct BitwiseCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: AluTypeReader<T>,

    /// The output operand.
    pub a: Word<T>,

    /// If the opcode is NOR.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_nor: T,

    /// If the opcode is XOR.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_xor: T,

    // If the opcode is OR.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_or: T,

    /// If the opcode is AND.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_and: T,
}

impl<F: PrimeField32> MachineAir<F> for BitwiseChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Bitwise".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        BitwiseCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.bitwise_events.len(),
            None,
            <BitwiseChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut rows = input
            .bitwise_events
            .par_iter()
            .map(|event| {
                let mut row = [F::ZERO; NUM_BITWISE_COLS];
                let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(event, cols, &mut blu, &input.program);
                row
            })
            .collect::<Vec<_>>();

        // Pad the trace to a power of two.
        pad_rows_fixed(
            &mut rows,
            || [F::ZERO; NUM_BITWISE_COLS],
            None,
            <BitwiseChip as MachineAir<F>>::name(self).as_str(),
        );

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_BITWISE_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.bitwise_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .bitwise_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_BITWISE_COLS];
                    let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.bitwise_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl BitwiseChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut BitwiseCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.a = Word::from(event.a);

        cols.is_nor = F::from_bool(event.opcode == Opcode::NOR);
        cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
        cols.is_or = F::from_bool(event.opcode == Opcode::OR);
        cols.is_and = F::from_bool(event.opcode == Opcode::AND);

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
            instruction.imm_c,
        );

        let a = event.a.to_le_bytes();
        let b = event.b.to_le_bytes();
        let c = event.c.to_le_bytes();
        for ((b_a, b_b), b_c) in a.into_iter().zip(b).zip(c) {
            let byte_event = ByteLookupEvent {
                opcode: ByteOpcode::from(event.opcode),
                a1: b_a as u16,
                a2: 0,
                b: b_b,
                c: b_c,
            };
            blu.add_byte_lookup_event(byte_event);
        }
    }
}

impl<F> BaseAir<F> for BitwiseChip {
    fn width(&self) -> usize {
        NUM_BITWISE_COLS
    }
}

impl<AB> Air<AB> for BitwiseChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BitwiseCols<AB::Var> = (*local).borrow();

        let is_real = local.is_xor + local.is_or + local.is_and + local.is_nor;
        builder.assert_bool(local.is_xor);
        builder.assert_bool(local.is_or);
        builder.assert_bool(local.is_and);
        builder.assert_bool(local.is_nor);
        builder.assert_bool(is_real.clone());

        // Get the byte-table opcode for the operation.
        let byte_opcode = local.is_xor * ByteOpcode::XOR.as_field::<AB::F>()
            + local.is_or * ByteOpcode::OR.as_field::<AB::F>()
            + local.is_and * ByteOpcode::AND.as_field::<AB::F>()
            + local.is_nor * ByteOpcode::NOR.as_field::<AB::F>();

        for ((a, b), c) in
            local.a.into_iter().zip(local.adapter.op_b_val()).zip(local.adapter.op_c_val())
        {
            builder.send_byte(byte_opcode.clone(), a, b, c, is_real.clone());
        }

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode` is a degree-1
        // linear combination of the one-hot selectors above (never a variable a malicious prover
        // could substitute -- the program lookup against `ProgramChip`'s preprocessed ROM is what
        // makes `op_a`/`op_b`/`op_c` trustworthy, so there's no separate opcode-binding check
        // needed), `op_a_0`/`imm_b` are compile-time constants (this chip only ever sees a
        // non-zero destination and a register `op_b` -- see `AluTypeReader`'s doc comment), and
        // `op_b` is zero-extended from the adapter's register-index column.
        let cpu_opcode = local.is_xor * Opcode::XOR.as_field::<AB::F>()
            + local.is_or * Opcode::OR.as_field::<AB::F>()
            + local.is_and * Opcode::AND.as_field::<AB::F>()
            + local.is_nor * Opcode::NOR.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: cpu_opcode,
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: local.adapter.imm_c.into(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        eval_alu_type_reader(
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
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::BitwiseChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![
                Instruction {
                    opcode: Opcode::XOR,
                    op_a: 5,
                    op_b: 8,
                    op_c: 9,
                    imm_b: false,
                    imm_c: false,
                    raw: None,
                },
                Instruction {
                    opcode: Opcode::OR,
                    op_a: 5,
                    op_b: 8,
                    op_c: 9,
                    imm_b: false,
                    imm_c: false,
                    raw: None,
                },
                Instruction {
                    opcode: Opcode::AND,
                    op_a: 5,
                    op_b: 8,
                    op_c: 9,
                    imm_b: false,
                    imm_c: false,
                    raw: None,
                },
                Instruction {
                    opcode: Opcode::NOR,
                    op_a: 5,
                    op_b: 8,
                    op_c: 9,
                    imm_b: false,
                    imm_c: false,
                    raw: None,
                },
                Instruction {
                    opcode: Opcode::XOR,
                    op_a: 5,
                    op_b: 8,
                    op_c: 19,
                    imm_b: false,
                    imm_c: true,
                    raw: None,
                },
            ],
            pc_start: 0,
            pc_base: 0,
            next_pc: 4,
            image: Default::default(),
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.bitwise_events = vec![
            AluEvent::new(0, Opcode::XOR, 25, 10, 19),
            AluEvent::new(4, Opcode::OR, 27, 10, 19),
            AluEvent::new(8, Opcode::AND, 2, 10, 19),
            AluEvent::new(12, Opcode::NOR, 228, 10, 19),
            AluEvent::new(16, Opcode::XOR, 25, 10, 19),
        ];
        let chip = BitwiseChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
