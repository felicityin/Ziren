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
    events::{AluEvent, ByteLookupEvent, ByteRecord, MemoryRecordEnum},
    ByteOpcode, ExecutionRecord, Opcode, Program, UNUSED_PC,
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
    utils::{next_power_of_two, pad_rows_fixed},
    CoreChipError,
};

/// The number of main trace columns for `BitwiseChip`.
pub const NUM_BITWISE_COLS: usize = size_of::<BitwiseCols<u8>>();

/// A chip that implements bitwise operations for the opcodes XOR, OR, AND, and NOR.
///
/// As with `AddChip`, not every row is a real retired instruction -- though as of this
/// writing no other chip emits a synthetic dependency row into `bitwise_events`, the
/// `is_real_X` split is kept for consistency with the other opcode-family chips and to stay
/// sound if a future dependency producer is added.
#[derive(Default)]
pub struct BitwiseChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct BitwiseCols<T: Copy> {
    /// The current shard and clk. Only meaningful when this row is a real instruction.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The raw fetched instruction. Only meaningful when this row is a real instruction.
    pub instruction: InstructionCols<T>,

    /// Register operand access for `a`/`b`/`c`. Only meaningful when this row is a real
    /// instruction.
    pub reader: RegisterReader<T>,

    /// Whether this row is a real, retired NOR instruction.
    pub is_real_nor: T,

    /// Whether this row is a real, retired XOR instruction.
    pub is_real_xor: T,

    /// Whether this row is a real, retired OR instruction.
    pub is_real_or: T,

    /// Whether this row is a real, retired AND instruction.
    pub is_real_and: T,

    /// The output operand.
    pub a: Word<T>,

    /// The first input operand.
    pub b: Word<T>,

    /// The second input operand.
    pub c: Word<T>,

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
            || {
                let mut row = [F::ZERO; NUM_BITWISE_COLS];
                let cols: &mut BitwiseCols<F> = row.as_mut_slice().borrow_mut();
                // Padding row: force the register reader's b/c memory-access multiplicities to
                // zero (see cpuchip-migration-register-reader-gotchas memory).
                cols.instruction.imm_b = F::ONE;
                cols.instruction.imm_c = F::ONE;
                row
            },
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
        cols.b = Word::from(event.b);
        cols.c = Word::from(event.c);

        cols.is_nor = F::from_bool(event.opcode == Opcode::NOR);
        cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
        cols.is_or = F::from_bool(event.opcode == Opcode::OR);
        cols.is_and = F::from_bool(event.opcode == Opcode::AND);

        // Default: not a real instruction, so the register reader's b/c memory accesses have
        // zero multiplicity unless overwritten by a real fetched instruction's actual immediate
        // flags just below.
        cols.instruction.imm_b = F::ONE;
        cols.instruction.imm_c = F::ONE;

        let is_real_instruction = event.pc != UNUSED_PC;
        if is_real_instruction {
            cols.is_real_nor = cols.is_nor;
            cols.is_real_xor = cols.is_xor;
            cols.is_real_or = cols.is_or;
            cols.is_real_and = cols.is_and;

            cols.state.populate(blu, event.shard, event.clk);

            let instruction = program.fetch(event.pc);
            cols.instruction.populate(&instruction);

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

        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        // Get the opcode for the operation.
        let opcode = local.is_xor * ByteOpcode::XOR.as_field::<AB::F>()
            + local.is_or * ByteOpcode::OR.as_field::<AB::F>()
            + local.is_and * ByteOpcode::AND.as_field::<AB::F>()
            + local.is_nor * ByteOpcode::NOR.as_field::<AB::F>();

        // Get a multiplicity of `1` only for a true row. The underlying byte arithmetic must
        // hold for both real-instruction rows and synthetic dependency rows alike, so this stays
        // gated on `is_real` (the full opcode-selector sum), not the real-instruction-only flags.
        let mult = local.is_xor + local.is_or + local.is_and + local.is_nor;
        for ((a, b), c) in local.a.into_iter().zip(local.b).zip(local.c) {
            builder.send_byte(opcode.clone(), a, b, c, mult.clone());
        }

        let is_real = local.is_xor + local.is_or + local.is_and + local.is_nor;
        builder.assert_bool(local.is_xor);
        builder.assert_bool(local.is_or);
        builder.assert_bool(local.is_and);
        builder.assert_bool(local.is_nor);
        builder.assert_bool(is_real.clone());

        builder.assert_bool(local.is_real_xor);
        builder.assert_bool(local.is_real_or);
        builder.assert_bool(local.is_real_and);
        builder.assert_bool(local.is_real_nor);
        builder.when_not(local.is_xor).assert_zero(local.is_real_xor);
        builder.when_not(local.is_or).assert_zero(local.is_real_or);
        builder.when_not(local.is_and).assert_zero(local.is_real_and);
        builder.when_not(local.is_nor).assert_zero(local.is_real_nor);
        let is_real_instruction =
            local.is_real_xor + local.is_real_or + local.is_real_and + local.is_real_nor;

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk = clk_expr::<AB>(&local.state);

        builder.send_program(local.pc, local.instruction, is_real_instruction.clone());

        eval_register_reader(
            builder,
            &local.reader,
            local.state.shard,
            clk.clone(),
            &local.instruction,
            // Gated by `is_real_instruction`: `register.rs`'s `assert_word_eq(op_a_value,
            // reader.op_a_val())` fires unconditionally whenever `op_a_0` is unset, which it is
            // by default on synthetic rows (their `instruction` column is never populated) --
            // an ungated `op_a_value` would then have to equal `reader.op_a_val()` (always zero
            // on synthetic rows) even when the true result is nonzero.
            local.a.map(|x| is_real_instruction.clone() * Into::<AB::Expr>::into(x)),
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
            .assert_word_eq(local.reader.op_b_val(), local.b.map(Into::into));
        builder
            .when(is_real_instruction.clone())
            .assert_word_eq(local.reader.op_c_val(), local.c.map(Into::into));

        // Bind `is_real_X` to the row's actual fetched opcode, so a real-instruction row can't
        // claim the wrong bitwise variant while still passing the program lookup.
        builder
            .when(local.is_real_xor)
            .assert_eq(local.instruction.opcode, Opcode::XOR.as_field::<AB::F>());
        builder
            .when(local.is_real_or)
            .assert_eq(local.instruction.opcode, Opcode::OR.as_field::<AB::F>());
        builder
            .when(local.is_real_and)
            .assert_eq(local.instruction.opcode, Opcode::AND.as_field::<AB::F>());
        builder
            .when(local.is_real_nor)
            .assert_eq(local.instruction.opcode, Opcode::NOR.as_field::<AB::F>());

        // Get the cpu opcode, which corresponds to the opcode being sent in the CPU table. This
        // mux only depends on which `is_X` selector is set (mutually exclusive), so it's
        // identical for real-instruction and synthetic rows -- only the multiplicity below
        // changes to select the synthetic-only rows.
        let cpu_opcode = local.is_xor * Opcode::XOR.as_field::<AB::F>()
            + local.is_or * Opcode::OR.as_field::<AB::F>()
            + local.is_and * Opcode::AND.as_field::<AB::F>()
            + local.is_nor * Opcode::NOR.as_field::<AB::F>();

        // ---- Synthetic dependency path: matches whichever chip generated this internal check via
        // `send_alu` (always at the `UNUSED_PC` sentinel, shard/clk zero). No current producer
        // targets `bitwise_events`, so this multiplicity is always zero today, but the wiring is
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
            local.c,
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::one(),
            is_real - is_real_instruction,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode, UNUSED_PC};
    use zkm_hypercube::air::MachineAir;
    // use zkm_stark::{
    //     air::MachineAir, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig,
    // };
    //
    // use crate::utils::{uni_stark_prove, uni_stark_verify};

    use super::BitwiseChip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.bitwise_events = vec![
            AluEvent::new(UNUSED_PC, Opcode::XOR, 25, 10, 19),
            AluEvent::new(UNUSED_PC, Opcode::OR, 27, 10, 19),
            AluEvent::new(UNUSED_PC, Opcode::AND, 2, 10, 19),
            AluEvent::new(UNUSED_PC, Opcode::NOR, 228, 10, 19),
        ];
        let chip = BitwiseChip::default();
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
        // let mut shard = ExecutionRecord::default();
        // shard.bitwise_events = [
        //     AluEvent::new(0, Opcode::XOR, 25, 10, 19),
        //     AluEvent::new(0, Opcode::OR, 27, 10, 19),
        //     AluEvent::new(0, Opcode::AND, 2, 10, 19),
        //     AluEvent::new(0, Opcode::NOR, 228, 10, 19),
        // ]
        // .repeat(1000);
        // let chip = BitwiseChip::default();
        // let trace: RowMajorMatrix<KoalaBear> =
        //     chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        // let proof =
        //     uni_stark_prove::<KoalaBearPoseidon2, _>(&config, &chip, &mut challenger, trace);
        //
        // let mut challenger = config.challenger();
        // uni_stark_verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
