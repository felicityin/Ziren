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
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent, MemoryAccessPosition},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{
        clk_low_expr, eval_cpu_state, eval_i_type_immutable_reader, eval_state_chain, CpuState,
        InstructionCols, ITypeImmutableReader,
    },
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, MemoryReadWriteCols},
    operations::UnalignedWordAddressOperation,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `StoreHalfChip`.
pub const NUM_STORE_HALF_COLS: usize = size_of::<StoreHalfCols<u8>>();

/// A chip that implements the halfword-store opcode SH.
///
/// See `StoreByteChip`'s doc comment for the general rationale. Unlike `StoreByteChip`, only
/// offsets `0`/`2` are valid (the accessed halfword must be 2-byte aligned), so the offset flags
/// `ls_bits_is_one`/`ls_bits_is_three` are constrained to `0`.
#[derive(Default)]
pub struct StoreHalfChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct StoreHalfCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// Register operand access for `a`/`b`/`c`.
    pub adapter: ITypeImmutableReader<T>,

    /// The computed, validated (possibly unaligned) memory address.
    pub word_address: UnalignedWordAddressOperation<T>,

    /// The memory word being written (partially, via a halfword mask).
    pub memory_access: MemoryReadWriteCols<T>,

    /// Whether this row is a real, retired SH instruction (as opposed to padding).
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for StoreHalfChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "StoreHalf".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.store_half_events.len(),
            None,
            <StoreHalfChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.store_half_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <StoreHalfChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_STORE_HALF_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_STORE_HALF_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_STORE_HALF_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut StoreHalfCols<F> = row.borrow_mut();

                    if idx < input.store_half_events.len() {
                        let event = &input.store_half_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_STORE_HALF_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.store_half_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl StoreHalfChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut StoreHalfCols<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
        program: &Program,
    ) {
        cols.state.populate(blu, event.clk);
        cols.is_real = F::ONE;

        let instruction = program.fetch(event.pc);
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        cols.adapter.populate(
            blu,
            instruction.op_a,
            event.a_record,
            instruction.op_b,
            event.b_record,
            instruction.op_c,
        );

        cols.word_address.populate(blu, event.b, event.c);

        cols.memory_access.populate(event.mem_access, blu);
    }
}

impl<F> BaseAir<F> for StoreHalfChip {
    fn width(&self) -> usize {
        NUM_STORE_HALF_COLS
    }
}

impl<AB> Air<AB> for StoreHalfChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &StoreHalfCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);
        let is_real = local.is_real;

        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SH.as_field::<AB::F>().into(),
            op_a: local.adapter.op_a.into(),
            op_b: Word::extend_var::<AB>(local.adapter.op_b),
            op_c: local.adapter.op_c.map(Into::into),
            op_a_0: local.adapter.op_a_0.into(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.into());

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        let addr_aligned = UnalignedWordAddressOperation::<AB::F>::eval(
            builder,
            local.adapter.op_b_val(),
            local.adapter.op_c,
            local.word_address,
            is_real.into(),
        );

        // A halfword access must be 2-byte aligned: offset 1 and 3 are invalid.
        builder
            .when(is_real)
            .assert_zero(local.word_address.ls_bits_is_one + local.word_address.ls_bits_is_three);

        builder.eval_memory_access(
            clk_high.clone(),
            clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::Memory as u32),
            addr_aligned.into(),
            &local.memory_access,
            is_real,
        );

        // The stored halfword overwrites exactly the halfword selected by the offset mux; the
        // other half is unchanged.
        let offset_is_zero = AB::Expr::one()
            - local.word_address.ls_bits_is_one.into()
            - local.word_address.ls_bits_is_two.into()
            - local.word_address.ls_bits_is_three.into();
        let a_val = local.adapter.op_a_val();
        let mem_val = *local.memory_access.value();
        let prev_mem_val = *local.memory_access.prev_value();
        let one = AB::Expr::one();
        let a_is_lower_half = offset_is_zero;
        let a_is_upper_half = local.word_address.ls_bits_is_two;
        let sh_expected_stored_value = Word([
            a_val[0].into() * a_is_lower_half.clone()
                + (one.clone() - a_is_lower_half.clone()) * prev_mem_val[0].into(),
            a_val[1].into() * a_is_lower_half.clone()
                + (one.clone() - a_is_lower_half) * prev_mem_val[1].into(),
            a_val[0].into() * a_is_upper_half
                + (one.clone() - a_is_upper_half.into()) * prev_mem_val[2].into(),
            a_val[1].into() * a_is_upper_half
                + (one - a_is_upper_half.into()) * prev_mem_val[3].into(),
        ]);
        builder.when(is_real).assert_word_eq(mem_val.map(Into::into), sh_expected_stored_value);

        eval_i_type_immutable_reader(
            builder,
            &local.adapter,
            clk_high.clone(),
            clk_low.clone(),
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
    use zkm_core_executor::{
        events::{MemInstrEvent, MemoryWriteRecord},
        ExecutionRecord, Instruction, Opcode, Program,
    };
    use zkm_hypercube::air::MachineAir;

    use super::StoreHalfChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SH,
                op_a: 5,
                op_b: 8,
                op_c: 4,
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
        shard.store_half_events = vec![MemInstrEvent {
            clk: 0,
            pc: 0,
            next_pc: 4,
            opcode: Opcode::SH,
            a: 42,
            b: 100,
            c: 4,
            mem_access: MemoryWriteRecord::new(0xDEAD_002A, 5, 0xDEAD_BE42, 0).into(),
            prev_a_val: 0,
            a_record: None,
            b_record: None,
            c_record: None,
        }];
        let chip = StoreHalfChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
