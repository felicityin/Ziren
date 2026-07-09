pub mod register;

use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use zkm_core_executor::ByteOpcode;
use zkm_hypercube::{
    air::{BaseAirBuilder, PublicValues, ZKMAirBuilder, ZKM_PROOF_NUM_PV_ELTS},
    word::Word,
};

use crate::{
    air::{MemoryAirBuilder, ZKMCoreAirBuilder},
    cpu::{
        columns::{CpuCols, NUM_CPU_COLS},
        CpuChip,
    },
};

impl<AB> Air<AB> for CpuChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &CpuCols<AB::Var> = (*local).borrow();

        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        let clk =
            AB::Expr::from_canonical_u32(1u32 << 16) * local.clk_8bit_limb + local.clk_16bit_limb;

        // Program constraints.
        builder.send_program(local.pc, local.instruction, local.is_real);

        // Register constraints.
        self.eval_registers::<AB>(builder, local, clk.clone());

        // Assert the shard and clk to send.  Only the memory and syscall instructions need the
        // actual shard and clk values for memory access evals.
        // SAFETY: The usage of `builder.if_else` requires `is_memory + is_syscall` to be boolean.
        // The correctness of `is_memory` and `is_syscall` will be checked in the opcode specific chips.
        // In these correct cases, `is_memory + is_syscall` will be always boolean.
        let expected_shard_to_send =
            builder.if_else(local.is_check_memory, local.shard, AB::Expr::zero());
        let expected_clk_to_send =
            builder.if_else(local.is_check_memory, clk.clone(), AB::Expr::zero());
        builder.when(local.is_real).assert_eq(local.shard_to_send, expected_shard_to_send);
        builder.when(local.is_real).assert_eq(local.clk_to_send, expected_clk_to_send);

        builder.send_instruction(
            local.shard_to_send,
            local.clk_to_send,
            local.pc,
            local.next_pc,
            local.next_next_pc,
            local.num_extra_cycles,
            local.instruction.opcode,
            local.op_a_value,
            local.op_b_val(),
            local.op_c_val(),
            local.hi_or_prev_a,
            local.op_a_immutable,
            local.is_rw_a,
            local.is_check_memory,
            local.is_halt,
            local.is_sequential,
            local.is_real,
        );

        // Check that the shard and clk are well-formed.
        self.eval_shard_clk(builder, local, clk.clone(), public_values);

        // Chain this row's state against its predecessor and successor instructions.
        self.eval_state_chain(builder, local, clk);

        // Check control flag consistency.
        self.eval_control_flags(builder, local);

        let not_real = AB::Expr::one() - local.is_real;
        builder.when(not_real.clone()).assert_zero(AB::Expr::one() - local.instruction.imm_b);
        builder.when(not_real.clone()).assert_zero(AB::Expr::one() - local.instruction.imm_c);
        builder.when(not_real.clone()).assert_zero(AB::Expr::one() - local.is_rw_a);
        builder.when(not_real.clone()).assert_zero(local.is_check_memory);
        builder.when(not_real.clone()).assert_zero(local.is_halt);
        builder.when(not_real.clone()).assert_zero(local.is_sequential);
        builder.when(not_real.clone()).assert_zero(local.next_pc);
        builder.when(not_real).assert_zero(local.next_next_pc);
    }
}

impl CpuChip {
    /// Constraints for control flags carried in the CPU row.
    pub(crate) fn eval_control_flags<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
    ) {
        builder.assert_bool(local.is_real);
        builder.when(local.is_real).assert_bool(local.is_rw_a);
        builder.when(local.is_real).assert_bool(local.is_check_memory);
        builder.when(local.is_real).assert_bool(local.is_halt);
        builder.when(local.is_real).assert_bool(local.is_sequential);

        // Halting instructions are not sequential.
        builder.when(local.is_real).assert_zero(local.is_halt * local.is_sequential);
    }

    /// Constraints related to the shard and clk.
    ///
    /// This method checks that `local.shard` matches the shard's public value (the only other
    /// per-row use of `local.shard` is [`Self::eval_registers`]'s memory-consistency check, which
    /// needs the true shard number to order accesses correctly), and range checks that the shard
    /// value is within 16 bits and the clk value is within 24 bits. Those range checks are needed
    /// for the memory access timestamp check, which assumes those values are within 2^24. See
    /// [`MemoryAirBuilder::verify_mem_access_ts`].
    pub(crate) fn eval_shard_clk<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
        public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar>,
    ) {
        builder.when(local.is_real).assert_eq(public_values.execution_shard, local.shard);

        // Verify that the shard value is within 16 bits.
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            local.shard,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );

        // Range check that the clk is within 24 bits using it's limb values.
        builder.eval_range_check_24bits(
            clk,
            local.clk_16bit_limb,
            local.clk_8bit_limb,
            local.is_real,
        );
    }

    /// Chains this row's own `(clk, pc, next_pc)` state to whichever instruction precedes and
    /// follows it, replacing the old row-adjacency chaining with a value-matched
    /// `LookupKind::State` interaction (the zerocheck framework's constraint-evaluation contexts
    /// never expose a "next row"). Every real row unconditionally receives its own incoming state
    /// and sends its successor's -- a genuinely first/last row of a shard has an unmatched
    /// receive/send, closed against public values in [`crate::record::eval_public_values`]
    /// (`ExecutionRecord::eval_public_values`) instead of `when_first_row()`/`when_last_row()`,
    /// which this framework's builders don't support.
    ///
    /// `next_pc` is always this row's own `pc + 4` (the delay slot always follows immediately in
    /// program order); `next_next_pc` is the pc that actually runs after the delay slot, and for a
    /// sequential (non-branch/jump) row that's just `next_pc + 4`. A row whose true successor
    /// doesn't match the state it sent (e.g. a branch/jump row ending a shard, whose target has no
    /// single-pc public value to be exported into) simply fails to find a match, so the shard's
    /// `LookupKind::State` local sum won't close -- this is what replaces the old explicit
    /// "shard boundary must be sequential" assertion.
    pub(crate) fn eval_state_chain<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &CpuCols<AB::Var>,
        clk: AB::Expr,
    ) {
        builder.receive_state(clk.clone(), local.pc, local.next_pc, local.is_real);

        // We already assert that `local.clk < 2^24`. `num_extra_cycles` is an entry of a word and
        // therefore less than `2^8`, this means that the sum cannot overflow in a 31 bit field.
        let expected_next_clk = clk + AB::Expr::from_canonical_u32(5) + local.num_extra_cycles;
        builder.send_state(expected_next_clk, local.next_pc, local.next_next_pc, local.is_real);

        // A non-halting row's own delay slot is always at `pc + 4`.
        builder
            .when(local.is_real)
            .when_not(local.is_halt)
            .assert_eq(local.pc + AB::Expr::from_canonical_u32(4), local.next_pc);

        // A sequential row's post-delay-slot pc is just its delay slot's fall-through.
        builder
            .when(local.is_real)
            .when(local.is_sequential)
            .assert_eq(local.next_next_pc, local.next_pc + AB::Expr::from_canonical_u32(4));
    }
}

impl<F> BaseAir<F> for CpuChip {
    fn width(&self) -> usize {
        NUM_CPU_COLS
    }
}
