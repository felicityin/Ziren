use std::borrow::Borrow;

use p3_air::AirBuilder;
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use slop_air::{Air, AirBuilderWithPublicValues};
use zkm_core_executor::Opcode;
use zkm_hypercube::{
    air::{PublicValues, ZKM_PROOF_NUM_PV_ELTS},
    word::Word,
};

use crate::air::{WordAirBuilder, ZKMCoreAirBuilder};
use crate::adapter::{clk_high_expr, clk_low_expr, eval_cpu_state, eval_register_reader, eval_state_chain};

use crate::operations::KoalaBearWordRangeChecker;

use super::{JumpChip, JumpColumns};

impl<AB> Air<AB> for JumpChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &JumpColumns<AB::Var> = (*local).borrow();

        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        // SAFETY: All selectors `is_jump`, `is_jumpi`, `is_jumpdirect`  are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real = is_jump + is_jumpi + is_jumpdirect` is boolean.
        // Therefore, the `opcode` matches the corresponding opcode.
        builder.assert_bool(local.is_jump);
        builder.assert_bool(local.is_jumpi);
        builder.assert_bool(local.is_jumpdirect);
        let is_real = local.is_jump + local.is_jumpi + local.is_jumpdirect;
        builder.assert_bool(is_real.clone());

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk = clk_low_expr::<AB>(&local.state);

        builder.send_program(local.pc, local.instruction, is_real.clone());

        // Jump instructions always write the return address to `op_a` (`op_a_immutable = 0`);
        // when the target register is $0, `eval_register_reader`'s `op_a_0` handling correctly
        // forces the actual write to zero regardless of what `op_a_value` claims.
        eval_register_reader(
            builder,
            &local.reader,
            local.state.clk_high,
            clk.clone(),
            &local.instruction,
            local.op_a_value.map(Into::into),
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            AB::Expr::zero(),
            AB::Expr::zero(),
            is_real.clone(),
        );

        eval_cpu_state(
            builder,
            &local.state,
            public_values.execution_shard,
            clk.clone(),
            is_real.clone(),
        );

        // Unlike every other migrated chip, `outgoing_next_next_pc` here is NOT `next_pc + 4` --
        // it's the jump-resolved value this chip's own logic below already fully derives and
        // constrains (`op_b_value` for J/JR, the ADD dependency for JAL), fed straight into the
        // state chain instead of being validated against a value pulled in from `CpuChip`.
        eval_state_chain(
            builder,
            clk_high_expr::<AB>(&local.state),
            clk,
            local.pc.into(),
            local.next_pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            local.next_next_pc.reduce::<AB>(),
            AB::Expr::from_canonical_u32(5),
            is_real.clone(),
        );

        builder
            .when(is_real.clone())
            .assert_word_eq(local.reader.op_b_val(), local.op_b_value.map(Into::into));
        builder
            .when(is_real.clone())
            .assert_word_eq(local.reader.op_c_val(), local.op_c_value.map(Into::into));

        // Bind each opcode flag to the row's actual fetched opcode, so a real-instruction row
        // can't claim the wrong jump variant while still passing the program lookup.
        builder
            .when(local.is_jump)
            .assert_eq(local.instruction.opcode, Opcode::Jump.as_field::<AB::F>());
        builder
            .when(local.is_jumpi)
            .assert_eq(local.instruction.opcode, Opcode::Jumpi.as_field::<AB::F>());
        builder
            .when(local.is_jumpdirect)
            .assert_eq(local.instruction.opcode, Opcode::JumpDirect.as_field::<AB::F>());

        // Verify that the local.next_pc + 4 is op_a_value for all jump instructions.
        builder.when(is_real.clone()).assert_eq(
            local.op_a_value.reduce::<AB>(),
            local.next_pc.reduce::<AB>() + AB::F::from_canonical_u32(4),
        );

        // Range check op_a, next_pc, and next_next_pc.
        // SAFETY: `is_real` is already checked to be boolean.
        // `op_a_value` is checked to be a valid word, as it matches the one in the CpuChip.
        // In the CpuChip's `RegisterReader::eval`, it's checked that this is valid word saved in op_a when `op_a_0 = 0`
        // Combined with the `op_a_value = next_pc + 4` check above, this fully constrains `op_a_value`.
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.op_a_value,
            local.op_a_range_checker,
            is_real.clone(),
        );
        // SAFETY: `is_real` is already checked to be boolean.
        // `local.next_pc`, `local.next_next_pc` are checked to a valid word when relevant.
        // This is due to the ADD ALU table checking all inputs and outputs are valid words.
        // This is done when the `AddOperation` is invoked in the ADD ALU table.
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_pc,
            local.next_pc_range_checker,
            is_real.clone(),
        );
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.next_next_pc,
            local.next_next_pc_range_checker,
            is_real.clone(),
        );

        // We now constrain `next_next_pc` for J/JR/JALR.
        builder
            .when(local.is_jump + local.is_jumpi)
            .assert_word_eq(local.next_next_pc, local.op_b_value);

        // Verify that the next_next_pc is calculated correctly for BAL instructions.
        // SAFETY: `is_jumpdirect` is boolean, and zero for padding rows.
        builder.send_alu(
            AB::Expr::from_canonical_u32(Opcode::ADD as u32),
            local.next_next_pc,
            local.next_pc,
            local.op_b_value,
            local.is_jumpdirect,
        );
    }
}
