use std::borrow::Borrow;

use p3_air::AirBuilder;
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use slop_air::{Air, AirBuilderWithPublicValues};
use zkm_core_executor::{events::MemoryAccessPosition, Opcode};
use zkm_hypercube::{
    air::BaseAirBuilder,
    word::Word,
};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, InstructionCols},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    operations::KoalaBearWordRangeChecker,
};

use super::{BranchChip, BranchColumns};

/// Verifies all the branching related columns.
///
/// It does this in few parts:
/// 1. It verifies that the next next pc is correct based on the branching column.  That column is a
///    boolean that indicates whether the branch condition is true.
/// 2. It verifies the correct value of branching based on the helper bool columns (a_eq_b,
///    a_gt_b, a_lt_b).
/// 3. It verifies the correct values of the helper bool columns based on op_a and op_b.
///
impl<AB> Air<AB> for BranchChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &BranchColumns<AB::Var> = (*local).borrow();


        // SAFETY: All selectors `is_beq`, `is_bne`, `is_bltz`, `is_bgez`, `is_blez`, `is_bgtz` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real`, the sum of the six selectors, is boolean.
        // Therefore, the `opcode` matches the corresponding opcode.
        builder.assert_bool(local.is_beq);
        builder.assert_bool(local.is_bne);
        builder.assert_bool(local.is_bltz);
        builder.assert_bool(local.is_bgez);
        builder.assert_bool(local.is_blez);
        builder.assert_bool(local.is_bgtz);
        let is_real = local.is_beq
            + local.is_bne
            + local.is_bltz
            + local.is_bgez
            + local.is_blez
            + local.is_bgtz;
        builder.assert_bool(is_real.clone());

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // BEQ/BNE compare two real registers (`op_b` is `rt`, `imm_b = 0`); BLTZ/BGEZ/BLEZ/BGTZ
        // compare `op_a` against a hardcoded-zero immediate in `op_b`'s slot (`imm_b = 1`, no
        // register access at all for it) -- see `Instruction::decode`'s BGEZ/BLTZ/BLEZ/BGTZ arms.
        // `op_a`/`op_c` don't have this split: `op_a` is always a real register for all six
        // opcodes, `op_c` is always the encoded immediate offset.
        let reads_op_b_as_register = local.is_beq + local.is_bne;

        // The instruction word is reconstructed here rather than stored: `op_a`/`op_a_0`/`op_b`/
        // `op_c` come from the adapter, and `opcode` is a sum over the six mutually-exclusive
        // selectors (so, unlike a hardcoded opcode, there is nothing left to separately bind a
        // selector to -- the correspondence holds by construction).
        let opcode = local.is_beq.into() * AB::Expr::from_canonical_u32(Opcode::BEQ as u32)
            + local.is_bne.into() * AB::Expr::from_canonical_u32(Opcode::BNE as u32)
            + local.is_bltz.into() * AB::Expr::from_canonical_u32(Opcode::BLTZ as u32)
            + local.is_bgez.into() * AB::Expr::from_canonical_u32(Opcode::BGEZ as u32)
            + local.is_blez.into() * AB::Expr::from_canonical_u32(Opcode::BLEZ as u32)
            + local.is_bgtz.into() * AB::Expr::from_canonical_u32(Opcode::BGTZ as u32);
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: local.reader.op_a.into(),
            op_b: Word::extend_var::<AB>(local.reader.op_b),
            op_c: local.reader.op_c.map(Into::into),
            op_a_0: local.reader.op_a_0.into(),
            imm_b: AB::Expr::one() - reads_op_b_as_register.clone(),
            imm_c: AB::Expr::one(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        // `op_a` is always a real register access. `op_b` only is for BEQ/BNE -- for the other
        // four opcodes it's a hardcoded-zero immediate (see `reads_op_b_as_register` above), so
        // its register-access lookup must be gated accordingly, and its witnessed value forced to
        // zero on the remaining rows (otherwise a dishonest prover could witness any value there
        // and use it, unconstrained, in the SLT comparisons below).
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            local.reader.op_b,
            &local.reader.op_b_access,
            reads_op_b_as_register.clone(),
        );
        builder
            .when(is_real.clone() - reads_op_b_as_register)
            .assert_word_zero(local.reader.op_b_val());
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            local.reader.op_a,
            &local.reader.op_a_access,
            is_real.clone(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.clone());

        // `op_b`/`op_c` are read directly from `local.reader` below wherever their value is
        // needed (the SLT sends, the ADD lookup) -- there is no separate `op_b_value`/
        // `op_c_value` column to keep in sync with the reader.

        // Unlike every other migrated chip, `outgoing_next_next_pc` here is NOT `next_pc + 4` --
        // it's the branch-resolved value this chip's own logic below already fully derives and
        // constrains (via the ADD/`is_branching` lookup), fed straight into the state chain
        // instead of being validated against a value pulled in from `CpuChip`.
        eval_state_chain(
            builder,
            clk_high,
            clk_low,
            local.pc.into(),
            local.next_pc.reduce::<AB>(),
            local.next_pc.reduce::<AB>(),
            local.next_next_pc.reduce::<AB>(),
            AB::Expr::from_canonical_u32(5),
            is_real.clone(),
        );

        // Evaluate program counter constraints.
        {
            // Range check local.next_pc and local.next_next_pc.
            // SAFETY: `is_real` is already checked to be boolean.
            // The `KoalaBearWordRangeChecker` assumes that the value is checked to be a valid word.
            // This is done when the word form is relevant, i.e. when `pc` and `next_pc` are sent to the ADD ALU table.
            // The ADD ALU table checks the inputs are valid words, when it invokes `AddOperation`.
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

            // When we are branching, assert that local.next_next_pc <==> local.next_pc + c.
            // `next_next_pc` is sent directly as the ADD lookup's result -- no intermediate
            // `target_pc` column is needed, since the only thing it was ever compared against
            // was `next_next_pc` itself.
            builder.send_alu(
                Opcode::ADD.as_field::<AB::F>(),
                local.next_next_pc,
                local.next_pc,
                local.reader.op_c,
                local.is_branching,
            );

            // When we are not branching, assert that local.next_pc + 4 <==> next.next_next_pc.
            builder.when(is_real.clone()).when_not(local.is_branching).assert_eq(
                local.next_pc.reduce::<AB>() + AB::Expr::from_canonical_u32(4),
                local.next_next_pc.reduce::<AB>(),
            );

            // check local.next_pc/next_next_pc to be valid word when we are not branching.
            // they are checked as valid value by the ADD ALU table when we are branching.
            builder.slice_range_check_u8(&local.next_pc.0, is_real.clone() - local.is_branching);
            builder
                .slice_range_check_u8(&local.next_next_pc.0, is_real.clone() - local.is_branching);

            // To prevent the ALU send above to be non-zero when the row is a padding row.
            builder.when_not(is_real.clone()).assert_zero(local.is_branching);

            // Assert the branching or not branching when the instruction is a
            builder.when(is_real.clone()).assert_bool(local.is_branching);
        }

        // Evaluate branching value constraints.
        {
            // When the opcode is BEQ and we are branching, assert that a_gt_b + a_lt_b is false.
            builder
                .when(local.is_beq * local.is_branching)
                .assert_zero(local.a_gt_b + local.a_lt_b);

            // When the opcode is BEQ and we are not branching, assert that either a_gt_b or a_lt_b
            // is true.
            builder
                .when(local.is_beq)
                .when_not(local.is_branching)
                .assert_one(local.a_gt_b + local.a_lt_b);

            // When the opcode is BNE and we are branching, assert that either a_gt_b or a_lt_b is
            // true.
            builder.when(local.is_bne * local.is_branching).assert_one(local.a_gt_b + local.a_lt_b);

            // When the opcode is BNE and we are not branching, assert that a_gt_b + a_lt_b is false.
            builder
                .when(local.is_bne)
                .when_not(local.is_branching)
                .assert_zero(local.a_gt_b + local.a_lt_b);

            // When the opcode is BLTZ and we are branching, assert that a_lt_b is true.
            builder.when(local.is_bltz * local.is_branching).assert_one(local.a_lt_b);

            // When the opcode is BLTZ and we are not branching, assert a_lt_b is false.
            builder.when(local.is_bltz).when_not(local.is_branching).assert_zero(local.a_lt_b);

            // When the opcode is BLEZ and we are branching, assert that either a_gt_b is false
            builder.when(local.is_blez * local.is_branching).assert_zero(local.a_gt_b);

            // When the opcode is BLEZ and we are not branching, assert that a_gt_b is true.
            builder.when(local.is_blez).when_not(local.is_branching).assert_one(local.a_gt_b);

            // When the opcode is BGTZ and we are branching, assert that a_gt_b is true.
            builder.when(local.is_bgtz * local.is_branching).assert_one(local.a_gt_b);

            // When the opcode is BGTZ and we are not branching, assert that a_gt_b is false.
            builder.when(local.is_bgtz).when_not(local.is_branching).assert_zero(local.a_gt_b);

            // When the opcode is BGEZ and we are branching, assert that a_lt_b is false.
            builder.when(local.is_bgez * local.is_branching).assert_zero(local.a_lt_b);

            // When the opcode is BGEZ and we are not branching, assert that a_lt_b is true.
            builder.when(local.is_bgez).when_not(local.is_branching).assert_one(local.a_lt_b);
        }

        // Calculate a_lt_b <==> a < b (using appropriate signedness).
        // SAFETY: `use_signed_comparison` is boolean, since at most one selector is turned on.
        builder.send_alu(
            Opcode::SLT.as_field::<AB::F>(),
            Word::extend_var::<AB>(local.a_lt_b),
            local.reader.op_a_val(),
            local.reader.op_b_val(),
            is_real.clone(),
        );

        // Calculate a_gt_b <==> a > b (using appropriate signedness).
        builder.send_alu(
            Opcode::SLT.as_field::<AB::F>(),
            Word::extend_var::<AB>(local.a_gt_b),
            local.reader.op_b_val(),
            local.reader.op_a_val(),
            is_real.clone(),
        );
    }
}
