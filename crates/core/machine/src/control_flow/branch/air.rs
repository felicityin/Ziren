use std::borrow::Borrow;

use p3_air::AirBuilder;
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use slop_air::{Air, AirBuilderWithPublicValues};
use zkm_core_executor::{events::MemoryAccessPosition, ByteOpcode, Opcode};
use zkm_hypercube::{
    air::BaseAirBuilder,
    word::Word,
};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, InstructionCols},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    operations::{AddOperation, IsEqualWordOperation, KoalaBearWordRangeChecker},
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

            // When we are branching, assert that local.next_next_pc <==> local.next_pc + c,
            // computed locally by `add_operation` (no cross-chip lookup). `add_operation` is
            // populated with `next_pc + op_c` unconditionally (see `trace.rs`), but only
            // meaningful here when `is_branching` -- the not-branching case is handled by the
            // separate `next_pc + 4` assertion below instead.
            AddOperation::<AB::F>::eval(
                builder,
                local.next_pc,
                local.reader.op_c,
                local.add_operation,
                local.is_branching.into(),
            );
            builder
                .when(local.is_branching)
                .assert_word_eq(local.next_next_pc, local.add_operation.value);

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

        // `a_eq_b`/`msb_a`, computed locally (no cross-chip lookup into `LtChip`): `a_eq_b`
        // covers BEQ/BNE's real two-register comparison and doubles as `op_a == 0` for
        // BLTZ/BGEZ/BLEZ/BGTZ (whose `op_b` is always the hardcoded-zero immediate, so
        // `reader.op_b_val()` is the zero word there -- see `reads_op_b_as_register` above).
        IsEqualWordOperation::<AB::F>::eval(
            builder,
            local.reader.op_a_val().map(Into::into),
            local.reader.op_b_val().map(Into::into),
            local.a_eq_b,
            is_real.clone(),
        );
        let a_eq_b: AB::Expr = local.a_eq_b.is_diff_zero.result.into();

        // `msb_a` is `op_a`'s sign bit -- the only other primitive BLTZ/BGEZ/BLEZ/BGTZ need
        // (they never compare `op_a` against an arbitrary second operand, only against zero).
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.msb_a,
            local.reader.op_a_val()[3],
            AB::Expr::zero(),
            is_real.clone(),
        );

        // Evaluate branching value constraints. `msb_a==1` and `a_eq_b==1` are mutually
        // exclusive (a negative word is never zero), so every sum below is a safe disjoint OR,
        // not just an upper bound.
        {
            // BEQ branches iff a == b.
            builder.when(local.is_beq).assert_eq(local.is_branching, a_eq_b.clone());

            // BNE branches iff a != b.
            builder
                .when(local.is_bne)
                .assert_eq(local.is_branching, AB::Expr::one() - a_eq_b.clone());

            // BLTZ branches iff a < 0, i.e. msb_a.
            builder.when(local.is_bltz).assert_eq(local.is_branching, local.msb_a.into());

            // BGEZ branches iff a >= 0, i.e. !msb_a.
            builder
                .when(local.is_bgez)
                .assert_eq(local.is_branching, AB::Expr::one() - local.msb_a.into());

            // BLEZ branches iff a < 0 || a == 0.
            builder
                .when(local.is_blez)
                .assert_eq(local.is_branching, local.msb_a.into() + a_eq_b.clone());

            // BGTZ branches iff !(a < 0 || a == 0).
            builder
                .when(local.is_bgtz)
                .assert_eq(local.is_branching, AB::Expr::one() - local.msb_a.into() - a_eq_b);
        }
    }
}
