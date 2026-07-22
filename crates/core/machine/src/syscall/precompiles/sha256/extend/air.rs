use p3_air::{Air, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use zkm_core_executor::ByteOpcode;
use zkm_hypercube::air::{AirLookup, LookupScope, ZKMAirBuilder};
use zkm_hypercube::lookup::LookupKind;

use super::{ShaExtendChip, ShaExtendCols, NUM_SHA_EXTEND_COLS};
use crate::{
    adapter::{clk_low_expr, eval_cpu_state},
    air::{MemoryAirBuilder, WordAirBuilder},
    memory::MemoryCols,
    operations::{
        Add4Operation, FixedRotateRightOperation, FixedShiftRightOperation, XorOperation,
    },
};

use core::borrow::Borrow;

impl<F> BaseAir<F> for ShaExtendChip {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_COLS
    }
}

impl<AB> Air<AB> for ShaExtendChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        // Initialize columns.
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaExtendCols<AB::Var> = (*local).borrow();

        let i_start = AB::F::from_canonical_u32(16);
        let nb_bytes_in_word = AB::F::from_canonical_u32(4);

        builder.assert_bool(local.is_real);

        let clk_high = local.state.clk_high;
        let clk_low = clk_low_expr::<AB>(&local.state);
        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

        // Bound `16 <= i < 64` so the `AddrAddOperation`-style pointer arithmetic below is safe.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::one(),
            local.i - i_start,
            AB::Expr::from_canonical_u32(48),
            local.is_real,
        );

        // Receive this row's own incoming `(clk_high, clk_low, w_ptr, i)`, and send the
        // successor's -- replacing the old row-adjacency chaining. The genuinely first (`i == 16`)
        // and last (`i == 64`) links are closed by `ShaExtendControlChip`, which brackets the
        // syscall.
        builder.receive(
            AirLookup::new(
                vec![clk_high.into(), clk_low.clone(), local.w_ptr.into(), local.i.into()],
                local.is_real.into(),
                LookupKind::ShaExtend,
            ),
            LookupScope::Local,
        );
        builder.send(
            AirLookup::new(
                vec![
                    clk_high.into(),
                    clk_low.clone(),
                    local.w_ptr.into(),
                    local.i.into() + AB::Expr::one(),
                ],
                local.is_real.into(),
                LookupKind::ShaExtend,
            ),
            LookupScope::Local,
        );

        // Read w[i-15].
        builder.eval_memory_access(
            clk_high,
            clk_low.clone() + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(15)) * nb_bytes_in_word,
            &local.w_i_minus_15,
            local.is_real,
        );

        // Read w[i-2].
        builder.eval_memory_access(
            clk_high,
            clk_low.clone() + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(2)) * nb_bytes_in_word,
            &local.w_i_minus_2,
            local.is_real,
        );

        // Read w[i-16].
        builder.eval_memory_access(
            clk_high,
            clk_low.clone() + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(16)) * nb_bytes_in_word,
            &local.w_i_minus_16,
            local.is_real,
        );

        // Read w[i-7].
        builder.eval_memory_access(
            clk_high,
            clk_low.clone() + (local.i - i_start),
            local.w_ptr + (local.i - AB::F::from_canonical_u32(7)) * nb_bytes_in_word,
            &local.w_i_minus_7,
            local.is_real,
        );

        // Compute `s0`.
        // w[i-15] rightrotate 7.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            *local.w_i_minus_15.value(),
            7,
            local.w_i_minus_15_rr_7,
            local.is_real,
        );
        // w[i-15] rightrotate 18.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            *local.w_i_minus_15.value(),
            18,
            local.w_i_minus_15_rr_18,
            local.is_real,
        );
        // w[i-15] rightshift 3.
        FixedShiftRightOperation::<AB::F>::eval(
            builder,
            *local.w_i_minus_15.value(),
            3,
            local.w_i_minus_15_rs_3,
            local.is_real.into(),
        );
        // (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18)
        XorOperation::<AB::F>::eval(
            builder,
            local.w_i_minus_15_rr_7.value,
            local.w_i_minus_15_rr_18.value,
            local.s0_intermediate,
            local.is_real,
        );
        // s0 := (w[i-15] rightrotate 7) xor (w[i-15] rightrotate 18) xor (w[i-15] rightshift 3)
        XorOperation::<AB::F>::eval(
            builder,
            local.s0_intermediate.value,
            local.w_i_minus_15_rs_3.value,
            local.s0,
            local.is_real,
        );

        // Compute `s1`.
        // w[i-2] rightrotate 17.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            *local.w_i_minus_2.value(),
            17,
            local.w_i_minus_2_rr_17,
            local.is_real,
        );
        // w[i-2] rightrotate 19.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            *local.w_i_minus_2.value(),
            19,
            local.w_i_minus_2_rr_19,
            local.is_real,
        );
        // w[i-2] rightshift 10.
        FixedShiftRightOperation::<AB::F>::eval(
            builder,
            *local.w_i_minus_2.value(),
            10,
            local.w_i_minus_2_rs_10,
            local.is_real.into(),
        );
        // (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19)
        XorOperation::<AB::F>::eval(
            builder,
            local.w_i_minus_2_rr_17.value,
            local.w_i_minus_2_rr_19.value,
            local.s1_intermediate,
            local.is_real,
        );
        // s1 := (w[i-2] rightrotate 17) xor (w[i-2] rightrotate 19) xor (w[i-2] rightshift 10)
        XorOperation::<AB::F>::eval(
            builder,
            local.s1_intermediate.value,
            local.w_i_minus_2_rs_10.value,
            local.s1,
            local.is_real,
        );

        // s2 := w[i-16] + s0 + w[i-7] + s1.
        Add4Operation::<AB::F>::eval(
            builder,
            *local.w_i_minus_16.value(),
            local.s0.value,
            *local.w_i_minus_7.value(),
            local.s1.value,
            local.is_real,
            local.s2,
        );

        // Write `s2` to `w[i]`.
        builder.eval_memory_access(
            clk_high,
            clk_low + (local.i - i_start),
            local.w_ptr + local.i * nb_bytes_in_word,
            &local.w_i,
            local.is_real,
        );

        builder.assert_word_eq(*local.w_i.value(), local.s2.value);
    }
}
