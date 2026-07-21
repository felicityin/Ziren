use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use zkm_hypercube::{
    air::{AirLookup, LookupScope, ZKMAirBuilder},
    lookup::LookupKind,
    word::Word,
};

use super::{
    columns::{ShaCompressCols, NUM_SHA_COMPRESS_COLS},
    ShaCompressChip, SHA_COMPRESS_K,
};
use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    memory::MemoryCols,
    operations::{
        Add5Operation, AddOperation, AndOperation, FixedRotateRightOperation, NotOperation,
        XorOperation,
    },
};
use zkm_hypercube::air::BaseAirBuilder;

impl<F> BaseAir<F> for ShaCompressChip {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_COLS
    }
}

impl<AB> Air<AB> for ShaCompressChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaCompressCols<AB::Var> = (*local).borrow();

        self.eval_control_flow_flags(builder, local);

        self.eval_memory(builder, local);

        self.eval_compression_ops(builder, local);

        self.eval_finalize_ops(builder, local);
    }
}

impl ShaCompressChip {
    fn eval_control_flow_flags<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        builder.assert_bool(local.is_real);

        let mut computed_index = AB::Expr::zero();

        // Verify that all of the octet columns are bool, and exactly one is true.
        let mut octet_sum = AB::Expr::zero();
        for i in 0..8 {
            builder.assert_bool(local.octet[i]);
            octet_sum = octet_sum.clone() + local.octet[i].into();
            computed_index = computed_index.clone() + local.octet[i].into() * AB::Expr::from_canonical_u32(i as u32);
        }
        builder.assert_one(octet_sum);

        // Verify that all of the octet_num columns are bool, and exactly one is true.
        let mut octet_num_sum = AB::Expr::zero();
        for i in 0..10 {
            builder.assert_bool(local.octet_num[i]);
            octet_num_sum = octet_num_sum.clone() + local.octet_num[i].into();
            computed_index = computed_index.clone() + local.octet_num[i].into() * AB::Expr::from_canonical_u32(8 * i as u32);
        }
        builder.assert_one(octet_num_sum);

        // Check that `local.index` matches the `octet`/`octet_num` one-hot flags.
        builder.assert_eq(local.index, computed_index);

        // Assert that the is_initialize flag is correct.
        builder.assert_eq(local.is_initialize, local.octet_num[0] * local.is_real);

        // Assert that the is_compression flag is correct.
        builder.assert_eq(
            local.is_compression,
            (local.octet_num[1]
                + local.octet_num[2]
                + local.octet_num[3]
                + local.octet_num[4]
                + local.octet_num[5]
                + local.octet_num[6]
                + local.octet_num[7]
                + local.octet_num[8])
                * local.is_real,
        );

        // Assert that the is_finalize flag is correct.
        builder.assert_eq(local.is_finalize, local.octet_num[9] * local.is_real);

        // Chain this row's own `(shard, clk, w_ptr, h_ptr, index, a..h)` state against whichever
        // row (or `ShaCompressControlChip`, which brackets the syscall at index 0 and 80) sent
        // it, and this row's own successor state against whichever row (or the control chip)
        // receives it -- replacing the old row-adjacency chaining of octet/octet_num, the A-H
        // registers, and shard/clk/w_ptr/h_ptr invariance all at once.
        let state = [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h];
        let receive_values = [local.shard.into(), local.clk.into(), local.w_ptr.into(), local.h_ptr.into(), local.index.into()]
            .into_iter()
            .chain(state.iter().flat_map(|word| word.0.iter().map(|&e| e.into())))
            .collect::<Vec<_>>();
        builder.receive(
            AirLookup::new(receive_values, local.is_real.into(), LookupKind::ShaCompress),
            LookupScope::Local,
        );

        // During initialize and finalize, the state passes through unchanged.
        let pass_through_send_values =
            [local.shard.into(), local.clk.into(), local.w_ptr.into(), local.h_ptr.into(), local.index.into() + AB::Expr::one()]
                .into_iter()
                .chain(state.iter().flat_map(|word| word.0.iter().map(|&e| e.into())))
                .collect::<Vec<_>>();
        builder.send(
            AirLookup::new(
                pass_through_send_values,
                local.is_initialize + local.is_finalize,
                LookupKind::ShaCompress,
            ),
            LookupScope::Local,
        );

        // During compression, the registers rotate: h:=g, g:=f, f:=e, e:=d+temp1, d:=c, c:=b,
        // b:=a, a:=temp1+temp2.
        let rotated_state =
            [local.temp1_add_temp2.value, local.a, local.b, local.c, local.d_add_temp1.value, local.e, local.f, local.g];
        let compression_send_values =
            [local.shard.into(), local.clk.into(), local.w_ptr.into(), local.h_ptr.into(), local.index.into() + AB::Expr::one()]
                .into_iter()
                .chain(rotated_state.iter().flat_map(|word| word.0.iter().map(|&e| e.into())))
                .collect::<Vec<_>>();
        builder.send(
            AirLookup::new(compression_send_values, local.is_compression.into(), LookupKind::ShaCompress),
            LookupScope::Local,
        );
    }

    /// Constrains that memory address is correct and that memory is correctly written/read.
    fn eval_memory<AB: ZKMAirBuilder>(&self, builder: &mut AB, local: &ShaCompressCols<AB::Var>) {
        builder.eval_memory_access(
            local.clk_high,
            local.clk + local.is_finalize,
            local.mem_addr,
            &local.mem,
            local.is_initialize + local.is_compression + local.is_finalize,
        );

        // Calculate the current cycle_num.
        let mut cycle_num = AB::Expr::zero();
        for i in 0..10 {
            cycle_num = cycle_num.clone() + local.octet_num[i] * AB::Expr::from_canonical_usize(i);
        }

        // Calculate the current step of the cycle 8.
        let mut cycle_step = AB::Expr::zero();
        for i in 0..8 {
            cycle_step = cycle_step.clone() + local.octet[i] * AB::Expr::from_canonical_usize(i);
        }

        // Verify correct mem address for initialize phase
        builder.when(local.is_initialize).assert_eq(
            local.mem_addr,
            local.h_ptr + cycle_step.clone() * AB::Expr::from_canonical_u32(4),
        );

        // Verify correct mem address for compression phase
        builder.when(local.is_compression).assert_eq(
            local.mem_addr,
            local.w_ptr
                + (((cycle_num - AB::Expr::one()) * AB::Expr::from_canonical_u32(8))
                    + cycle_step.clone())
                    * AB::Expr::from_canonical_u32(4),
        );

        // Verify correct mem address for finalize phase
        builder.when(local.is_finalize).assert_eq(
            local.mem_addr,
            local.h_ptr + cycle_step.clone() * AB::Expr::from_canonical_u32(4),
        );

        // In the initialize phase, verify that local.a, local.b, ... is correctly read from memory
        // and does not change
        let vars = [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h];
        for (i, var) in vars.iter().enumerate() {
            builder
                .when(local.is_initialize)
                .when(local.octet[i])
                .assert_word_eq(*var, *local.mem.prev_value());
            builder
                .when(local.is_initialize)
                .when(local.octet[i])
                .assert_word_eq(*var, *local.mem.value());
        }

        // During compression, verify that memory is read only and does not change.
        builder
            .when(local.is_compression)
            .assert_word_eq(*local.mem.prev_value(), *local.mem.value());

        // In the finalize phase, verify that the correct value is written to memory.
        builder
            .when(local.is_finalize)
            .assert_word_eq(*local.mem.value(), local.finalize_add.value);
    }

    fn eval_compression_ops<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        // Constrain k column which loops over 64 constant values.
        for i in 0..64 {
            let octet_num = i / 8;
            let inner_index = i % 8;
            builder
                .when(local.octet_num[octet_num + 1] * local.octet[inner_index])
                .assert_all_eq(local.k, Word::<AB::F>::from(SHA_COMPRESS_K[i]));
        }

        // S1 := (e rightrotate 6) xor (e rightrotate 11) xor (e rightrotate 25).
        // Calculate e rightrotate 6.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.e,
            6,
            local.e_rr_6,
            local.is_compression,
        );
        // Calculate e rightrotate 11.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.e,
            11,
            local.e_rr_11,
            local.is_compression,
        );
        // Calculate e rightrotate 25.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.e,
            25,
            local.e_rr_25,
            local.is_compression,
        );
        // Calculate (e rightrotate 6) xor (e rightrotate 11).
        XorOperation::<AB::F>::eval(
            builder,
            local.e_rr_6.value,
            local.e_rr_11.value,
            local.s1_intermediate,
            local.is_compression,
        );
        // Calculate S1 := ((e rightrotate 6) xor (e rightrotate 11)) xor (e rightrotate 25).
        XorOperation::<AB::F>::eval(
            builder,
            local.s1_intermediate.value,
            local.e_rr_25.value,
            local.s1,
            local.is_compression,
        );

        // Calculate ch := (e and f) xor ((not e) and g).
        // Calculate e and f.
        AndOperation::<AB::F>::eval(builder, local.e, local.f, local.e_and_f, local.is_compression);
        // Calculate not e.
        NotOperation::<AB::F>::eval(builder, local.e, local.e_not, local.is_compression);
        // Calculate (not e) and g.
        AndOperation::<AB::F>::eval(
            builder,
            local.e_not.value,
            local.g,
            local.e_not_and_g,
            local.is_compression,
        );
        // Calculate ch := (e and f) xor ((not e) and g).
        XorOperation::<AB::F>::eval(
            builder,
            local.e_and_f.value,
            local.e_not_and_g.value,
            local.ch,
            local.is_compression,
        );

        // Calculate temp1 := h + S1 + ch + k[i] + w[i].
        Add5Operation::<AB::F>::eval(
            builder,
            &[local.h, local.s1.value, local.ch.value, local.k, local.mem.access.value],
            local.is_compression,
            local.temp1,
        );

        // Calculate S0 := (a rightrotate 2) xor (a rightrotate 13) xor (a rightrotate 22).
        // Calculate a rightrotate 2.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.a,
            2,
            local.a_rr_2,
            local.is_compression,
        );
        // Calculate a rightrotate 13.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.a,
            13,
            local.a_rr_13,
            local.is_compression,
        );
        // Calculate a rightrotate 22.
        FixedRotateRightOperation::<AB::F>::eval(
            builder,
            local.a,
            22,
            local.a_rr_22,
            local.is_compression,
        );
        // Calculate (a rightrotate 2) xor (a rightrotate 13).
        XorOperation::<AB::F>::eval(
            builder,
            local.a_rr_2.value,
            local.a_rr_13.value,
            local.s0_intermediate,
            local.is_compression,
        );
        // Calculate S0 := ((a rightrotate 2) xor (a rightrotate 13)) xor (a rightrotate 22).
        XorOperation::<AB::F>::eval(
            builder,
            local.s0_intermediate.value,
            local.a_rr_22.value,
            local.s0,
            local.is_compression,
        );

        // Calculate maj := (a and b) xor (a and c) xor (b and c).
        // Calculate a and b.
        AndOperation::<AB::F>::eval(builder, local.a, local.b, local.a_and_b, local.is_compression);
        // Calculate a and c.
        AndOperation::<AB::F>::eval(builder, local.a, local.c, local.a_and_c, local.is_compression);
        // Calculate b and c.
        AndOperation::<AB::F>::eval(builder, local.b, local.c, local.b_and_c, local.is_compression);
        // Calculate (a and b) xor (a and c).
        XorOperation::<AB::F>::eval(
            builder,
            local.a_and_b.value,
            local.a_and_c.value,
            local.maj_intermediate,
            local.is_compression,
        );
        // Calculate maj := ((a and b) xor (a and c)) xor (b and c).
        XorOperation::<AB::F>::eval(
            builder,
            local.maj_intermediate.value,
            local.b_and_c.value,
            local.maj,
            local.is_compression,
        );

        // Calculate temp2 := s0 + maj.
        AddOperation::<AB::F>::eval(
            builder,
            local.s0.value,
            local.maj.value,
            local.temp2,
            local.is_compression.into(),
        );

        // Calculate d + temp1 for the new value of e.
        AddOperation::<AB::F>::eval(
            builder,
            local.d,
            local.temp1.value,
            local.d_add_temp1,
            local.is_compression.into(),
        );

        // Calculate temp1 + temp2 for the new value of a.
        AddOperation::<AB::F>::eval(
            builder,
            local.temp1.value,
            local.temp2.value,
            local.temp1_add_temp2,
            local.is_compression.into(),
        );
    }

    fn eval_finalize_ops<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &ShaCompressCols<AB::Var>,
    ) {
        // In the finalize phase, need to execute h[0] + a, h[1] + b, ..., h[7] + h, for each of the
        // phase's 8 rows.
        // We can get the needed operand (a,b,c,...,h) by doing an inner product between octet and
        // [a,b,c,...,h] which will act as a selector.
        let add_operands = [local.a, local.b, local.c, local.d, local.e, local.f, local.g, local.h];
        let zero = AB::Expr::zero();
        let mut filtered_operand = Word([zero.clone(), zero.clone(), zero.clone(), zero]);
        for (i, operand) in local.octet.iter().zip(add_operands.iter()) {
            for j in 0..4 {
                filtered_operand.0[j] = filtered_operand.0[j].clone() + *i * operand.0[j];
            }
        }

        builder
            .when(local.is_finalize)
            .assert_word_eq(filtered_operand, local.finalized_operand.map(|x| x.into()));

        // finalize_add.result = h[i] + finalized_operand
        AddOperation::<AB::F>::eval(
            builder,
            local.mem.prev_value,
            local.finalized_operand,
            local.finalize_add,
            local.is_finalize.into(),
        );

        // Memory write is constrained in constrain_memory.
    }
}
