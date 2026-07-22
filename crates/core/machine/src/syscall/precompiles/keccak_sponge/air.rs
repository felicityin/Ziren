use crate::adapter::{clk_low_expr, eval_cpu_state};
use crate::air::{MemoryAirBuilder, WordAirBuilder};
use crate::memory::MemoryCols;
use crate::operations::{IsZeroOperation, XorOperation};
use crate::syscall::precompiles::keccak_sponge::columns::{
    KeccakSpongeCols, NUM_KECCAK_SPONGE_COLS,
};
use crate::syscall::precompiles::keccak_sponge::constants::rc_value_bit;
use crate::syscall::precompiles::keccak_sponge::{
    KeccakSpongeChip, BITS_PER_LIMB, KECCAK_GENERAL_OUTPUT_U32S, KECCAK_GENERAL_RATE_U32S,
    KECCAK_STATE_U32S,
};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_keccak_air::{NUM_ROUNDS, U64_LIMBS};
use p3_matrix::Matrix;
use std::borrow::Borrow;
use std::iter::once;
use zkm_core_executor::syscalls::SyscallCode;
use zkm_hypercube::air::{AirLookup, LookupScope, ZKMAirBuilder};
use zkm_hypercube::lookup::LookupKind;

impl<F> BaseAir<F> for KeccakSpongeChip {
    fn width(&self) -> usize {
        NUM_KECCAK_SPONGE_COLS
    }
}

impl<AB> Air<AB> for KeccakSpongeChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &KeccakSpongeCols<AB::Var> = (*local).borrow();

        let clk_high = local.state.clk_high;
        let clk_low = clk_low_expr::<AB>(&local.state);
        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

        // Constrain flags
        self.eval_flags(builder, local);
        // Constrain memory
        self.eval_memory_access(builder, local);
        // Constrain the state bridging between `original_state`/memory and the permutation's
        // own `a`/`a'''` lanes at the start/end of one block's Keccak-f invocation.
        self.eval_state_keccakf(builder, local);
        // Constrain the Keccak-f round math and chain the 24 rounds of one permutation
        // invocation together via an index-keyed interaction.
        self.eval_keccakf_round(builder, local);
        // Chain one block's ending state into the next block's starting state via a second,
        // block-scoped interaction (replaces the old row-adjacent absorbed-state continuity).
        self.eval_sponge_chain(builder, local);

        // Receive syscall
        builder.receive_syscall(
            clk_high,
            clk_low,
            AB::F::from_canonical_u32(SyscallCode::KECCAK_SPONGE.syscall_id()),
            local.input_address,
            local.output_address,
            local.receive_syscall,
            LookupScope::Local,
        );

        // Xor
        let not_read_block = AB::Expr::one() - local.read_block;
        for i in 0..KECCAK_GENERAL_RATE_U32S {
            XorOperation::<AB::F>::eval(
                builder,
                local.original_state[i],
                local.block_mem[i].access.value,
                local.xored_general_rate[i],
                local.read_block,
            );
            for j in 0..4 {
                builder
                    .when(not_read_block.clone())
                    .assert_zero(local.xored_general_rate[i].value[j]);
                builder
                    .when(not_read_block.clone())
                    .assert_zero(local.block_mem[i].access.value[j]);
            }
        }

        // Range-constrain the sponge state bytes.
        //
        // `original_state` is interpreted as bytes when building u16/u64 limbs.
        // Enforce each byte is in [0, 255] to prevent unconstrained field limbs
        // from satisfying packed equalities spuriously.
        let mut original_state_bytes = Vec::with_capacity(KECCAK_STATE_U32S * 4);
        for i in 0..KECCAK_STATE_U32S {
            for j in 0..4 {
                original_state_bytes.push(local.original_state[i][j]);
            }
        }
        builder.slice_range_check_u8(&original_state_bytes, local.is_real);

        // Range-constrain memory words that are interpreted as byte-packed u16/u64 limbs.
        // `input_length_mem` is constrained against `input_len` on all real rows,
        // so keep it byte-range constrained on all real rows as well.
        // `block_mem` is only read on `read_block`,
        // and `output_mem` is only used on `write_output`.
        let mut input_len_bytes = Vec::with_capacity(4);
        for j in 0..4 {
            input_len_bytes.push(local.input_length_mem.value()[j]);
        }
        builder.slice_range_check_u8(&input_len_bytes, local.is_real);

        let mut block_mem_bytes = Vec::with_capacity(KECCAK_GENERAL_RATE_U32S * 4);
        for i in 0..KECCAK_GENERAL_RATE_U32S {
            for j in 0..4 {
                block_mem_bytes.push(local.block_mem[i].access.value[j]);
            }
        }
        builder.slice_range_check_u8(&block_mem_bytes, local.read_block);

        let mut output_mem_bytes = Vec::with_capacity(KECCAK_GENERAL_OUTPUT_U32S * 4);
        for i in 0..KECCAK_GENERAL_OUTPUT_U32S {
            for j in 0..4 {
                output_mem_bytes.push(local.output_mem[i].value()[j]);
            }
        }
        builder.slice_range_check_u8(&output_mem_bytes, local.write_output);

        // If this is the first block, absorbed bytes should be 0
        builder
            .when(local.is_first_input_block)
            .assert_eq(local.already_absorbed_u32s, AB::Expr::zero());
        // If this is the first block, the sponge state must start from the
        // fixed all-zero Keccak IV.
        let mut first_block_builder = builder.when(local.is_first_input_block);
        for i in 0..KECCAK_STATE_U32S {
            first_block_builder.assert_word_zero(local.original_state[i]);
        }
        // If this is the final block, absorbed bytes should be equal to the input length - KECCAK_GENERAL_RATE_U32S
        builder.when(local.is_final_input_block).assert_eq(
            local.already_absorbed_u32s,
            local.input_len - AB::Expr::from_canonical_u32(KECCAK_GENERAL_RATE_U32S as u32),
        );
    }
}

impl KeccakSpongeChip {
    fn eval_flags<AB: ZKMAirBuilder>(&self, builder: &mut AB, local: &KeccakSpongeCols<AB::Var>) {
        let first_block = local.is_first_input_block;
        let final_block = local.is_final_input_block;
        let not_final_block = AB::Expr::one() - final_block;

        let first_step = local.keccak.step_flags[0];
        let final_step = local.keccak.step_flags[NUM_ROUNDS - 1];

        // Defensive constraints for the summarized Keccak sub-AIR boundary:
        // enforce booleanity and mutual exclusion of first/final step flags.
        // This prevents degenerate witnesses where a single row is both
        // first-round and final-round when summary internals are hidden.
        builder.assert_bool(first_block);
        builder.assert_bool(final_block);
        builder.assert_bool(local.read_block);
        builder.when(local.is_real).assert_bool(first_step);
        builder.when(local.is_real).assert_bool(final_step);
        builder.when(local.is_real).assert_zero(first_step * final_step);

        // `is_first_input_block`/`is_final_input_block` are combinatorial functions of
        // `already_absorbed_u32s`/`input_len`, checked row-locally, rather than values
        // propagated from row to row. This replaces the old `next`-row "flag holds constant
        // across a block, resets between blocks" checks.
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.already_absorbed_u32s.into(),
            local.is_absorbed_zero,
            local.is_real.into(),
        );
        let final_diff = local.already_absorbed_u32s.into()
            - (local.input_len.into()
                - AB::Expr::from_canonical_u32(KECCAK_GENERAL_RATE_U32S as u32));
        IsZeroOperation::<AB::F>::eval(
            builder,
            final_diff,
            local.is_final_block_zero,
            local.is_real.into(),
        );
        builder.when(local.is_real).assert_eq(first_block, local.is_absorbed_zero.result);
        builder.when(local.is_real).assert_eq(final_block, local.is_final_block_zero.result);

        // receive syscall
        builder.assert_eq(first_block * first_step * local.is_real, local.receive_syscall);

        // Input block memory is only read on the first Keccak round of a real row.
        builder.assert_eq(local.read_block, first_step * local.is_real);

        // write output flag
        builder.assert_eq(final_block * final_step * local.is_real, local.write_output);

        // check the absorbed bytes
        builder.assert_eq(local.is_absorbed, final_step * not_final_block * local.is_real);
    }

    fn eval_memory_access<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakSpongeCols<AB::Var>,
    ) {
        let clk_high = local.state.clk_high;
        let clk_low = clk_low_expr::<AB>(&local.state);

        // if this is the first row, populate reading input length
        builder.eval_memory_access(
            clk_high,
            clk_low.clone(),
            local.output_address + AB::Expr::from_canonical_u32(64),
            &local.input_length_mem,
            local.receive_syscall,
        );
        // Bind the scalar `input_len` to the memory-provided 4-byte word on
        // the syscall row.
        builder
            .when(local.receive_syscall)
            .assert_eq(local.input_len, local.input_length_mem.value().reduce::<AB>());
        // Verify the input length has not changed
        builder
            .when(local.is_real)
            .assert_word_eq(*local.input_length_mem.value(), *local.input_length_mem.prev_value());

        // Read the input block
        for i in 0..KECCAK_GENERAL_RATE_U32S as u32 {
            builder.eval_memory_access(
                clk_high,
                clk_low.clone(),
                local.input_address + AB::Expr::from_canonical_u32(i * 4),
                &local.block_mem[i as usize],
                local.read_block,
            );
        }
        // Verify the input has not changed
        for i in 0..KECCAK_GENERAL_RATE_U32S {
            builder
                .when(local.is_real)
                .assert_word_eq(*local.block_mem[i].value(), *local.block_mem[i].prev_value());
        }

        // If this is the final round of the final block, write the output
        for i in 0..KECCAK_GENERAL_OUTPUT_U32S as u32 {
            builder.eval_memory_access(
                clk_high,
                clk_low.clone() + AB::Expr::one(),
                local.output_address + AB::Expr::from_canonical_u32(i * 4),
                &local.output_mem[i as usize],
                local.write_output,
            );
        }
    }

    /// Bridges `original_state`/memory to the permutation's own `a`/`a'''` lanes at the two
    /// row-local edges of one block: round 0's input (`first_step`) and the final output write
    /// (`write_output`). The block-to-block continuity in between is handled entirely by
    /// `eval_sponge_chain`'s interaction, not here.
    fn eval_state_keccakf<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakSpongeCols<AB::Var>,
    ) {
        let first_step = local.keccak.step_flags[0];
        // constrain the state
        let expr_2_pow_8 = AB::Expr::from_canonical_u32(2u32.pow(8));

        for i in 0..(KECCAK_GENERAL_RATE_U32S / 2) as u32 {
            let y_idx = i / 5;
            let x_idx = i % 5;

            // Interpret u32 memory words as u16 limbs
            let least_sig_word = local.xored_general_rate[(i * 2) as usize].value;
            let most_sig_word = local.xored_general_rate[(i * 2 + 1) as usize].value;
            let memory_limbs = [
                least_sig_word[0] + least_sig_word[1] * expr_2_pow_8.clone(),
                least_sig_word[2] + least_sig_word[3] * expr_2_pow_8.clone(),
                most_sig_word[0] + most_sig_word[1] * expr_2_pow_8.clone(),
                most_sig_word[2] + most_sig_word[3] * expr_2_pow_8.clone(),
            ];
            // On a first round, verify memory matches with local.keccak.a
            let a_value_limbs = local.keccak.a[y_idx as usize][x_idx as usize];
            for j in 0..U64_LIMBS {
                builder
                    .when(first_step * local.is_real)
                    .assert_eq(memory_limbs[j].clone(), a_value_limbs[j]);
            }
        }

        for i in (KECCAK_GENERAL_RATE_U32S / 2)..(KECCAK_STATE_U32S / 2) {
            let y_idx = i / 5;
            let x_idx = i % 5;

            let least_sig_word = local.original_state[(i * 2) as usize];
            let most_sig_word = local.original_state[(i * 2 + 1) as usize];
            let memory_limbs = [
                least_sig_word[0] + least_sig_word[1] * expr_2_pow_8.clone(),
                least_sig_word[2] + least_sig_word[3] * expr_2_pow_8.clone(),
                most_sig_word[0] + most_sig_word[1] * expr_2_pow_8.clone(),
                most_sig_word[2] + most_sig_word[3] * expr_2_pow_8.clone(),
            ];
            let a_value_limbs = local.keccak.a[y_idx as usize][x_idx as usize];
            for j in 0..U64_LIMBS {
                builder
                    .when(first_step * local.is_real)
                    .assert_eq(memory_limbs[j].clone(), a_value_limbs[j]);
            }
        }

        // if this is the final round of the final block, verify output memory with
        // local.keccak.a_prime_prime_prime
        for i in 0..(KECCAK_GENERAL_OUTPUT_U32S / 2) as u32 {
            let y_idx = i / 5;
            let x_idx = i % 5;

            let least_sig_word = local.output_mem[(i * 2) as usize].value();
            let most_sig_word = local.output_mem[(i * 2 + 1) as usize].value();
            let memory_limbs = [
                least_sig_word[0] + least_sig_word[1] * expr_2_pow_8.clone(),
                least_sig_word[2] + least_sig_word[3] * expr_2_pow_8.clone(),
                most_sig_word[0] + most_sig_word[1] * expr_2_pow_8.clone(),
                most_sig_word[2] + most_sig_word[3] * expr_2_pow_8.clone(),
            ];
            for j in 0..U64_LIMBS {
                builder.when(local.write_output).assert_eq(
                    memory_limbs[j].clone(),
                    local.keccak.a_prime_prime_prime(y_idx as usize, x_idx as usize, j),
                )
            }
        }
    }

    /// The Keccak-f round math, ported from `p3_keccak_air::KeccakAir::eval` (which is itself
    /// row-adjacent -- it reads `next` and uses `when_transition` -- and therefore incompatible
    /// with this framework's single-row constraint evaluation). Checked unconditionally on every
    /// row, relying on trace-gen to populate a self-consistent dummy Keccak-f witness on padding
    /// rows, exactly as the vendored sub-AIR itself did.
    ///
    /// The 24 rounds of one permutation invocation are chained together via an index-keyed
    /// interaction covering the 23 *internal* round-to-round transitions; round 0's input and
    /// round `NUM_ROUNDS - 1`'s output are bridged elsewhere (`eval_state_keccakf` and
    /// `eval_sponge_chain`/`write_output`), so this interaction excludes both endpoints.
    fn eval_keccakf_round<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakSpongeCols<AB::Var>,
    ) {
        let andn_gen = |a: AB::Expr, b: AB::Expr| b.clone() - a * b;
        let xor_gen = |a: AB::Expr, b: AB::Expr| a.clone() + b.clone() - a * b.double();
        let xor3_gen = |a: AB::Expr, b: AB::Expr, c: AB::Expr| xor_gen(a, xor_gen(b, c));

        // Flag constraints.
        let mut sum_flags = AB::Expr::zero();
        let mut computed_index = AB::Expr::zero();
        for i in 0..NUM_ROUNDS {
            builder.assert_bool(local.keccak.step_flags[i]);
            sum_flags = sum_flags.clone() + local.keccak.step_flags[i];
            computed_index = computed_index.clone()
                + AB::Expr::from_canonical_u32(i as u32) * local.keccak.step_flags[i];
        }
        builder.assert_one(sum_flags);
        builder.when(local.is_real).assert_eq(computed_index, local.round_index);

        // C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1]).
        for x in 0..5 {
            for z in 0..64 {
                builder.assert_bool(local.keccak.c[x][z]);
                let xor = xor3_gen(
                    local.keccak.c[x][z].into(),
                    local.keccak.c[(x + 4) % 5][z].into(),
                    local.keccak.c[(x + 1) % 5][(z + 63) % 64].into(),
                );
                let c_prime = local.keccak.c_prime[x][z];
                builder.assert_eq(c_prime, xor);
            }
        }

        // Check that the input limbs are consistent with A' and D.
        // A[x, y, z] = xor(A'[x, y, z], D[x, y, z])
        //            = xor(A'[x, y, z], C[x - 1, z], C[x + 1, z - 1])
        //            = xor(A'[x, y, z], C[x, z], C'[x, z]).
        // The last step is valid based on the identity we checked above.
        // It isn't required, but makes this check a bit cleaner.
        for y in 0..5 {
            for x in 0..5 {
                let get_bit = |z| {
                    let a_prime: AB::Var = local.keccak.a_prime[y][x][z];
                    let c: AB::Var = local.keccak.c[x][z];
                    let c_prime: AB::Var = local.keccak.c_prime[x][z];
                    xor3_gen(a_prime.into(), c.into(), c_prime.into())
                };

                for limb in 0..U64_LIMBS {
                    let a_limb = local.keccak.a[y][x][limb];
                    let computed_limb = (limb * BITS_PER_LIMB..(limb + 1) * BITS_PER_LIMB)
                        .rev()
                        .fold(AB::Expr::zero(), |acc, z| {
                            builder.assert_bool(local.keccak.a_prime[y][x][z]);
                            acc.double() + get_bit(z)
                        });
                    builder.assert_eq(computed_limb, a_limb);
                }
            }
        }

        // xor_{i=0}^4 A'[x, i, z] = C'[x, z], so for each x, z,
        // diff * (diff - 2) * (diff - 4) = 0, where
        // diff = sum_{i=0}^4 A'[x, i, z] - C'[x, z]
        for x in 0..5 {
            for z in 0..64 {
                let sum: AB::Expr = (0..5).map(|y| local.keccak.a_prime[y][x][z].into()).sum();
                let diff = sum - local.keccak.c_prime[x][z];
                let four = AB::Expr::from_canonical_u8(4);
                builder
                    .assert_zero(diff.clone() * (diff.clone() - AB::Expr::two()) * (diff - four));
            }
        }

        // A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
        for y in 0..5 {
            for x in 0..5 {
                let get_bit = |z| {
                    let andn = andn_gen(
                        local.keccak.b((x + 1) % 5, y, z).into(),
                        local.keccak.b((x + 2) % 5, y, z).into(),
                    );
                    xor_gen(local.keccak.b(x, y, z).into(), andn)
                };

                for limb in 0..U64_LIMBS {
                    let computed_limb = (limb * BITS_PER_LIMB..(limb + 1) * BITS_PER_LIMB)
                        .rev()
                        .fold(AB::Expr::zero(), |acc, z| acc.double() + get_bit(z));
                    builder.assert_eq(computed_limb, local.keccak.a_prime_prime[y][x][limb]);
                }
            }
        }

        // A'''[0, 0] = A''[0, 0] XOR RC
        for limb in 0..U64_LIMBS {
            let computed_a_prime_prime_0_0_limb = (limb * BITS_PER_LIMB
                ..(limb + 1) * BITS_PER_LIMB)
                .rev()
                .fold(AB::Expr::zero(), |acc, z| {
                    builder.assert_bool(local.keccak.a_prime_prime_0_0_bits[z]);
                    acc.double() + local.keccak.a_prime_prime_0_0_bits[z]
                });
            let a_prime_prime_0_0_limb = local.keccak.a_prime_prime[0][0][limb];
            builder.assert_eq(computed_a_prime_prime_0_0_limb, a_prime_prime_0_0_limb);
        }

        let get_xored_bit = |i| {
            let mut rc_bit_i = AB::Expr::zero();
            for r in 0..NUM_ROUNDS {
                let this_round = local.keccak.step_flags[r];
                let this_round_constant = AB::Expr::from_canonical_u8(rc_value_bit(r, i));
                rc_bit_i = rc_bit_i.clone() + this_round * this_round_constant;
            }

            xor_gen(local.keccak.a_prime_prime_0_0_bits[i].into(), rc_bit_i)
        };

        for limb in 0..U64_LIMBS {
            let a_prime_prime_prime_0_0_limb = local.keccak.a_prime_prime_prime_0_0_limbs[limb];
            let computed_a_prime_prime_prime_0_0_limb = (limb * BITS_PER_LIMB
                ..(limb + 1) * BITS_PER_LIMB)
                .rev()
                .fold(AB::Expr::zero(), |acc, z| acc.double() + get_xored_bit(z));
            builder.assert_eq(computed_a_prime_prime_prime_0_0_limb, a_prime_prime_prime_0_0_limb);
        }

        let clk_high = local.state.clk_high;
        let clk_low = clk_low_expr::<AB>(&local.state);
        let base = [
            clk_high.into(),
            clk_low,
            local.output_address.into(),
            local.input_len.into(),
            local.already_absorbed_u32s.into(),
        ];

        let receive_values = base
            .iter()
            .cloned()
            .chain(once(local.round_index.into()))
            .chain((0..5).flat_map(|y| {
                (0..5).flat_map(move |x| {
                    (0..U64_LIMBS).map(move |limb| local.keccak.a[y][x][limb].into())
                })
            }))
            .collect::<Vec<_>>();
        // Gate = `is_real * (1 - first_step)`, rewritten as the affine combination
        // `is_real - read_block` (since `read_block == first_step * is_real` exactly) --
        // interaction arguments must be degree <= 1 in the trace columns, so a literal product
        // of two columns isn't allowed here.
        builder.receive(
            AirLookup::new(
                receive_values,
                local.is_real.into() - local.read_block.into(),
                LookupKind::KeccakPermuteRound,
            ),
            LookupScope::Local,
        );

        let send_values = base
            .iter()
            .cloned()
            .chain(once(local.round_index.into() + AB::Expr::one()))
            .chain((0..5).flat_map(|y| {
                (0..5).flat_map(move |x| {
                    (0..U64_LIMBS)
                        .map(move |limb| local.keccak.a_prime_prime_prime(y, x, limb).into())
                })
            }))
            .collect::<Vec<_>>();
        // Gate = `is_real * (1 - final_step)`, rewritten as the affine combination
        // `is_real - (is_absorbed + write_output)`: since `not_final_block + final_block == 1`,
        // `is_absorbed + write_output == final_step * not_final_block * is_real
        // + final_step * final_block * is_real == final_step * is_real` exactly.
        builder.send(
            AirLookup::new(
                send_values,
                local.is_real.into() - (local.is_absorbed.into() + local.write_output.into()),
                LookupKind::KeccakPermuteRound,
            ),
            LookupScope::Local,
        );
    }

    /// Chains one block's ending permutation state (last round of a non-final block) into the
    /// next block's starting permutation state (round 0 of a non-first block), replacing the old
    /// row-adjacent `already_absorbed_u32s`/`input_address`/`original_state` continuity checks.
    /// The two true chain boundaries -- the first block's zero IV and the final block's output
    /// write -- don't need this interaction at all, since they're already row-local constant
    /// anchors (see the top-level `eval` and `eval_state_keccakf`).
    fn eval_sponge_chain<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakSpongeCols<AB::Var>,
    ) {
        let expr_2_pow_8 = AB::Expr::from_canonical_u32(2u32.pow(8));
        let clk_high = local.state.clk_high;
        let clk_low = clk_low_expr::<AB>(&local.state);
        let base = [
            clk_high.into(),
            clk_low,
            local.output_address.into(),
            local.input_len.into(),
        ];

        let send_values = base
            .iter()
            .cloned()
            .chain(once(
                local.already_absorbed_u32s.into()
                    + AB::Expr::from_canonical_u32(KECCAK_GENERAL_RATE_U32S as u32),
            ))
            .chain(once(
                local.input_address.into()
                    + AB::Expr::from_canonical_u32(KECCAK_GENERAL_RATE_U32S as u32 * 4),
            ))
            .chain((0..5).flat_map(|y| {
                (0..5).flat_map(move |x| {
                    (0..U64_LIMBS)
                        .map(move |limb| local.keccak.a_prime_prime_prime(y, x, limb).into())
                })
            }))
            .collect::<Vec<_>>();
        builder.send(
            AirLookup::new(send_values, local.is_absorbed.into(), LookupKind::KeccakSpongeBlock),
            LookupScope::Local,
        );

        // Must carry the same raw (pre-XOR) value that the send side transmits
        // (`a_prime_prime_prime`, i.e. the ending state of block i's Keccak-f) -- not
        // `local.keccak.a`, which for the rate lanes has already been XORed with the newly-read
        // input block by `eval_state_keccakf`'s round-0 bridging. `original_state` is exactly
        // this raw carry-over (trace-gen writes the un-XORed permutation output straight into
        // it), for both rate and capacity lanes alike, so recompute the same u32-pair -> 4x u16
        // limb packing `eval_state_keccakf` uses, from `original_state` instead of `keccak.a`.
        let receive_values = base
            .iter()
            .cloned()
            .chain(once(local.already_absorbed_u32s.into()))
            .chain(once(local.input_address.into()))
            .chain((0..(KECCAK_STATE_U32S / 2) as u32).flat_map(|i| {
                let least_sig_word = local.original_state[(i * 2) as usize];
                let most_sig_word = local.original_state[(i * 2 + 1) as usize];
                [
                    least_sig_word[0] + least_sig_word[1] * expr_2_pow_8.clone(),
                    least_sig_word[2] + least_sig_word[3] * expr_2_pow_8.clone(),
                    most_sig_word[0] + most_sig_word[1] * expr_2_pow_8.clone(),
                    most_sig_word[2] + most_sig_word[3] * expr_2_pow_8.clone(),
                ]
            }))
            .collect::<Vec<_>>();
        // Gate = `(1 - first_block) * first_step * is_real`, rewritten as the affine combination
        // `read_block - receive_syscall` (since `read_block == first_step * is_real` and
        // `receive_syscall == first_block * first_step * is_real` exactly).
        builder.receive(
            AirLookup::new(
                receive_values,
                local.read_block.into() - local.receive_syscall.into(),
                LookupKind::KeccakSpongeBlock,
            ),
            LookupScope::Local,
        );
    }
}
