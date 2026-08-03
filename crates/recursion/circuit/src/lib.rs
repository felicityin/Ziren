//! Copied from [`zkm_recursion_program`].

use hash::FieldHasherVariable;
use itertools::izip;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use std::iter::{repeat, zip};
use zkm_recursion_compiler::{
    circuit::CircuitV2Builder,
    config::{InnerConfig, OuterConfig},
    ir::{Builder, Config, DslIr, Ext, Felt, SymbolicFelt, Var, Variable},
};

pub mod basefold;
pub mod challenger;
pub mod dummy;
pub mod hash;
pub mod jagged;
pub mod logup_gkr;
pub mod machine;
pub mod merkle_tree;
pub mod shard;
pub mod sumcheck;
mod symbolic;
pub(crate) mod utils;
pub mod witness;
pub mod zerocheck;

use slop_koala_bear::{KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS};
use zkm_recursion_core::{
    chips::poseidon2_wide::{NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS},
    D, DIGEST_SIZE, PERMUTATION_WIDTH,
};

pub type Digest<C, SC> = <SC as FieldHasherVariable<C>>::DigestVariable;

pub trait CircuitConfig: Config {
    type Bit: Copy + Variable<Self>;

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit;

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F>;

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit);

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit);

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D];

    /// Reconstructs an extension field element from its base-field limbs (the inverse of
    /// `ext2felt`).
    fn felt2ext(
        builder: &mut Builder<Self>,
        felt: [Felt<<Self as Config>::F>; D],
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.ext_from_base_slice(&felt)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Self::Bit>,
    ) -> Felt<<Self as Config>::F>;

    /// Exponentiates a felt x to a list of bits in little endian. Uses precomputed powers
    /// of x.
    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F>;

    /// Evaluates `eq(x1, x2)` and reconstructs the integer value of the first half of `x1`'s
    /// bits as a felt. See `CircuitV2Builder::prefix_sum_checks_v2` for the exact contract on
    /// `x1`/`x2`'s shape.
    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        x1: Vec<Felt<Self::F>>,
        x2: Vec<Ext<Self::F, Self::EF>>,
    ) -> (Ext<Self::F, Self::EF>, Felt<Self::F>);

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Self::Bit>;

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Self::Bit>,
    ) -> Felt<<Self as Config>::F>;

    #[allow(clippy::type_complexity)]
    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>>;

    #[allow(clippy::type_complexity)]
    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>;

    fn range_check_felt(builder: &mut Builder<Self>, value: Felt<Self::F>, num_bits: usize) {
        let bits = Self::num2bits(builder, value, 31);
        for bit in bits.into_iter().skip(num_bits) {
            Self::assert_bit_zero(builder, bit);
        }
    }

    /// Applies the Poseidon2 permutation to the given array. `WrapConfig` overrides this to use
    /// the row-local `Poseidon2SBoxChip`/`Poseidon2LinearLayerChip` gadget (degree <= 3) instead
    /// of the default monolithic permutation instruction.
    fn poseidon2_permute_v2(
        builder: &mut Builder<Self>,
        input: [Felt<<Self as Config>::F>; PERMUTATION_WIDTH],
    ) -> [Felt<<Self as Config>::F>; PERMUTATION_WIDTH]
    where
        Builder<Self>: CircuitV2Builder<Self>,
    {
        CircuitV2Builder::poseidon2_permute_v2(builder, input)
    }

    /// Applies the Poseidon2 compression function to the given array.
    fn poseidon2_compress_v2(
        builder: &mut Builder<Self>,
        input: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> [Felt<<Self as Config>::F>; DIGEST_SIZE]
    where
        Builder<Self>: CircuitV2Builder<Self>,
    {
        let mut pre_iter =
            input.into_iter().chain(repeat(builder.eval(<Self as Config>::F::ZERO)));
        let pre = core::array::from_fn(move |_| pre_iter.next().unwrap());
        let post = Self::poseidon2_permute_v2(builder, pre);
        post[..DIGEST_SIZE].try_into().unwrap()
    }
}

impl CircuitConfig for InnerConfig {
    type Bit = Felt<<Self as Config>::F>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::ZERO);
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::ONE);
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        let mut result = builder.constant(Self::F::ONE);
        let mut power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod: Felt<_> = builder.eval(result * power_f);
            result = builder.eval(bit * prod + (SymbolicFelt::ONE - bit) * result);
            power_f = builder.eval(power_f * power_f);
        }
        result
    }

    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        x1: Vec<Felt<<Self as Config>::F>>,
        x2: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
    ) -> (Ext<<Self as Config>::F, <Self as Config>::EF>, Felt<<Self as Config>::F>) {
        builder.prefix_sum_checks_v2(x1, x2)
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::ONE);
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::ONE);
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WrapConfig;

impl Config for WrapConfig {
    type F = <InnerConfig as Config>::F;
    type EF = <InnerConfig as Config>::EF;
    type N = <InnerConfig as Config>::N;

    /// Saves the Poseidon2 round constants ahead of time, so `poseidon2_permute_v2`'s row-local
    /// gadget doesn't need to re-derive them on every round.
    fn initialize(builder: &mut Builder<Self>) {
        for round in 0..NUM_EXTERNAL_ROUNDS + NUM_INTERNAL_ROUNDS {
            for i in 0..PERMUTATION_WIDTH / D {
                let add_rc = if (NUM_EXTERNAL_ROUNDS / 2..NUM_EXTERNAL_ROUNDS / 2 + NUM_INTERNAL_ROUNDS)
                    .contains(&round)
                {
                    builder.constant(<Self as Config>::EF::from_base({
                        let result =
                            KoalaBear_PARTIAL_CONSTS[round - NUM_EXTERNAL_ROUNDS / 2].as_canonical_u32();
                        <Self as Config>::F::from_wrapped_u32(result)
                    }))
                } else {
                    builder.constant(<Self as Config>::EF::from_base_fn(|idx| {
                        let result = if round < NUM_EXTERNAL_ROUNDS / 2 {
                            KoalaBear_BEGIN_EXT_CONSTS[round][i * D + idx].as_canonical_u32()
                        } else {
                            KoalaBear_END_EXT_CONSTS
                                [round - NUM_INTERNAL_ROUNDS - NUM_EXTERNAL_ROUNDS / 2][i * D + idx]
                                .as_canonical_u32()
                        };
                        <Self as Config>::F::from_wrapped_u32(result)
                    }))
                };

                builder.poseidon2_constants.push(add_rc);
            }
        }
    }
}

impl CircuitConfig for WrapConfig {
    type Bit = <InnerConfig as CircuitConfig>::Bit;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::ZERO);
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::ONE);
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_chip_v2(ext)
    }

    fn felt2ext(
        builder: &mut Builder<Self>,
        felt: [Felt<<Self as Config>::F>; D],
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.felt2ext_chip_v2(felt)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        let mut result = builder.constant(Self::F::ONE);
        let mut power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod: Felt<_> = builder.eval(result * power_f);
            result = builder.eval(bit * prod + (SymbolicFelt::ONE - bit) * result);
            power_f = builder.eval(power_f * power_f);
        }
        result
    }

    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        x1: Vec<Felt<<Self as Config>::F>>,
        x2: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
    ) -> (Ext<<Self as Config>::F, <Self as Config>::EF>, Felt<<Self as Config>::F>) {
        builder.prefix_sum_checks_v2(x1, x2)
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::ONE);
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::ONE);
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }

    fn poseidon2_permute_v2(
        builder: &mut Builder<Self>,
        input: [Felt<<Self as Config>::F>; PERMUTATION_WIDTH],
    ) -> [Felt<<Self as Config>::F>; PERMUTATION_WIDTH] {
        let mut state = Self::blockify(builder, input);
        for i in 0..NUM_EXTERNAL_ROUNDS / 2 {
            state = Self::external_round(builder, state, i);
        }
        for i in 0..NUM_INTERNAL_ROUNDS {
            state[0] = Self::internal_constant_addition(builder, state[0], i);
            state[0] = Self::pow3_internal(builder, state[0]);
            state = Self::internal_linear_layer(builder, state);
        }
        for i in NUM_EXTERNAL_ROUNDS / 2..NUM_EXTERNAL_ROUNDS {
            state = Self::external_round(builder, state, i);
        }
        Self::unblockify(builder, state)
    }
}

impl WrapConfig {
    fn blockify(
        builder: &mut Builder<Self>,
        input: [Felt<<Self as Config>::F>; PERMUTATION_WIDTH],
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        core::array::from_fn(|i| {
            <Self as CircuitConfig>::felt2ext(
                builder,
                input[i * D..i * D + D].try_into().unwrap(),
            )
        })
    }

    fn unblockify(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
    ) -> [Felt<<Self as Config>::F>; PERMUTATION_WIDTH] {
        let mut ret = core::array::from_fn(|_| builder.uninit());
        for i in 0..PERMUTATION_WIDTH / D {
            let felts = <Self as CircuitConfig>::ext2felt(builder, input[i]);
            ret[i * D..i * D + D].copy_from_slice(&felts);
        }
        ret
    }

    fn external_round(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
        round_index: usize,
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        let mut state = input;
        if round_index == 0 {
            state = Self::external_linear_layer(builder, state);
        }
        state = Self::external_constant_addition(builder, state, round_index);
        #[allow(clippy::needless_range_loop)]
        for i in 0..PERMUTATION_WIDTH / D {
            state[i] = Self::pow3(builder, state[i]);
        }
        state = Self::external_linear_layer(builder, state);
        state
    }

    fn external_linear_layer(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        builder.poseidon2_external_linear_layer_v2(input)
    }

    fn internal_linear_layer(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        builder.poseidon2_internal_linear_layer_v2(input)
    }

    /// Cubes every element of the block (the external S-box case; named `pow3` since KoalaBear's
    /// Poseidon2 S-box exponent is 3, unlike SP1's BabyBear-derived `pow7` naming).
    fn pow3(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.poseidon2_external_sbox_v2(input)
    }

    /// Cubes only the first element of the block, passing the rest through unchanged (the
    /// internal S-box case).
    fn pow3_internal(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.poseidon2_internal_sbox_v2(input)
    }

    fn external_constant_addition(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
        round_index: usize,
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        let round = if round_index < NUM_EXTERNAL_ROUNDS / 2 {
            round_index
        } else {
            round_index + NUM_INTERNAL_ROUNDS
        };
        core::array::from_fn(|i| {
            let add_rc = builder.poseidon2_constants[(PERMUTATION_WIDTH / D) * round + i];
            builder.eval(input[i] + add_rc)
        })
    }

    fn internal_constant_addition(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        round_index: usize,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let round = round_index + NUM_EXTERNAL_ROUNDS / 2;
        let add_rc = builder.poseidon2_constants[(PERMUTATION_WIDTH / D) * round];
        builder.eval(input + add_rc)
    }
}

impl CircuitConfig for OuterConfig {
    type Bit = Var<<Self as Config>::N>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::ZERO);
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::ONE);
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.witness_var()
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.witness_felt()
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.witness_ext()
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        let felts = core::array::from_fn(|_| builder.uninit());
        builder.push_op(DslIr::CircuitExt2Felt(felts, ext));
        felts
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let mut result = builder.constant(Self::F::ONE);
        let power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod = builder.eval(result * power_f);
            result = builder.select_f(bit, prod, result);
            builder.assign(power_f, power_f * power_f);
        }
        result
    }

    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        x1: Vec<Felt<<Self as Config>::F>>,
        x2: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
    ) -> (Ext<<Self as Config>::F, <Self as Config>::EF>, Felt<<Self as Config>::F>) {
        assert_eq!(x1.len(), x2.len());
        let len = x1.len();
        assert!(len > 0 && len % 2 == 0);

        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::ONE));
        let mut field_acc: Felt<_> = builder.uninit();
        builder.push_op(DslIr::ImmF(field_acc, <Self as Config>::F::ZERO));
        let mut half_result = None;

        for (i, (x1_i, x2_i)) in izip!(x1, x2).enumerate() {
            // Boolean check: x1_i * (x1_i - 1) == 0.
            let x1_minus_one: Felt<_> = builder.uninit();
            builder.push_op(DslIr::SubFI(x1_minus_one, x1_i, <Self as Config>::F::ONE));
            let bool_check: Felt<_> = builder.uninit();
            builder.push_op(DslIr::MulF(bool_check, x1_i, x1_minus_one));
            builder.assert_felt_eq(bool_check, <Self as Config>::F::ZERO);

            // lagrange_term = 1 - x1_i - x2_i + 2 * x1_i * x2_i.
            let product: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulEF(product, x2_i, x1_i));
            let two_product: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::AddE(two_product, product, product));
            let one: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::ImmE(one, <Self as Config>::EF::ONE));
            let one_minus_x2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubE(one_minus_x2, one, x2_i));
            let one_minus_x1_minus_x2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubEF(one_minus_x1_minus_x2, one_minus_x2, x1_i));
            let lagrange_term: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::AddE(lagrange_term, one_minus_x1_minus_x2, two_product));

            // acc *= lagrange_term.
            let new_acc: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulE(new_acc, acc, lagrange_term));
            acc = new_acc;

            // field_acc = x1_i + 2 * field_acc, only meaningful for the first half of `x1`.
            if i < len / 2 {
                let doubled: Felt<_> = builder.uninit();
                builder.push_op(DslIr::MulFI(
                    doubled,
                    field_acc,
                    <Self as Config>::F::from_canonical_u32(2),
                ));
                let new_field_acc: Felt<_> = builder.uninit();
                builder.push_op(DslIr::AddF(new_field_acc, x1_i, doubled));
                field_acc = new_field_acc;
                if i == len / 2 - 1 {
                    half_result = Some(field_acc);
                }
            }
        }

        (acc, half_result.unwrap())
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Var<<Self as Config>::N>> {
        builder.num2bits_f_circuit(num)[..num_bits].to_vec()
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let result = builder.eval(Self::F::ZERO);
        for (i, bit) in bits.into_iter().enumerate() {
            let to_add: Felt<_> = builder.uninit();
            let pow2 = builder.constant(Self::F::from_canonical_u32(1 << i));
            let zero = builder.constant(Self::F::ZERO);
            builder.push_op(DslIr::CircuitSelectF(bit, pow2, zero, to_add));
            builder.assign(result, result + to_add);
        }
        result
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Felt<_>, Felt<_>)| -> Felt<_> {
                let result: Felt<_> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectF(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Ext<_, _>, Ext<_, _>)| -> Ext<_, _> {
                let result: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectE(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        let mut result: Felt<_> = builder.eval(Self::F::ONE);
        let one = builder.constant(Self::F::ONE);
        for (&bit, &power) in power_bits.iter().zip(two_adic_powers_of_x) {
            let multiplier = builder.select_f(bit, power, one);
            result = builder.eval(multiplier * result);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use p3_koala_bear::Poseidon2InternalLayerKoalaBear;
    use p3_symmetric::Permutation;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use slop_challenger::IopCtx;

    use zkm_core_machine::utils::setup_logger;
    use zkm_hypercube::{
        config::{default_fri_config, ZkmGlobalContext},
        prover::{AirProver, ProverSemaphore, ZkmShardProver},
        ShardVerifier,
    };
    use zkm_recursion_compiler::circuit::AsmCompiler;
    use zkm_recursion_core::{machine::RecursionAir, ExecutionRecord, RecursionProgram, Runtime};
    use zkm_stark::{inner_perm, koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use super::*;

    type SC = KoalaBearPoseidon2;
    type F = <SC as StarkGenericConfig>::Val;
    type EF = <SC as StarkGenericConfig>::Challenge;

    fn prove_and_verify(program: Arc<RecursionProgram<F>>, record: ExecutionRecord<F>) {
        // Proves against the actual production wrap machine (not `machine_wide_with_all_chips`),
        // since exercising the real `wrap_machine` chip wiring is the point of this test.
        let machine = RecursionAir::<F, 3>::wrap_machine();
        let max_log_row_count = zkm_stark::RECURSION_MAX_LOG_ROW_COUNT;
        let shard_prover = ZkmShardProver::<RecursionAir<F, 3>>::new(
            ShardVerifier::from_basefold_parameters(
                default_fri_config(),
                zkm_stark::RECURSION_LOG_STACKING_HEIGHT,
                max_log_row_count,
                machine.clone(),
            ),
        );

        let setup_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (vk, proof, _permit) = setup_rt.block_on(shard_prover.setup_and_prove_shard(
            program,
            record,
            None,
            ProverSemaphore::new(1),
        ));

        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            zkm_stark::RECURSION_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        let mut challenger = ZkmGlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        if let Err(e) = shard_verifier.verify_shard(&vk, &proof, &mut challenger) {
            panic!("Verification failed: {e:?}");
        }
    }

    /// Checks that `WrapConfig`'s row-local Poseidon2 gadget (S-box/linear-layer chips, chained
    /// through memory addresses) computes the exact same permutation as the native reference,
    /// and that the resulting program actually proves/verifies end to end.
    #[test]
    fn test_wrap_poseidon2_permute() {
        setup_logger();

        let mut builder = Builder::<WrapConfig>::default();
        let mut rng = StdRng::seed_from_u64(0xBEEFCAFE)
            .sample_iter::<[F; PERMUTATION_WIDTH], _>(rand::distributions::Standard);
        let input: [F; PERMUTATION_WIDTH] = rng.next().unwrap();
        let expected = inner_perm().permute(input);

        let input_felts = input.map(|x| builder.eval(x));
        let output_felts = WrapConfig::poseidon2_permute_v2(&mut builder, input_felts);
        let expected_felts: [Felt<_>; PERMUTATION_WIDTH] = expected.map(|x| builder.eval(x));
        for (lhs, rhs) in output_felts.into_iter().zip(expected_felts) {
            builder.assert_felt_eq(lhs, rhs);
        }

        let mut compiler = AsmCompiler::<WrapConfig>::default();
        let program = Arc::new(compiler.compile(builder.into_operations()));

        let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
            program.clone(),
            KoalaBearPoseidon2::new().perm,
        );
        runtime.run().unwrap();

        prove_and_verify(program, runtime.record);
    }

    /// Calls `WrapConfig::poseidon2_permute_v2` many times within a single builder/program,
    /// checking every call's output against the reference immediately (so a runtime panic
    /// pinpoints exactly which repetition -- if any -- first diverges). No proving, just DSL
    /// build + runtime execution, to iterate fast.
    #[test]
    fn test_wrap_poseidon2_permute_repeated() {
        setup_logger();

        const ITERS: usize = 1000;
        let mut builder = Builder::<WrapConfig>::default();
        let mut rng = StdRng::seed_from_u64(0xBEEFCAFE)
            .sample_iter::<[F; PERMUTATION_WIDTH], _>(rand::distributions::Standard);

        for _ in 0..ITERS {
            let input: [F; PERMUTATION_WIDTH] = rng.next().unwrap();
            let expected = inner_perm().permute(input);
            let input_felts = input.map(|x| builder.eval(x));
            let output_felts = WrapConfig::poseidon2_permute_v2(&mut builder, input_felts);
            let expected_felts: [Felt<_>; PERMUTATION_WIDTH] = expected.map(|x| builder.eval(x));
            for (lhs, rhs) in output_felts.into_iter().zip(expected_felts) {
                builder.assert_felt_eq(lhs, rhs);
            }
        }

        let mut compiler = AsmCompiler::<WrapConfig>::default();
        let program = Arc::new(compiler.compile(builder.into_operations()));

        let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
            program,
            KoalaBearPoseidon2::new().perm,
        );
        runtime.run().unwrap();
    }

    /// Checks `WrapConfig`'s `poseidon2_hash` (the sponge built on top of
    /// `poseidon2_permute_v2`, used for e.g. root public values digests) against a native
    /// reference sponge over `inner_perm()`, for an input spanning multiple `HASH_RATE`-sized
    /// chunks -- unlike `test_wrap_poseidon2_permute`, which only exercises a single permutation
    /// call on a full-width input.
    #[test]
    fn test_wrap_poseidon2_hash_sponge() {
        use crate::hash::Poseidon2KoalaBearHasherVariable;
        use zkm_hypercube::config::ZkmGlobalContext;

        setup_logger();

        let mut builder = Builder::<WrapConfig>::default();
        let mut rng = StdRng::seed_from_u64(0xBEEFCAFE).sample_iter::<F, _>(rand::distributions::Standard);
        let input: [F; 40] = core::array::from_fn(|_| rng.next().unwrap());

        // Native reference sponge (rate = HASH_RATE, capacity = PERMUTATION_WIDTH - HASH_RATE,
        // overwrite-mode, matching `Poseidon2KoalaBearHasherVariable::poseidon2_hash`).
        let mut state = [F::ZERO; PERMUTATION_WIDTH];
        for chunk in input.chunks(zkm_recursion_core::HASH_RATE) {
            state[..chunk.len()].copy_from_slice(chunk);
            state = inner_perm().permute(state);
        }
        let expected: [F; DIGEST_SIZE] = state[..DIGEST_SIZE].try_into().unwrap();

        let input_felts: Vec<Felt<F>> = input.iter().map(|x| builder.eval(*x)).collect();
        let output_felts =
            <ZkmGlobalContext as Poseidon2KoalaBearHasherVariable<WrapConfig>>::poseidon2_hash(
                &mut builder,
                &input_felts,
            );
        let expected_felts: [Felt<F>; DIGEST_SIZE] = expected.map(|x| builder.eval(x));
        for (lhs, rhs) in output_felts.into_iter().zip(expected_felts) {
            builder.assert_felt_eq(lhs, rhs);
        }

        let mut compiler = AsmCompiler::<WrapConfig>::default();
        let program = Arc::new(compiler.compile(builder.into_operations()));

        let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
            program.clone(),
            KoalaBearPoseidon2::new().perm,
        );
        runtime.run().unwrap();

        prove_and_verify(program, runtime.record);
    }
}
