use std::iter::once;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use slop_air::{AirBuilder, AirBuilderWithPublicValues, FilteredAirBuilder};
use slop_algebra::{FieldAlgebra, Field};
use slop_uni_stark::SymbolicAirBuilder;
use strum_macros::{Display, EnumIter};

use crate::{
    air::{lookup::AirLookup, BinomialExtension},
    lookup::LookupKind,
    septic_extension::SepticExtension,
    word::Word,
};

pub const DEFAULT_PC_INC: u32 = 4;
pub const UNUSED_PC: u32 = 1;

/// The scope of a lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, EnumIter, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LookupScope {
    Global = 0,
    Local,
}

/// A builder that can send and receive messages (or lookups) with other AIRs.
pub trait MessageBuilder<M> {
    fn send(&mut self, message: M, scope: LookupScope);
    fn receive(&mut self, message: M, scope: LookupScope);
}

/// A message builder for which sending and receiving messages is a no-op.
pub trait EmptyMessageBuilder: AirBuilder {}

impl<AB: EmptyMessageBuilder, M> MessageBuilder<M> for AB {
    fn send(&mut self, _message: M, _scope: LookupScope) {}
    fn receive(&mut self, _message: M, _scope: LookupScope) {}
}

/// A trait which contains basic methods for building an AIR.
pub trait BaseAirBuilder: AirBuilder + MessageBuilder<AirLookup<Self::Expr>> {
    fn when_not<I: Into<Self::Expr>>(&mut self, condition: I) -> FilteredAirBuilder<'_, Self> {
        self.when_ne(condition, Self::F::one())
    }

    fn assert_all_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(
        &mut self,
        left: impl IntoIterator<Item = I1>,
        right: impl IntoIterator<Item = I2>,
    ) {
        for (left, right) in left.into_iter().zip_eq(right) {
            self.assert_eq(left, right);
        }
    }

    fn assert_all_zero<I: Into<Self::Expr>>(&mut self, iter: impl IntoIterator<Item = I>) {
        iter.into_iter().for_each(|expr| self.assert_zero(expr));
    }

    #[inline]
    fn if_else(
        &mut self,
        condition: impl Into<Self::Expr> + Clone,
        a: impl Into<Self::Expr> + Clone,
        b: impl Into<Self::Expr> + Clone,
    ) -> Self::Expr {
        condition.clone().into() * a.into() + (Self::Expr::one() - condition.into()) * b.into()
    }

    fn index_array(
        &mut self,
        array: &[impl Into<Self::Expr> + Clone],
        index_bitmap: &[impl Into<Self::Expr> + Clone],
    ) -> Self::Expr {
        let mut result = Self::Expr::zero();

        for (value, i) in array.iter().zip_eq(index_bitmap) {
            result = result.clone() + value.clone().into() * i.clone().into();
        }

        result
    }
}

/// A trait which contains methods for byte lookups in an AIR.
pub trait ByteAirBuilder: BaseAirBuilder {
    #[allow(clippy::too_many_arguments)]
    fn send_byte(
        &mut self,
        opcode: impl Into<Self::Expr>,
        a: impl Into<Self::Expr>,
        b: impl Into<Self::Expr>,
        c: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.send_byte_pair(opcode, a, Self::Expr::zero(), b, c, multiplicity);
    }

    #[allow(clippy::too_many_arguments)]
    fn send_byte_pair(
        &mut self,
        opcode: impl Into<Self::Expr>,
        a1: impl Into<Self::Expr>,
        a2: impl Into<Self::Expr>,
        b: impl Into<Self::Expr>,
        c: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.send(
            AirLookup::new(
                vec![opcode.into(), a1.into(), a2.into(), b.into(), c.into()],
                multiplicity.into(),
                LookupKind::Byte,
            ),
            LookupScope::Local,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn receive_byte(
        &mut self,
        opcode: impl Into<Self::Expr>,
        a: impl Into<Self::Expr>,
        b: impl Into<Self::Expr>,
        c: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.receive_byte_pair(opcode, a, Self::Expr::zero(), b, c, multiplicity);
    }

    #[allow(clippy::too_many_arguments)]
    fn receive_byte_pair(
        &mut self,
        opcode: impl Into<Self::Expr>,
        a1: impl Into<Self::Expr>,
        a2: impl Into<Self::Expr>,
        b: impl Into<Self::Expr>,
        c: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.receive(
            AirLookup::new(
                vec![opcode.into(), a1.into(), a2.into(), b.into(), c.into()],
                multiplicity.into(),
                LookupKind::Byte,
            ),
            LookupScope::Local,
        );
    }
}

/// Describes which ranges inside a hidden witness row correspond to the caller-visible
/// semantic boundary of an outlined Picus module. See [`OperationSummaryAirBuilder`].
#[derive(Debug, Clone, Default)]
pub struct PicusProjectionInfo {
    /// Column to projected-name mapping for every byte covered by the projection.
    pub col_to_name: hashbrown::HashMap<usize, String>,
    /// Projected field names to concrete column ranges in the source layout.
    pub name_to_colrange: hashbrown::HashMap<String, (usize, usize)>,
    /// Projected ranges that should be treated as module inputs.
    pub input_ranges: Vec<(usize, usize, String)>,
    /// Projected ranges that should be treated as module outputs.
    pub output_ranges: Vec<(usize, usize, String)>,
}

/// Optional hooks for builders that want to replace exact operation AIR with a summary.
///
/// Builders should return `true` only when they have emitted a semantically sound replacement for
/// the exact constraints. Returning `false` tells the caller to proceed with exact lowering.
pub trait OperationSummaryAirBuilder: AirBuilder {
    /// Returns whether `expr` is known to be the constant one in the current
    /// extraction context.
    fn is_known_one(&self, _expr: &Self::Expr) -> bool {
        false
    }

    fn try_emit_is_zero_summary(
        &mut self,
        _input: Self::Expr,
        _result: Self::Expr,
        _is_real: Self::Expr,
    ) -> bool {
        false
    }

    fn try_emit_is_zero_word_summary(
        &mut self,
        _input: Word<Self::Expr>,
        _is_lower_half_zero: Self::Expr,
        _is_upper_half_zero: Self::Expr,
        _result: Self::Expr,
        _is_real: Self::Expr,
    ) -> bool {
        false
    }

    /// Optional hook for replacing the exact `KoalaBearWordRangeChecker` AIR
    /// with an equivalent semantic summary.
    fn try_emit_koala_bear_word_range_summary(
        &mut self,
        _input: Word<Self::Expr>,
        _is_real: Self::Expr,
    ) -> bool {
        false
    }

    /// Optional hook for replacing the exact memory timestamp ordering AIR
    /// with a compact checker-style summary.
    #[allow(clippy::too_many_arguments)]
    fn try_emit_memory_timestamp_summary(
        &mut self,
        _do_check: Self::Expr,
        _shard: Self::Expr,
        _clk: Self::Expr,
        _prev_shard: Self::Expr,
        _prev_clk: Self::Expr,
        _compare_clk: Self::Expr,
        _diff_16bit_limb: Self::Expr,
        _diff_8bit_limb: Self::Expr,
        _diff_4bit_limb: Self::Expr,
    ) -> bool {
        false
    }

    /// Optional hook for replacing a large exact operation AIR with a semantic
    /// module call that exposes only projected inputs/outputs.
    fn try_emit_projected_summary<F>(
        &mut self,
        _module_name: &str,
        _projection_info: &PicusProjectionInfo,
        _current_inputs: &[Self::Expr],
        _current_outputs: &[Self::Expr],
        _source_width: usize,
        _build_exact: F,
    ) -> bool
    where
        F: FnOnce(&mut Self, &[Self::Var]),
    {
        false
    }

    /// Variant of [`Self::try_emit_projected_summary`] that lets the caller
    /// pin selected hidden witness columns to constants inside the outlined
    /// module.
    #[allow(clippy::too_many_arguments)]
    fn try_emit_projected_summary_with_hidden_consts<F>(
        &mut self,
        _module_name: &str,
        _projection_info: &PicusProjectionInfo,
        _current_inputs: &[Self::Expr],
        _current_outputs: &[Self::Expr],
        _source_width: usize,
        _hidden_consts: &[(usize, u64)],
        _build_exact: F,
    ) -> bool
    where
        F: FnOnce(&mut Self, &[Self::Var]),
    {
        false
    }

    /// Optional hook for replacing an embedded exact sub-AIR with a semantic
    /// module call whose boundary is still described by a projection.
    #[allow(clippy::too_many_arguments)]
    fn try_emit_hidden_subair_summary<F>(
        &mut self,
        _module_name: &str,
        _projection_info: &PicusProjectionInfo,
        _current_inputs: &[Self::Expr],
        _current_outputs: &[Self::Expr],
        _source_width: usize,
        _source_local_only: bool,
        _build_exact: F,
    ) -> bool
    where
        F: FnOnce(&mut Self),
    {
        false
    }
}

/// A trait which contains methods related to ALU/instruction lookups in an AIR.
pub trait InstructionAirBuilder: BaseAirBuilder {
    #[allow(clippy::too_many_arguments)]
    fn send_instruction(
        &mut self,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr> + Clone,
        pc: impl Into<Self::Expr>,
        next_pc: impl Into<Self::Expr>,
        next_next_pc: impl Into<Self::Expr>,
        num_extra_cycles: impl Into<Self::Expr>,
        opcode: impl Into<Self::Expr>,
        a: Word<impl Into<Self::Expr>>,
        b: Word<impl Into<Self::Expr>>,
        c: Word<impl Into<Self::Expr>>,
        hi: Word<impl Into<Self::Expr>>,
        op_a_immutable: impl Into<Self::Expr>,
        is_rw_a: impl Into<Self::Expr>,
        is_check_memory: impl Into<Self::Expr>,
        is_halt: impl Into<Self::Expr>,
        is_sequential: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        let values = once(shard.into())
            .chain(once(clk.into()))
            .chain(once(pc.into()))
            .chain(once(next_pc.into()))
            .chain(once(next_next_pc.into()))
            .chain(once(num_extra_cycles.into()))
            .chain(once(opcode.into()))
            .chain(a.0.into_iter().map(Into::into))
            .chain(b.0.into_iter().map(Into::into))
            .chain(c.0.into_iter().map(Into::into))
            .chain(hi.0.into_iter().map(Into::into))
            .chain(once(op_a_immutable.into()))
            .chain(once(is_rw_a.into()))
            .chain(once(is_check_memory.into()))
            .chain(once(is_halt.into()))
            .chain(once(is_sequential.into()))
            .collect();

        self.send(AirLookup::new(values, multiplicity.into(), LookupKind::Instruction), LookupScope::Local);
    }

    #[allow(clippy::too_many_arguments)]
    fn receive_instruction(
        &mut self,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr> + Clone,
        pc: impl Into<Self::Expr>,
        next_pc: impl Into<Self::Expr>,
        next_next_pc: impl Into<Self::Expr>,
        num_extra_cycles: impl Into<Self::Expr>,
        opcode: impl Into<Self::Expr>,
        a: Word<impl Into<Self::Expr>>,
        b: Word<impl Into<Self::Expr>>,
        c: Word<impl Into<Self::Expr>>,
        hi: Word<impl Into<Self::Expr>>,
        op_a_immutable: impl Into<Self::Expr>,
        is_rw_a: impl Into<Self::Expr>,
        is_check_memory: impl Into<Self::Expr>,
        is_halt: impl Into<Self::Expr>,
        is_sequential: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        let values = once(shard.into())
            .chain(once(clk.into()))
            .chain(once(pc.into()))
            .chain(once(next_pc.into()))
            .chain(once(next_next_pc.into()))
            .chain(once(num_extra_cycles.into()))
            .chain(once(opcode.into()))
            .chain(a.0.into_iter().map(Into::into))
            .chain(b.0.into_iter().map(Into::into))
            .chain(c.0.into_iter().map(Into::into))
            .chain(hi.0.into_iter().map(Into::into))
            .chain(once(op_a_immutable.into()))
            .chain(once(is_rw_a.into()))
            .chain(once(is_check_memory.into()))
            .chain(once(is_halt.into()))
            .chain(once(is_sequential.into()))
            .collect();

        self.receive(AirLookup::new(values, multiplicity.into(), LookupKind::Instruction), LookupScope::Local);
    }

    /// Sends this row's own `(clk_high, clk, pc, next_pc)` as the incoming CPU state for whichever
    /// row receives it, and separately receives this row's own incoming
    /// `(clk_high, clk, pc, next_pc)` via [`Self::receive_state`] -- together these replace the
    /// old row-adjacency `shard`/`clk`/`pc` chaining, matched by value instead of physical row
    /// position. `next_pc` here is the pc two steps ahead (MIPS has a branch-delay slot, so
    /// successor state carries two forward pcs, not one): the row-local pc always advances by
    /// [`DEFAULT_PC_INC`] to reach the delay slot, and `next_pc` is the (possibly branch/jump-
    /// resolved) pc after the delay slot. `clk_high` is the clk's high limb (see
    /// `zkm_core_machine::adapter::state::CpuState`'s doc comment) -- constant within a shard, so
    /// it never needs carry handling here.
    fn send_state(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        pc: impl Into<Self::Expr>,
        next_pc: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.send(
            AirLookup::new(
                vec![clk_high.into(), clk.into(), pc.into(), next_pc.into()],
                multiplicity.into(),
                LookupKind::State,
            ),
            LookupScope::Local,
        );
    }

    /// See [`Self::send_state`].
    fn receive_state(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        pc: impl Into<Self::Expr>,
        next_pc: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.receive(
            AirLookup::new(
                vec![clk_high.into(), clk.into(), pc.into(), next_pc.into()],
                multiplicity.into(),
                LookupKind::State,
            ),
            LookupScope::Local,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn send_alu(
        &mut self,
        opcode: impl Into<Self::Expr>,
        a: Word<impl Into<Self::Expr>>,
        b: Word<impl Into<Self::Expr>>,
        c: Word<impl Into<Self::Expr>>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.send_alu_with_hi(opcode, a, b, c, Word([Self::F::zero(); 4]), multiplicity);
    }

    #[allow(clippy::too_many_arguments)]
    fn send_alu_with_hi(
        &mut self,
        opcode: impl Into<Self::Expr>,
        a: Word<impl Into<Self::Expr>>,
        b: Word<impl Into<Self::Expr>>,
        c: Word<impl Into<Self::Expr>>,
        hi: Word<impl Into<Self::Expr>>,
        multiplicity: impl Into<Self::Expr>,
    ) {
        self.send_instruction(
            Self::Expr::zero(),
            Self::Expr::zero(),
            Self::Expr::from_canonical_u32(UNUSED_PC),
            Self::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC),
            Self::Expr::from_canonical_u32(UNUSED_PC + DEFAULT_PC_INC + DEFAULT_PC_INC),
            Self::Expr::zero(),
            opcode,
            a,
            b,
            c,
            hi,
            Self::Expr::zero(),
            Self::Expr::zero(),
            Self::Expr::zero(),
            Self::Expr::zero(),
            Self::Expr::one(),
            multiplicity,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn send_syscall(
        &mut self,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr> + Clone,
        syscall_id: impl Into<Self::Expr> + Clone,
        arg1: impl Into<Self::Expr> + Clone,
        arg2: impl Into<Self::Expr> + Clone,
        multiplicity: impl Into<Self::Expr>,
        scope: LookupScope,
    ) {
        self.send(
            AirLookup::new(
                vec![shard.clone().into(), clk.clone().into(), syscall_id.clone().into(), arg1.clone().into(), arg2.clone().into()],
                multiplicity.into(),
                LookupKind::Syscall,
            ),
            scope,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn receive_syscall(
        &mut self,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr> + Clone,
        syscall_id: impl Into<Self::Expr> + Clone,
        arg1: impl Into<Self::Expr> + Clone,
        arg2: impl Into<Self::Expr> + Clone,
        multiplicity: impl Into<Self::Expr>,
        scope: LookupScope,
    ) {
        self.receive(
            AirLookup::new(
                vec![shard.clone().into(), clk.clone().into(), syscall_id.clone().into(), arg1.clone().into(), arg2.clone().into()],
                multiplicity.into(),
                LookupKind::Syscall,
            ),
            scope,
        );
    }

    fn word_to_halves(word: Word<impl Into<Self::Expr> + Copy>) -> [Self::Expr; 2] {
        let c256 = Self::Expr::from_canonical_u32(256);
        [word.0[0].into() + word.0[1].into() * c256.clone(), word.0[2].into() + word.0[3].into() * c256]
    }

    #[allow(clippy::too_many_arguments)]
    fn send_syscall_result(
        &mut self,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr> + Clone,
        result_word: Word<impl Into<Self::Expr> + Copy>,
        arg1_word: Word<impl Into<Self::Expr> + Copy>,
        arg2_word: Word<impl Into<Self::Expr> + Copy>,
        multiplicity: impl Into<Self::Expr>,
        scope: LookupScope,
    ) {
        let [r_lo, r_hi] = Self::word_to_halves(result_word);
        let [a0_lo, a0_hi] = Self::word_to_halves(arg1_word);
        let [a1_lo, a1_hi] = Self::word_to_halves(arg2_word);
        let values: Vec<Self::Expr> = vec![shard.into(), clk.into(), r_lo, r_hi, a0_lo, a0_hi, a1_lo, a1_hi];
        self.send(AirLookup::new(values, multiplicity.into(), LookupKind::SyscallResult), scope);
    }

    #[allow(clippy::too_many_arguments)]
    fn receive_syscall_result(
        &mut self,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr> + Clone,
        result_word: Word<impl Into<Self::Expr> + Copy>,
        arg1_word: Word<impl Into<Self::Expr> + Copy>,
        arg2_word: Word<impl Into<Self::Expr> + Copy>,
        multiplicity: impl Into<Self::Expr>,
        scope: LookupScope,
    ) {
        let [r_lo, r_hi] = Self::word_to_halves(result_word);
        let [a0_lo, a0_hi] = Self::word_to_halves(arg1_word);
        let [a1_lo, a1_hi] = Self::word_to_halves(arg2_word);
        let values: Vec<Self::Expr> = vec![shard.into(), clk.into(), r_lo, r_hi, a0_lo, a0_hi, a1_lo, a1_hi];
        self.receive(AirLookup::new(values, multiplicity.into(), LookupKind::SyscallResult), scope);
    }

    #[allow(clippy::too_many_arguments)]
    fn send_syscall_result_packed(
        &mut self,
        shard: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        result_lo: impl Into<Self::Expr>,
        result_hi: impl Into<Self::Expr>,
        arg1_lo: impl Into<Self::Expr>,
        arg1_hi: impl Into<Self::Expr>,
        arg2_lo: impl Into<Self::Expr>,
        arg2_hi: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
        scope: LookupScope,
    ) {
        let values: Vec<Self::Expr> = vec![
            shard.into(),
            clk.into(),
            result_lo.into(),
            result_hi.into(),
            arg1_lo.into(),
            arg1_hi.into(),
            arg2_lo.into(),
            arg2_hi.into(),
        ];
        self.send(AirLookup::new(values, multiplicity.into(), LookupKind::SyscallResult), scope);
    }

    #[allow(clippy::too_many_arguments)]
    fn receive_syscall_result_packed(
        &mut self,
        shard: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        result_lo: impl Into<Self::Expr>,
        result_hi: impl Into<Self::Expr>,
        arg1_lo: impl Into<Self::Expr>,
        arg1_hi: impl Into<Self::Expr>,
        arg2_lo: impl Into<Self::Expr>,
        arg2_hi: impl Into<Self::Expr>,
        multiplicity: impl Into<Self::Expr>,
        scope: LookupScope,
    ) {
        let values: Vec<Self::Expr> = vec![
            shard.into(),
            clk.into(),
            result_lo.into(),
            result_hi.into(),
            arg1_lo.into(),
            arg1_hi.into(),
            arg2_lo.into(),
            arg2_hi.into(),
        ];
        self.receive(AirLookup::new(values, multiplicity.into(), LookupKind::SyscallResult), scope);
    }
}

/// A builder that can operate on extension elements.
pub trait ExtensionAirBuilder: BaseAirBuilder {
    fn assert_ext_eq<I: Into<Self::Expr>>(&mut self, left: BinomialExtension<I>, right: BinomialExtension<I>) {
        for (left, right) in left.0.into_iter().zip(right.0) {
            self.assert_eq(left, right);
        }
    }

    fn assert_is_base_element<I: Into<Self::Expr> + Clone>(&mut self, element: BinomialExtension<I>) {
        let base_slice = element.as_base_slice();
        let degree = base_slice.len();
        base_slice[1..degree].iter().for_each(|coeff| {
            self.assert_zero(coeff.clone().into());
        });
    }

    fn if_else_ext(
        &mut self,
        condition: impl Into<Self::Expr> + Clone,
        a: BinomialExtension<impl Into<Self::Expr> + Clone>,
        b: BinomialExtension<impl Into<Self::Expr> + Clone>,
    ) -> BinomialExtension<Self::Expr> {
        BinomialExtension(std::array::from_fn(|i| self.if_else(condition.clone(), a.0[i].clone(), b.0[i].clone())))
    }
}

/// A builder that can operate on septic extension elements.
pub trait SepticExtensionAirBuilder: BaseAirBuilder {
    fn assert_septic_ext_eq<I: Into<Self::Expr>>(&mut self, left: SepticExtension<I>, right: SepticExtension<I>) {
        for (left, right) in left.0.into_iter().zip(right.0) {
            self.assert_eq(left, right);
        }
    }
}

/// A trait that contains the common helper methods for building Ziren machine AIRs.
pub trait MachineAirBuilder:
    BaseAirBuilder + ExtensionAirBuilder + SepticExtensionAirBuilder + AirBuilderWithPublicValues
{
}

/// A trait which contains all helper methods for building Ziren machine AIRs.
pub trait ZKMAirBuilder:
    MachineAirBuilder + ByteAirBuilder + InstructionAirBuilder + OperationSummaryAirBuilder
{
}

impl<AB: AirBuilder + MessageBuilder<M>, M> MessageBuilder<M> for FilteredAirBuilder<'_, AB> {
    fn send(&mut self, message: M, scope: LookupScope) {
        self.inner.send(message, scope);
    }

    fn receive(&mut self, message: M, scope: LookupScope) {
        self.inner.receive(message, scope);
    }
}

impl<AB: AirBuilder + MessageBuilder<AirLookup<AB::Expr>>> BaseAirBuilder for AB {}
impl<AB: BaseAirBuilder> ByteAirBuilder for AB {}
impl<AB: BaseAirBuilder> InstructionAirBuilder for AB {}
impl<AB: BaseAirBuilder> ExtensionAirBuilder for AB {}
impl<AB: BaseAirBuilder> SepticExtensionAirBuilder for AB {}
impl<AB: BaseAirBuilder + AirBuilderWithPublicValues> MachineAirBuilder for AB {}
impl<AB: BaseAirBuilder + AirBuilderWithPublicValues + OperationSummaryAirBuilder> ZKMAirBuilder for AB {}
impl<AB: EmptyMessageBuilder + AirBuilderWithPublicValues> OperationSummaryAirBuilder for AB {}

impl<F: Field> EmptyMessageBuilder for SymbolicAirBuilder<F> {}
