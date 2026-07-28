use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteRecord, MemoryAccessPosition, MemoryRecordEnum},
    Register,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{
    air::{BaseAirBuilder, ZKMAirBuilder},
    word::Word,
};

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    memory::{RegisterAccessCols, RegisterWriteAccessCols},
};

/// Register-register operand access for `a`/`b`, for chips whose `op_a` is written but *may* be
/// register 0 (e.g. `JumpChip`'s JR, where `op_a` is hardcoded to `$zero` -- the return address is
/// discarded -- and JALR, where it's a real, possibly-zero link register). Unlike
/// [`crate::adapter::RTypeReader`] (which asserts `op_a` is never register 0), this mirrors
/// [`crate::adapter::ITypeReader`]'s masking, just with `op_b` as a register instead of an
/// immediate.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RTypeReaderMasked<T: Copy> {
    /// The register index of `op_a` (written, possibly register 0).
    pub op_a: T,
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
}

impl<T: Copy> RTypeReaderMasked<T> {
    pub fn op_b_val(&self) -> Word<T> {
        self.op_b_access.prev_value
    }
}

impl<F: PrimeField32> RTypeReaderMasked<F> {
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        op_a: u8,
        a_record: Option<MemoryRecordEnum>,
        op_b: u32,
        b_record: Option<MemoryRecordEnum>,
    ) {
        self.op_a = F::from_canonical_u8(op_a);
        self.op_a_0 = F::from_bool(op_a == Register::ZERO as u8);
        if let Some(record) = a_record {
            self.op_a_access.populate(record, blu);
        }

        self.op_b = F::from_canonical_u32(op_b);
        if let Some(record) = b_record {
            self.op_b_access.populate(record, blu);
        }
    }
}

/// Evaluates a chip's register/register operand access (a/b) via an [`RTypeReaderMasked`].
/// `op_a_computed_value` is asserted equal to the witnessed `op_a_access.value` when not writing
/// to register 0, and ignored (`op_a_access.value` is independently asserted zero instead) when
/// writing to register 0 -- see [`crate::adapter::ITypeReader`]'s doc comment for why the write
/// can't just use a masked expression of `op_a_computed_value` directly as the interaction value.
pub fn eval_r_type_reader_masked<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &RTypeReaderMasked<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    op_a_computed_value: Word<AB::Expr>,
    do_check: AB::Expr,
) {
    builder.when(do_check.clone()).assert_bool(reader.op_a_0);

    let written_value: Word<AB::Expr> = reader.op_a_access.value.map(Into::into);
    builder
        .when(do_check.clone())
        .when(reader.op_a_0)
        .assert_word_zero(written_value.clone());
    builder
        .when(do_check.clone())
        .when_not(reader.op_a_0)
        .assert_word_eq(op_a_computed_value, written_value);

    // Register positions must be read/written in the order B, A (see `MemoryAccessPosition`'s
    // doc comment), matching the executor's own `rr_cpu`/`rw_cpu` timestamps for `execute_jump`.
    builder.eval_register_access_read(
        clk_high.clone(),
        clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
        reader.op_b.into(),
        &reader.op_b_access,
        do_check.clone(),
    );
    builder.eval_register_access_write(
        clk_high,
        clk_low + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
        reader.op_a.into(),
        &reader.op_a_access,
        do_check,
    );
}

/// Register-immediate operand access for `a`/`b`, for chips whose `op_a` is written but *may* be
/// register 0 (e.g. `JumpiChip`'s J, where `op_a` is hardcoded to `$zero`, and JAL, where it's
/// always the real link register 31) and whose `op_b` is always the instruction's own encoded
/// immediate (never a register at all) -- e.g. J/JAL's jump target. Mirrors
/// [`crate::adapter::ITypeReader`]'s masking, but `op_b` needs no register access whatsoever
/// (unlike `ITypeReader`, which reads a real `op_b` register and treats `op_c` as the immediate).
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct JTypeReader<T: Copy> {
    /// The register index of `op_a` (written, possibly register 0).
    pub op_a: T,
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,
    /// The immediate value of `op_b` (never a register).
    pub op_b: Word<T>,
}

impl<F: PrimeField32> JTypeReader<F> {
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        op_a: u8,
        a_record: Option<MemoryRecordEnum>,
        op_b: u32,
    ) {
        self.op_a = F::from_canonical_u8(op_a);
        self.op_a_0 = F::from_bool(op_a == Register::ZERO as u8);
        if let Some(record) = a_record {
            self.op_a_access.populate(record, blu);
        }

        self.op_b = Word::from(op_b);
    }
}

/// Evaluates a chip's register/immediate operand access (a/b) via a [`JTypeReader`]. See
/// [`eval_r_type_reader_masked`]'s doc comment for the masking rationale.
pub fn eval_j_type_reader<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &JTypeReader<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    op_a_computed_value: Word<AB::Expr>,
    do_check: AB::Expr,
) {
    builder.when(do_check.clone()).assert_bool(reader.op_a_0);

    let written_value: Word<AB::Expr> = reader.op_a_access.value.map(Into::into);
    builder
        .when(do_check.clone())
        .when(reader.op_a_0)
        .assert_word_zero(written_value.clone());
    builder
        .when(do_check.clone())
        .when_not(reader.op_a_0)
        .assert_word_eq(op_a_computed_value, written_value);

    builder.eval_register_access_write(
        clk_high,
        clk_low + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
        reader.op_a.into(),
        &reader.op_a_access,
        do_check,
    );
}

/// Register-immediate operand access for `a`/`b`, for chips whose `op_a` is written and
/// *guaranteed* to never be register 0 (e.g. `JumpDirectChip`'s BAL, whose link register is
/// unconditionally 31 -- unlike JAL, it isn't even encoded as a variable register field, so there
/// is no "BAL to $zero" case to route elsewhere) and whose `op_b` is always the instruction's own
/// encoded immediate. Mirrors [`crate::adapter::ITypeReaderNonZero`]'s no-masking rationale.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct JTypeReaderNonZero<T: Copy> {
    /// The register index of `op_a` (written, never register 0).
    pub op_a: T,
    pub op_a_access: RegisterAccessCols<T>,
    /// The immediate value of `op_b` (never a register).
    pub op_b: Word<T>,
}

impl<F: PrimeField32> JTypeReaderNonZero<F> {
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        op_a: u8,
        a_record: Option<MemoryRecordEnum>,
        op_b: u32,
    ) {
        self.op_a = F::from_canonical_u8(op_a);
        debug_assert_ne!(op_a, Register::ZERO as u8, "op_a==0 is impossible for this opcode");
        if let Some(record) = a_record {
            self.op_a_access.populate(record, blu);
            // Unlike a plain read, a write's current value is a *different* word that
            // `eval_register_access_write_value`'s defense-in-depth check also range-checks -- so
            // it needs its own byte-lookup event here too.
            blu.add_u8_range_checks(&record.current_record().value.to_le_bytes());
        }

        self.op_b = Word::from(op_b);
    }
}

/// Evaluates a chip's register/immediate operand access (a/b) via a [`JTypeReaderNonZero`].
/// `op_a_computed_value` is sent directly, unmasked, as `op_a`'s write value -- sound because this
/// adapter is only for chips whose `op_a` is guaranteed to never be register 0 (see its doc
/// comment).
pub fn eval_j_type_reader_non_zero<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &JTypeReaderNonZero<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    op_a_computed_value: Word<AB::Expr>,
    do_check: AB::Expr,
) {
    builder.eval_register_access_write_value(
        clk_high,
        clk_low + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
        reader.op_a.into(),
        op_a_computed_value,
        &reader.op_a_access,
        do_check,
    );
}
