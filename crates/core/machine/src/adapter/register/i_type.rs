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

/// Register-immediate (I-type) operand access for `a`/`b`/`c`: `op_a` is always a written
/// register, `op_b` always a read register, and `op_c` is always the instruction's own encoded
/// immediate (never a register at all) -- e.g. SLTI/SLTIU here, and the existing `AddiChip`'s
/// shape (not yet migrated to this adapter).
///
/// Unlike [`crate::adapter::RTypeReader`], this keeps `op_a==0` masking inline (mirroring `RTypeReader`'s
/// own original, pre-`AluX0Chip` design) rather than routing to `AluX0Chip` -- extending
/// `AluX0Chip` to also cover immediate-`op_c` rows is a bigger structural change than warranted
/// for a single user of this adapter. Once more `ITypeReader` users exist, extend `AluX0Chip` and
/// re-narrow this adapter the same way `RTypeReader` was re-narrowed.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ITypeReader<T: Copy> {
    /// The register index of `op_a` (written).
    pub op_a: T,
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// The immediate value of `op_c` (never a register).
    pub op_c: Word<T>,
}

impl<T: Copy> ITypeReader<T> {
    pub fn op_b_val(&self) -> Word<T> {
        self.op_b_access.prev_value
    }
}

impl<F: PrimeField32> ITypeReader<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        op_a: u8,
        a_record: Option<MemoryRecordEnum>,
        op_b: u32,
        b_record: Option<MemoryRecordEnum>,
        op_c: u32,
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

        self.op_c = Word::from(op_c);
    }
}

/// Evaluates a chip's register/immediate operand access (a/b/c) via an [`ITypeReader`].
/// `op_a_computed_value` is the chip's own computed result: asserted equal to the witnessed
/// `op_a_access.value` when not writing to register 0, and ignored (`op_a_access.value` is
/// independently asserted zero instead) when writing to register 0 -- see
/// [`RegisterWriteAccessCols`]'s doc comment for why the write can't just use a masked expression
/// of `op_a_computed_value` directly as the interaction value.
pub fn eval_i_type_reader<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &ITypeReader<AB::Var>,
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
    // doc comment); `op_c` is never a register, so it needs no access at all. Each gets its own
    // `clk_low` offset, matching the executor's own `rr_traced`/`rw_traced` timestamps.
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

/// Register-immediate (I-type) operand access for `a`/`b`/`c`, for chips whose `op_a` is written
/// and *guaranteed* to never be register 0 (any such row is routed to `LoadX0Chip` instead).
/// Unlike [`ITypeReader`], this needs no `op_a_0` masking at all -- `op_a_access` uses the plain,
/// unmasked [`RegisterAccessCols`] (6 bytes), with the write value supplied directly by the
/// caller, the same way [`crate::adapter::RTypeReader`]'s own `op_a` does.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ITypeReaderNonZero<T: Copy> {
    /// The register index of `op_a` (written, never register 0).
    pub op_a: T,
    pub op_a_access: RegisterAccessCols<T>,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// The immediate value of `op_c` (never a register).
    pub op_c: Word<T>,
}

impl<T: Copy> ITypeReaderNonZero<T> {
    pub fn op_b_val(&self) -> Word<T> {
        self.op_b_access.prev_value
    }
}

impl<F: PrimeField32> ITypeReaderNonZero<F> {
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        op_a: u8,
        a_record: Option<MemoryRecordEnum>,
        op_b: u32,
        b_record: Option<MemoryRecordEnum>,
        op_c: u32,
    ) {
        self.op_a = F::from_canonical_u8(op_a);
        debug_assert_ne!(op_a, Register::ZERO as u8, "op_a==0 rows must be routed to LoadX0Chip");
        if let Some(record) = a_record {
            self.op_a_access.populate(record, blu);
            // Unlike a plain read, a write's current value is a *different* word that
            // `eval_register_access_write_value`'s defense-in-depth check also range-checks -- so
            // it needs its own byte-lookup event here too.
            blu.add_u8_range_checks(&record.current_record().value.to_le_bytes());
        }

        self.op_b = F::from_canonical_u32(op_b);
        if let Some(record) = b_record {
            self.op_b_access.populate(record, blu);
        }

        self.op_c = Word::from(op_c);
    }
}

/// Evaluates a chip's register/immediate operand access (a/b/c) via an [`ITypeReaderNonZero`].
/// `op_a_computed_value` is sent directly, unmasked, as `op_a`'s write value -- sound because
/// this adapter is only for chips whose `op_a` is guaranteed to never be register 0 (see its doc
/// comment).
pub fn eval_i_type_reader_non_zero<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &ITypeReaderNonZero<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    op_a_computed_value: Word<AB::Expr>,
    do_check: AB::Expr,
) {
    builder.eval_register_access_read(
        clk_high.clone(),
        clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
        reader.op_b.into(),
        &reader.op_b_access,
        do_check.clone(),
    );
    builder.eval_register_access_write_value(
        clk_high,
        clk_low + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
        reader.op_a.into(),
        op_a_computed_value,
        &reader.op_a_access,
        do_check,
    );
}

/// Register-immediate (I-type) operand access for `a`/`b`/`c`, for chips whose `op_a` is *read*,
/// never written -- e.g. `StoreWordChip` (`op_a` is the value being stored) and `LoadX0Chip`
/// (`op_a` is hardcoded to register 0 every row, since the loaded value is always discarded).
/// Needs no masking at all: reading a register (including `$zero`) never needs special-casing,
/// unlike writing one.
///
/// `op_a_0` is still a real, populated witness column (not hardcoded), even though it plays no
/// role in this adapter's own register-access logic: `send_program`'s lookup tuple
/// (`InstructionCols::into_iter()`) includes it, and it must match the *true* decoded value or the
/// Program lookup won't balance for a real `sw $zero, ...` (a common, real compiled-code idiom for
/// zeroing memory). `LoadX0Chip` populates it to the constant `1` for every row, since `op_a`
/// there is always register 0 by construction.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ITypeImmutableReader<T: Copy> {
    /// The register index of `op_a` (read).
    pub op_a: T,
    pub op_a_access: RegisterAccessCols<T>,
    /// Whether `op_a` is register 0. See the struct-level doc comment for why this is still a
    /// real witness even though it's unused for masking here.
    pub op_a_0: T,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// The immediate value of `op_c` (never a register).
    pub op_c: Word<T>,
}

impl<T: Copy> ITypeImmutableReader<T> {
    pub fn op_a_val(&self) -> Word<T> {
        self.op_a_access.prev_value
    }

    pub fn op_b_val(&self) -> Word<T> {
        self.op_b_access.prev_value
    }
}

impl<F: PrimeField32> ITypeImmutableReader<F> {
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        op_a: u8,
        a_record: Option<MemoryRecordEnum>,
        op_b: u32,
        b_record: Option<MemoryRecordEnum>,
        op_c: u32,
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

        self.op_c = Word::from(op_c);
    }
}

/// Evaluates a chip's register/immediate operand access (a/b/c) via an [`ITypeImmutableReader`].
/// Both `op_a`/`op_b` are plain reads -- no write, no masking.
pub fn eval_i_type_immutable_reader<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &ITypeImmutableReader<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    do_check: AB::Expr,
) {
    builder.eval_register_access_read(
        clk_high.clone(),
        clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
        reader.op_b.into(),
        &reader.op_b_access,
        do_check.clone(),
    );
    builder.eval_register_access_read(
        clk_high,
        clk_low + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
        reader.op_a.into(),
        &reader.op_a_access,
        do_check,
    );
}
