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
/// Unlike [`super::RTypeReader`], this keeps `op_a==0` masking inline (mirroring `RTypeReader`'s
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
