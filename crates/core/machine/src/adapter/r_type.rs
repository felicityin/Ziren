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

/// Register-register (pure R-type) operand access for `a`/`b`/`c`: unlike [`super::RegisterReader`],
/// this shape guarantees `op_b`/`op_c` are *always* registers, never immediates, so:
/// - `op_b`/`op_c` need only store the register index (1 field each), not a full `Word` -- the
///   `send_program` lookup argument's word-shaped operand is reconstructed via
///   [`zkm_hypercube::word::Word::extend_var`] at zero extra column cost.
/// - every access can use the cheap [`RegisterAccessCols`]/[`RegisterWriteAccessCols`] timestamp
///   scheme in place of the general-purpose one (see their doc comments for the
///   `clk_high`-alignment invariant this depends on, maintained by
///   [`crate::memory::MemoryBumpChip`]).
///
/// Only suitable for chips whose AIR never needs an immediate variant of any operand (e.g.
/// `SubChip`; MIPS has no SUBI) and which always *write* `op_a` (never read it immutably).
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RTypeReader<T: Copy> {
    /// The register index of `op_a` (written).
    pub op_a: T,
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// Whether `op_a` is register 0.
    pub op_a_0: T,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// The register index of `op_c` (read).
    pub op_c: T,
    pub op_c_access: RegisterAccessCols<T>,
}

impl<T: Copy> RTypeReader<T> {
    pub fn op_b_val(&self) -> Word<T> {
        self.op_b_access.prev_value
    }

    pub fn op_c_val(&self) -> Word<T> {
        self.op_c_access.prev_value
    }
}

impl<F: PrimeField32> RTypeReader<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        op_a: u8,
        a_record: Option<MemoryRecordEnum>,
        op_b: u32,
        b_record: Option<MemoryRecordEnum>,
        op_c: u32,
        c_record: Option<MemoryRecordEnum>,
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

        self.op_c = F::from_canonical_u32(op_c);
        if let Some(record) = c_record {
            self.op_c_access.populate(record, blu);
        }
    }
}

/// Evaluates a chip's register operand access (a/b/c) via an [`RTypeReader`]. `op_a_computed_value`
/// is the chip's own computed result (e.g. an ALU operation's output): it's asserted equal to the
/// witnessed `op_a_access.value` when not writing to register 0, and ignored (`op_a_access.value`
/// is independently asserted zero instead) when writing to register 0 -- see
/// [`RegisterWriteAccessCols`]'s doc comment for why the write can't just use a masked expression
/// of `op_a_computed_value` directly as the interaction value.
pub fn eval_r_type_reader<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &RTypeReader<AB::Var>,
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

    // Register positions must be read/written in the order C, B, A (see
    // `MemoryAccessPosition`'s doc comment); each gets its own `clk_low` offset, matching the
    // executor's own `rr_traced`/`rw_traced` timestamps for these accesses.
    builder.eval_register_access_read(
        clk_high.clone(),
        clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
        reader.op_c.into(),
        &reader.op_c_access,
        do_check.clone(),
    );
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
