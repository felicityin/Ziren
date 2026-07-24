use p3_field::{FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteRecord, MemoryAccessPosition, MemoryRecordEnum},
    Register,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::ZKMAirBuilder, word::Word};

use crate::{air::MemoryAirBuilder, memory::RegisterAccessCols};

/// Register-register (pure R-type) operand access for `a`/`b`/`c`: unlike [`crate::adapter::RegisterReader`],
/// this shape guarantees `op_b`/`op_c` are *always* registers, never immediates, so:
/// - `op_b`/`op_c` need only store the register index (1 field each), not a full `Word` -- the
///   `send_program` lookup argument's word-shaped operand is reconstructed via
///   [`zkm_hypercube::word::Word::extend_var`] at zero extra column cost.
/// - every access can use the cheap [`RegisterAccessCols`] timestamp scheme in place of the
///   general-purpose one (see its doc comment for the `clk_high`-alignment invariant this depends
///   on, maintained by [`crate::memory::MemoryBumpChip`]).
///
/// Only suitable for chips whose AIR never needs an immediate variant of any operand (e.g.
/// `SubChip`; MIPS has no SUBI), which always *write* `op_a` (never read it immutably), and whose
/// `op_a` is guaranteed to never be register 0 -- any row that would write to `$zero` must be
/// routed to `AluX0Chip` instead (see its doc comment), since a masked write-value expression
/// would be unsound for the register-consistency interaction. This is why `op_a_access` uses the
/// plain [`RegisterAccessCols`] (6 bytes) rather than a write-shaped column with its own witnessed
/// value: the write value is always the caller's raw computed result, never masked.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RTypeReader<T: Copy> {
    /// The register index of `op_a` (written, never register 0).
    pub op_a: T,
    pub op_a_access: RegisterAccessCols<T>,
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
        debug_assert_ne!(op_a, Register::ZERO as u8, "op_a==0 rows must be routed to AluX0Chip");
        if let Some(record) = a_record {
            self.op_a_access.populate(record, blu);
            // Unlike a plain read (where the current value is the same as `prev_value`, already
            // range-checked by `populate` above), a write's current value is a *different* word
            // that `eval_register_access_write_value`'s defense-in-depth check also range-checks
            // -- so it needs its own byte-lookup event here too.
            blu.add_u8_range_checks(&record.current_record().value.to_le_bytes());
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
/// is the chip's own computed result (e.g. an ALU operation's output): it's sent directly, unmasked,
/// as `op_a`'s write value -- sound because `RTypeReader` is only for chips whose `op_a` is
/// guaranteed to never be register 0 (that case is routed to `AluX0Chip` instead; see
/// [`RTypeReader`]'s doc comment).
pub fn eval_r_type_reader<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &RTypeReader<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    op_a_computed_value: Word<AB::Expr>,
    do_check: AB::Expr,
) {
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
    builder.eval_register_access_write_value(
        clk_high,
        clk_low + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
        reader.op_a.into(),
        op_a_computed_value,
        &reader.op_a_access,
        do_check,
    );
}
