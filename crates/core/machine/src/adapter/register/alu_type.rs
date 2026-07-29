use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteRecord, MemoryAccessPosition, MemoryRecordEnum},
    Register,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::ZKMAirBuilder, word::Word};

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    memory::RegisterAccessCols,
};

/// Register operand access for `a`/`b`/`c`, for chips whose `op_b` is *always* a register (never
/// an immediate) but whose `op_c` may be either a register or the instruction's own encoded
/// immediate, selected at runtime by `imm_c` -- e.g. `BitwiseChip` (XOR/OR/AND/NOR share their
/// opcode with the register-immediate XORI/ORI/ANDI forms, unlike `RTypeReader`'s chips, which
/// have no immediate variant of any operand at all).
///
/// Like [`crate::adapter::RTypeReader`]:
/// - `op_b` needs only its register index (1 field), not a full `Word` -- the `send_program`
///   lookup argument's word-shaped operand is reconstructed via
///   [`zkm_hypercube::word::Word::extend_var`] at zero extra column cost.
/// - every register access uses the cheap [`RegisterAccessCols`] timestamp scheme.
/// - only suitable for chips that always *write* `op_a` (never read it immutably) and whose
///   `op_a` is guaranteed to never be register 0 -- any row that would write to `$zero` must be
///   routed to `AluX0Chip` instead (see its doc comment), same reasoning as `RTypeReader`.
///
/// Unlike `RTypeReader`, `op_c` needs a full `Word` (to hold a genuine immediate value) plus its
/// own `imm_c` selector; when `imm_c` is set, `op_c_access` is populated as a faked "read" whose
/// value is the immediate itself (mirroring [`crate::adapter::ITypeReader`]'s handling of a
/// register-vs-immediate `op_c`), so `op_c_val()` returns the right value either way while the
/// real register-consistency interaction is skipped (`do_check` gated by `1 - imm_c`).
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AluTypeReader<T: Copy> {
    /// The register index of `op_a` (written, never register 0).
    pub op_a: T,
    pub op_a_access: RegisterAccessCols<T>,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// Either the register index of `op_c` (byte 0, when `imm_c` is unset) or its immediate value
    /// (when `imm_c` is set).
    pub op_c: Word<T>,
    pub op_c_access: RegisterAccessCols<T>,
    /// Whether `op_c` is an immediate value.
    pub imm_c: T,
}

impl<T: Copy> AluTypeReader<T> {
    pub fn op_b_val(&self) -> Word<T> {
        self.op_b_access.prev_value
    }

    pub fn op_c_val(&self) -> Word<T> {
        self.op_c_access.prev_value
    }
}

impl<F: PrimeField32> AluTypeReader<F> {
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
        imm_c: bool,
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

        self.imm_c = F::from_bool(imm_c);
        self.op_c = Word::from(op_c);
        if imm_c {
            self.op_c_access.prev_value = self.op_c;
        } else if let Some(record) = c_record {
            self.op_c_access.populate(record, blu);
        }
    }
}

/// Evaluates a chip's register operand access (a/b/c) via an [`AluTypeReader`].
/// `op_a_computed_value` is the chip's own computed result: sent directly, unmasked, as `op_a`'s
/// write value -- sound because `AluTypeReader` is only for chips whose `op_a` is guaranteed to
/// never be register 0 (see its doc comment).
pub fn eval_alu_type_reader<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &AluTypeReader<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    op_a_computed_value: Word<AB::Expr>,
    do_check: AB::Expr,
) {
    builder.when(do_check.clone()).assert_bool(reader.imm_c);
    // Defense-in-depth: force `imm_c` to 0 outside real rows too, so `do_check - imm_c` below
    // (an affine combination, required since lookup multiplicities must have degree <= 1 --
    // unlike a regular constraint, `do_check * (1 - imm_c)` isn't allowed here) can only ever
    // land on 0 or 1.
    builder.when_not(do_check.clone()).assert_zero(reader.imm_c);

    // If `op_c` is an immediate, assert its value is copied into `op_c_access.prev_value` (so
    // `op_c_val()` returns it either way).
    builder
        .when(do_check.clone() * Into::<AB::Expr>::into(reader.imm_c))
        .assert_word_eq(reader.op_c_access.prev_value, reader.op_c);

    // Register positions must be read/written in the order C, B, A (see
    // `MemoryAccessPosition`'s doc comment); each gets its own `clk_low` offset, matching the
    // executor's own `rr_traced`/`rw_traced` timestamps for these accesses. `op_c`'s access is
    // skipped (zero multiplicity) when it's an immediate -- `do_check - imm_c` (not
    // `do_check * (1 - imm_c)`) to keep this an affine lookup multiplicity.
    builder.eval_register_access_read(
        clk_high.clone(),
        clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
        reader.op_c[0].into(),
        &reader.op_c_access,
        do_check.clone() - Into::<AB::Expr>::into(reader.imm_c),
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
