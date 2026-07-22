use p3_field::{FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemoryAccessPosition},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::ZKMAirBuilder, word::Word};

use crate::{
    adapter::InstructionCols,
    air::{MemoryAirBuilder, WordAirBuilder},
    memory::{MemoryCols, MemoryReadCols, MemoryReadWriteCols},
};

/// Register operand access (a/b/c) shared by every chip that reads/writes MIPS registers:
/// resolves immediates vs. memory reads for `b`/`c`, and the read-write access for `a`. Lifted
/// out of `CpuChip` so any chip can embed it.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RegisterReader<T: Copy> {
    pub op_a_access: MemoryReadWriteCols<T>,
    pub op_b_access: MemoryReadCols<T>,
    pub op_c_access: MemoryReadCols<T>,
}

impl<T: Copy> RegisterReader<T> {
    pub fn op_a_val(&self) -> Word<T> {
        *self.op_a_access.value()
    }

    pub fn op_b_val(&self) -> Word<T> {
        *self.op_b_access.value()
    }

    pub fn op_c_val(&self) -> Word<T> {
        *self.op_c_access.value()
    }
}

impl<F: PrimeField32> RegisterReader<F> {
    /// Registers the byte-range-check lookups for `op_a`'s written value. [`eval_register_reader`]
    /// unconditionally range-checks it (`slice_range_check_u8`, on top of the read/write access
    /// consistency checks `eval_memory_access` already covers) since some chips can witness an
    /// otherwise-unconstrained word there; every caller must register the matching lookup event
    /// at trace-gen time, same as any other `send_byte`.
    pub fn populate_op_a_range_checks(&self, blu: &mut impl ByteRecord) {
        let bytes = self.op_a_access.access.value.0.map(|x| x.as_canonical_u32() as u8);
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: bytes[0],
            c: bytes[1],
        });
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: bytes[2],
            c: bytes[3],
        });
    }
}

/// Evaluates a chip's register operand access (a/b/c) via a [`RegisterReader`]. `op_a_value`,
/// `hi_or_prev_a`, `is_rw_a`, `op_a_immutable`, `is_real` take expressions (not just raw columns)
/// so a chip can pass a computed value (e.g. a multi-opcode-variant mux) instead of a single
/// witnessed column.
#[allow(clippy::too_many_arguments)]
pub fn eval_register_reader<AB: ZKMAirBuilder>(
    builder: &mut AB,
    reader: &RegisterReader<AB::Var>,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    instruction: &InstructionCols<AB::Var>,
    op_a_value: Word<AB::Expr>,
    hi_or_prev_a: Word<AB::Expr>,
    is_rw_a: AB::Expr,
    op_a_immutable: AB::Expr,
    is_real: AB::Expr,
) {
    // Load immediates into b and c, if the immediate flags are on.
    builder.when(instruction.imm_b).assert_word_eq(reader.op_b_val(), instruction.op_b);
    builder.when(instruction.imm_c).assert_word_eq(reader.op_c_val(), instruction.op_c);

    // If they are not immediates, read `b` and `c` from memory.
    builder.eval_memory_access(
        clk_high.clone(),
        clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::B as u32),
        instruction.op_b[0],
        &reader.op_b_access,
        AB::Expr::one() - instruction.imm_b,
    );
    builder.eval_memory_access(
        clk_high.clone(),
        clk_low.clone() + AB::F::from_canonical_u32(MemoryAccessPosition::C as u32),
        instruction.op_c[0],
        &reader.op_c_access,
        AB::Expr::one() - instruction.imm_c,
    );

    // If we are writing to register 0, then the new value should be zero.
    builder.when(instruction.op_a_0).assert_word_zero(reader.op_a_val());

    // If we are not writing to register 0, then the new value should equal to op_a_value.
    builder.when_not(instruction.op_a_0).assert_word_eq(op_a_value.clone(), reader.op_a_val());

    // If `op_a` is an immutable read from register 0, the logical operand sent to
    // instruction chips must also be zero. Writes to register 0 are intentionally excluded
    // because their computed result is discarded.
    builder
        .when(Into::<AB::Expr>::into(instruction.op_a_0) * op_a_immutable.clone())
        .assert_word_zero(op_a_value);

    // If we are maddu，msubu，madd, msub, ins，mne, meq, syscall and memory instruction then
    // the hi_or_prev_a should equal to op_a_access.prev_value.
    builder.when(is_rw_a).assert_word_eq(hi_or_prev_a, reader.op_a_access.prev_value);

    // Write the `a` or the result to the first register described in the instruction unless
    // we are performing a branch or a store.
    builder.eval_memory_access(
        clk_high,
        clk_low + AB::F::from_canonical_u32(MemoryAccessPosition::A as u32),
        instruction.op_a,
        &reader.op_a_access,
        is_real.clone(),
    );

    // Always range check the word value in `op_a`, as JUMP instructions may witness an
    // invalid word and write it to memory.
    builder.slice_range_check_u8(&reader.op_a_access.access.value.0, is_real);

    // If we are performing a branch or a store or `teq`, then the value of `a` is the
    // previous value.
    builder.when(op_a_immutable).assert_word_eq(reader.op_a_val(), reader.op_a_access.prev_value);
}
