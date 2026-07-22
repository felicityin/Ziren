use p3_field::{FieldAlgebra, PrimeField};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::ZKMAirBuilder;

use crate::air::MemoryAirBuilder;

/// `clk_high` and `clk_low` (the latter as 16+8-bit limbs), the bookkeeping every chip that
/// executes a MIPS instruction needs. `clk_low` is range-checked to 24 bits every row;
/// `clk_high`'s correctness comes from the `LookupKind::State` chain (see [`eval_state_chain`])
/// and the `clk_high`-transition chip it links to, not from a per-row check here -- unlike the
/// shard number this replaces, `clk_high` may change partway through a shard. Lifted out of
/// `CpuChip` so any chip can embed it.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CpuState<T: Copy> {
    pub clk_high: T,
    pub clk_16bit_limb: T,
    pub clk_8bit_limb: T,
}

impl<F: PrimeField> CpuState<F> {
    pub fn populate(&mut self, blu: &mut impl ByteRecord, clk: u64) {
        self.clk_high = F::from_canonical_u64(clk >> 24);

        let clk_16bit_limb = (clk & 0xffff) as u16;
        let clk_8bit_limb = ((clk >> 16) & 0xff) as u8;
        self.clk_16bit_limb = F::from_canonical_u16(clk_16bit_limb);
        self.clk_8bit_limb = F::from_canonical_u8(clk_8bit_limb);

        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U16Range,
            clk_16bit_limb,
            0,
            0,
            0,
        ));
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U8Range,
            0,
            0,
            0,
            clk_8bit_limb as u8,
        ));
    }
}

/// Reassembles `clk_low` (the low 24 bits of clk) from its two limbs.
pub fn clk_low_expr<AB: ZKMAirBuilder>(state: &CpuState<AB::Var>) -> AB::Expr {
    AB::Expr::from_canonical_u32(1u32 << 16) * state.clk_8bit_limb + state.clk_16bit_limb
}

/// Range-checks `clk_low`. `clk_low` should be [`clk_low_expr`]'s reassembled expression.
pub fn eval_cpu_state<AB: ZKMAirBuilder>(
    builder: &mut AB,
    state: &CpuState<AB::Var>,
    clk_low: AB::Expr,
    is_real: AB::Expr,
) {
    builder.eval_range_check_24bits(clk_low, state.clk_16bit_limb, state.clk_8bit_limb, is_real);
}

/// Chains `(clk, pc) -> (next_clk, next_pc)` state across the whole machine via a
/// self-referential `LookupKind::State` interaction, replacing row-adjacency chaining (the
/// zerocheck framework's constraint-evaluation contexts never expose a "next row"). Every real
/// row unconditionally receives its own incoming state and sends its successor's; a genuine
/// first/last row of a shard has an unmatched receive/send, closed against public values in
/// [`crate::record::eval_public_values`] (`ExecutionRecord::eval_public_values`) instead.
///
/// `incoming_next_pc` is the value this row's predecessor sent as its own outgoing `next_pc`
/// (ordinarily just `pc + 4`, but see `CpuChip::eval_state_chain`'s halt-sentinel handling for
/// the one exception); `outgoing_next_pc`/`outgoing_next_next_pc` are what this row predicts its
/// successor will receive. `clk_low_increment` is how much `clk_low` advances by (normally a
/// small constant, larger for syscalls that consume extra cycles) -- `clk_low + clk_low_increment`
/// is sent as-is, uncorrected, even when it overflows 24 bits: only the `clk_high`-transition
/// chip (not an ordinary opcode chip's row) can receive an overflowed value, decompose it, and
/// send on the corrected `(clk_high + 1, wrapped clk_low)` pair (see [`crate::air::MemoryAirBuilder`]
/// via `send_state`'s doc comment).
#[allow(clippy::too_many_arguments)]
pub fn eval_state_chain<AB: ZKMAirBuilder>(
    builder: &mut AB,
    clk_high: AB::Expr,
    clk_low: AB::Expr,
    pc: AB::Expr,
    incoming_next_pc: AB::Expr,
    outgoing_next_pc: AB::Expr,
    outgoing_next_next_pc: AB::Expr,
    clk_low_increment: AB::Expr,
    is_real: AB::Expr,
) {
    builder.receive_state(clk_high.clone(), clk_low.clone(), pc, incoming_next_pc, is_real.clone());
    builder.send_state(clk_high, clk_low + clk_low_increment, outgoing_next_pc, outgoing_next_next_pc, is_real);
}
