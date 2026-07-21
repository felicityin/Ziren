use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::ZKMAirBuilder;

use crate::air::MemoryAirBuilder;

/// Shard number and clk (as a `clk_high` limb above a 16+8+4-bit low window), the bookkeeping
/// every chip that executes a MIPS instruction needs: range-checked, and (for `shard`)
/// cross-checked against the shard's own public value. Lifted out of `CpuChip` so any chip can
/// embed it.
///
/// `clk` is a single monotonic value that never resets across shard boundaries (every
/// `MemoryRecord` is ordered by it); `clk_high` (bits above the low 28-bit window,
/// `CORE_SHARD_CLK_LIMIT`) is constant for every row of a given shard, since
/// `SplicingVM::should_cut_shard` never lets a shard's clk range cross a window boundary -- so
/// `clk_high`, like `shard`, needs only a range check here, no cross-row transition logic.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CpuState<T: Copy> {
    pub shard: T,
    pub clk_high: T,
    pub clk_16bit_limb: T,
    pub clk_8bit_limb: T,
    pub clk_4bit_limb: T,
}

impl<F: PrimeField> CpuState<F> {
    pub fn populate(&mut self, blu: &mut impl ByteRecord, shard: u32, clk: u64) {
        self.shard = F::from_canonical_u32(shard);

        let clk_high = (clk >> 28) as u32;
        let clk_low = (clk & 0xfff_ffff) as u32;
        let clk_16bit_limb = (clk_low & 0xffff) as u16;
        let clk_8bit_limb = ((clk_low >> 16) & 0xff) as u8;
        let clk_4bit_limb = ((clk_low >> 24) & 0xf) as u8;
        self.clk_high = F::from_canonical_u32(clk_high);
        self.clk_16bit_limb = F::from_canonical_u16(clk_16bit_limb);
        self.clk_8bit_limb = F::from_canonical_u8(clk_8bit_limb);
        self.clk_4bit_limb = F::from_canonical_u8(clk_4bit_limb);

        blu.add_byte_lookup_event(ByteLookupEvent::new(ByteOpcode::U16Range, shard as u16, 0, 0, 0));
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U16Range,
            clk_high as u16,
            0,
            0,
            0,
        ));
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
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::U8Range,
            0,
            0,
            0,
            clk_4bit_limb,
        ));
        blu.add_byte_lookup_event(ByteLookupEvent::new(
            ByteOpcode::LTU,
            1,
            0,
            clk_4bit_limb,
            16,
        ));
    }
}

/// Reassembles the low 28 bits of the clk value from its three limbs.
pub fn clk_low_expr<AB: ZKMAirBuilder>(state: &CpuState<AB::Var>) -> AB::Expr {
    AB::Expr::from_canonical_u32(1u32 << 16) * state.clk_8bit_limb
        + AB::Expr::from_canonical_u32(1u32 << 24) * state.clk_4bit_limb
        + state.clk_16bit_limb
}

/// The clk value's `clk_high` limb (bits above the low 28-bit window).
pub fn clk_high_expr<AB: ZKMAirBuilder>(state: &CpuState<AB::Var>) -> AB::Expr {
    state.clk_high.into()
}

/// Range-checks `shard`/`clk_high`/`clk` and cross-checks `shard` against the shard's own public
/// value. `clk` should be [`clk_low_expr`]'s reassembled expression.
pub fn eval_cpu_state<AB: ZKMAirBuilder>(
    builder: &mut AB,
    state: &CpuState<AB::Var>,
    execution_shard: AB::PublicVar,
    clk: AB::Expr,
    is_real: AB::Expr,
) {
    builder.when(is_real.clone()).assert_eq(execution_shard, state.shard);

    builder.send_byte(
        AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
        state.shard,
        AB::Expr::zero(),
        AB::Expr::zero(),
        is_real.clone(),
    );

    builder.send_byte(
        AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
        state.clk_high,
        AB::Expr::zero(),
        AB::Expr::zero(),
        is_real.clone(),
    );

    builder.eval_range_check_28bits(
        clk,
        state.clk_16bit_limb,
        state.clk_8bit_limb,
        state.clk_4bit_limb,
        is_real,
    );
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
/// successor will receive. `clk_increment` is how much the clk advances by (normally a small
/// constant, larger for syscalls that consume extra cycles).
///
/// `clk_high` is constant within a shard (see [`CpuState`]'s doc comment), so the outgoing send
/// reuses the same `clk_high` unchanged -- `clk_low + clk_increment` never needs carry/overflow
/// handling into `clk_high` here: the lookup argument's equality match against the next row's own
/// independently range-checked `clk_high`/`clk_low` is what enforces correctness, and the
/// shard-cut rule guarantees that next row (if any) shares this row's `clk_high`.
#[allow(clippy::too_many_arguments)]
pub fn eval_state_chain<AB: ZKMAirBuilder>(
    builder: &mut AB,
    clk_high: AB::Expr,
    clk: AB::Expr,
    pc: AB::Expr,
    incoming_next_pc: AB::Expr,
    outgoing_next_pc: AB::Expr,
    outgoing_next_next_pc: AB::Expr,
    clk_increment: AB::Expr,
    is_real: AB::Expr,
) {
    builder.receive_state(clk_high.clone(), clk.clone(), pc, incoming_next_pc, is_real.clone());
    builder.send_state(
        clk_high,
        clk + clk_increment,
        outgoing_next_pc,
        outgoing_next_next_pc,
        is_real,
    );
}
