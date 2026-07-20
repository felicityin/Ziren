mod air;
mod columns;
mod trace;

pub use columns::*;
use p3_air::BaseAir;

/// A chip that implements the MIPS jump instructions (J/JR/JAL/JALR).
///
/// Unlike `AddChip`/`MulChip`, no other chip ever emits a synthetic dependency row into
/// `jump_events` -- Jump is purely a dependency *producer* (only `JumpDirect`/JAL sends a
/// synthetic ADD row to `AddChip` to verify the pc-relative target, see the `send_alu` call
/// in `air.rs`). Every row here is therefore a real, retired instruction.
#[derive(Default)]
pub struct JumpChip;

impl<F> BaseAir<F> for JumpChip {
    fn width(&self) -> usize {
        NUM_JUMP_COLS
    }
}
