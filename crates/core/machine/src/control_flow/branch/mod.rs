mod air;
mod columns;
mod trace;

pub use columns::*;
use p3_air::BaseAir;

/// A chip that implements the MIPS branch instructions (BEQ/BNE/BLTZ/BGEZ/BLEZ/BGTZ).
///
/// Unlike `AddSubChip`/`MulChip`, no other chip ever emits a synthetic dependency row into
/// `branch_events` -- Branch is purely a dependency *producer* (it sends synthetic ADD/SLT rows
/// to `AddSubChip`/`LtChip` to verify the branch target and comparison, see the `send_alu` calls
/// in `air.rs`). Every row here is therefore a real, retired instruction.
#[derive(Default)]
pub struct BranchChip;

impl<F> BaseAir<F> for BranchChip {
    fn width(&self) -> usize {
        NUM_BRANCH_COLS
    }
}
