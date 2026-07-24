use columns::NUM_MEMORY_INSTRUCTIONS_COLUMNS;
use p3_air::BaseAir;

pub mod air;
pub mod columns;
pub mod load_word;
pub mod load_x0;
pub mod store_word;
pub mod trace;

pub use load_word::LoadWordChip;
pub use load_x0::LoadX0Chip;
pub use store_word::StoreWordChip;

#[derive(Default)]
pub struct MemoryInstructionsChip;

impl<F> BaseAir<F> for MemoryInstructionsChip {
    fn width(&self) -> usize {
        NUM_MEMORY_INSTRUCTIONS_COLUMNS
    }
}
