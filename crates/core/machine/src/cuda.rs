use p3_koala_bear::KoalaBear;
use cust::memory::DevicePointer;
use zkm_core_executor::events::AluEvent;

use crate::alu::AddSubCols;

extern "C" {
    pub fn add_sub_events_to_rows(
        events: DevicePointer<u8>,
        rows: DevicePointer<u8>,
        n: usize,
    );
}
