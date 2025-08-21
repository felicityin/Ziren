#include "kb31_t.hpp"
#include "sys.hpp"
#include "cuda.h"

namespace zkm_core_machine_sys {
__global__ void add_sub_event_to_row(
    const AluEvent* event,
    AddSubCols<kb31_t>* cols,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        add_sub::event_to_row<kb31_t>(event[idx], cols[idx]);
    }
}

extern "C" const char* add_sub_events_to_rows(
    const AluEvent* event,
    AddSubCols<KoalaBearP3>* cols,
    size_t n
) {
    AddSubCols<kb31_t>* cols_kb31 = reinterpret_cast<AddSubCols<kb31_t>*>(cols);

    return launchKernel(add_sub_event_to_row, n, 0, event, cols_kb31, n);
}
} // namespace zkm_core_machine_sys
