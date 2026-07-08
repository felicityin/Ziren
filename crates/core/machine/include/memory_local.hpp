#pragma once

#include "prelude.hpp"
#include "utils.hpp"
#include "kb31_septic_extension_t.hpp"

namespace zkm_core_machine_sys::memory_local {
    template<class F, class EF7>
    __ZKM_HOSTDEV__ void event_to_row(const MemoryLocalEvent* event, SingleMemoryLocal<F>* cols) {
        cols->addr = F::from_canonical_u32(event->addr);

        // Bit decomposition of `addr` for the KoalaBear field-element range check.
        for (uintptr_t i = 0; i < 32; i++) {
            cols->addr_bits.bits[i] = F::from_canonical_u32(((event->addr) >> i) & 1);
        }
        cols->addr_bits.and_most_sig_byte_decomp_0_to_2 =
            cols->addr_bits.bits[24] * cols->addr_bits.bits[25];
        cols->addr_bits.and_most_sig_byte_decomp_0_to_3 =
            cols->addr_bits.and_most_sig_byte_decomp_0_to_2 * cols->addr_bits.bits[26];
        cols->addr_bits.and_most_sig_byte_decomp_0_to_4 =
            cols->addr_bits.and_most_sig_byte_decomp_0_to_3 * cols->addr_bits.bits[27];
        cols->addr_bits.and_most_sig_byte_decomp_0_to_5 =
            cols->addr_bits.and_most_sig_byte_decomp_0_to_4 * cols->addr_bits.bits[28];
        cols->addr_bits.and_most_sig_byte_decomp_0_to_6 =
            cols->addr_bits.and_most_sig_byte_decomp_0_to_5 * cols->addr_bits.bits[29];
        cols->addr_bits.and_most_sig_byte_decomp_0_to_7 =
            cols->addr_bits.and_most_sig_byte_decomp_0_to_6 * cols->addr_bits.bits[30];

        cols->initial_shard = F::from_canonical_u32(event->initial_mem_access.shard);
        cols->initial_clk = F::from_canonical_u32(event->initial_mem_access.timestamp);
        write_word_from_u32_v2<F>(cols->initial_value, event->initial_mem_access.value);

        cols->final_shard = F::from_canonical_u32(event->final_mem_access.shard);
        cols->final_clk = F::from_canonical_u32(event->final_mem_access.timestamp);
        write_word_from_u32_v2<F>(cols->final_value, event->final_mem_access.value);

        // 16-bit limbs for shards and 16-bit + 8-bit limbs for the 24-bit clk range checks.
        cols->initial_shard_16bit_limb = F::from_canonical_u32(event->initial_mem_access.shard & 0xffff);
        cols->final_shard_16bit_limb = F::from_canonical_u32(event->final_mem_access.shard & 0xffff);
        cols->initial_clk_16bit_limb = F::from_canonical_u32(event->initial_mem_access.timestamp & 0xffff);
        cols->initial_clk_8bit_limb = F::from_canonical_u32((event->initial_mem_access.timestamp >> 16) & 0xff);
        cols->final_clk_16bit_limb = F::from_canonical_u32(event->final_mem_access.timestamp & 0xffff);
        cols->final_clk_8bit_limb = F::from_canonical_u32((event->final_mem_access.timestamp >> 16) & 0xff);

        cols->is_real = F::one();
    }
}  // namespace zkm::memory_local
