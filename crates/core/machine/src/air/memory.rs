use std::iter::once;

use p3_field::FieldAlgebra;
use slop_air::AirBuilder;
use zkm_core_executor::ByteOpcode;
use zkm_hypercube::{
    air::{AirLookup, BaseAirBuilder, ByteAirBuilder, LookupScope, OperationSummaryAirBuilder},
    lookup::LookupKind,
};

use crate::{
    air::WordAirBuilder,
    memory::{MemoryAccessCols, MemoryCols},
};

pub trait MemoryAirBuilder: BaseAirBuilder {
    /// Constrain a memory read or write.
    ///
    /// This method verifies that a memory access timestamp (clk_high, clk_low) is greater than
    /// the previous access's timestamp.  It will also add to the memory argument.
    fn eval_memory_access<E: Into<Self::Expr> + Clone>(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
        addr: impl Into<Self::Expr>,
        memory_access: &impl MemoryCols<E>,
        do_check: impl Into<Self::Expr>,
    ) where
        Self: OperationSummaryAirBuilder,
    {
        let do_check: Self::Expr = do_check.into();
        let clk_high: Self::Expr = clk_high.into();
        let clk_low: Self::Expr = clk_low.into();
        let mem_access = memory_access.access();

        self.assert_bool(do_check.clone());

        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_timestamp(
            mem_access,
            do_check.clone(),
            clk_high.clone(),
            clk_low.clone(),
        );

        // Defense-in-depth: memory words entering the subsystem must remain byte-shaped even
        // if an upstream chip forgot to range check them.
        self.slice_range_check_u8(&memory_access.prev_value().0, do_check.clone());
        self.slice_range_check_u8(&memory_access.value().0, do_check.clone());

        // Add to the memory argument.
        let addr = addr.into();
        let prev_high = mem_access.prev_high.clone().into();
        let prev_clk = mem_access.prev_clk.clone().into();
        let prev_values = once(prev_high)
            .chain(once(prev_clk))
            .chain(once(addr.clone()))
            .chain(memory_access.prev_value().clone().map(Into::into))
            .collect();
        let current_values = once(clk_high)
            .chain(once(clk_low))
            .chain(once(addr.clone()))
            .chain(memory_access.value().clone().map(Into::into))
            .collect();

        // The previous values get sent with multiplicity = 1, for "read".
        self.send(
            AirLookup::new(prev_values, do_check.clone(), LookupKind::Memory),
            LookupScope::Local,
        );

        // The current values get "received", i.e. multiplicity = -1
        self.receive(
            AirLookup::new(current_values, do_check.clone(), LookupKind::Memory),
            LookupScope::Local,
        );
    }

    /// Constraints a memory read or write to a slice of `MemoryAccessCols`.
    fn eval_memory_access_slice<E: Into<Self::Expr> + Copy>(
        &mut self,
        clk_high: impl Into<Self::Expr> + Copy,
        clk_low: impl Into<Self::Expr> + Clone,
        initial_addr: impl Into<Self::Expr> + Clone,
        memory_access_slice: &[impl MemoryCols<E>],
        verify_memory_access: impl Into<Self::Expr> + Copy,
    ) where
        Self: OperationSummaryAirBuilder,
    {
        for (i, access_slice) in memory_access_slice.iter().enumerate() {
            self.eval_memory_access(
                clk_high,
                clk_low.clone(),
                initial_addr.clone().into() + Self::Expr::from_canonical_usize(i * 4),
                access_slice,
                verify_memory_access,
            );
        }
    }

    /// Verifies the memory access timestamp.
    ///
    /// This method verifies that the current memory access happened after the previous one's.
    /// Specifically it will ensure that if the current and previous access have the same
    /// `clk_high`, then the current's `clk_low` val is greater than the previous's.  If they
    /// don't, then it will ensure that the current's `clk_high` val is greater than the
    /// previous's.
    fn eval_memory_access_timestamp(
        &mut self,
        mem_access: &MemoryAccessCols<impl Into<Self::Expr> + Clone>,
        do_check: impl Into<Self::Expr>,
        clk_high: impl Into<Self::Expr> + Clone,
        clk_low: impl Into<Self::Expr>,
    ) where
        Self: OperationSummaryAirBuilder,
    {
        let do_check: Self::Expr = do_check.into();
        let compare_high: Self::Expr = mem_access.compare_high.clone().into();
        let clk_high: Self::Expr = clk_high.clone().into();
        let prev_high: Self::Expr = mem_access.prev_high.clone().into();
        let prev_clk: Self::Expr = mem_access.prev_clk.clone().into();
        let clk_low: Self::Expr = clk_low.into();
        let diff_16bit_limb: Self::Expr = mem_access.diff_16bit_limb.clone().into();
        let diff_8bit_limb: Self::Expr = mem_access.diff_8bit_limb.clone().into();
        let diff_4bit_limb: Self::Expr = mem_access.diff_4bit_limb.clone().into();

        if self.try_emit_memory_timestamp_summary(
            do_check.clone(),
            clk_high.clone(),
            clk_low.clone(),
            prev_high.clone(),
            prev_clk.clone(),
            compare_high.clone(),
            diff_16bit_limb.clone(),
            diff_8bit_limb.clone(),
            diff_4bit_limb.clone(),
        ) {
            return;
        }

        // First verify that compare_high's value is correct.
        self.when(do_check.clone()).assert_bool(compare_high.clone());
        self.when(do_check.clone()).when(compare_high.clone()).assert_eq(clk_high.clone(), prev_high);

        // Get the comparison timestamp values for the current and previous memory access.
        let prev_comp_value =
            self.if_else(mem_access.compare_high.clone(), prev_clk, mem_access.prev_high.clone());

        let current_comp_val = self.if_else(compare_high.clone(), clk_low, clk_high.clone());

        // Assert `current_comp_val > prev_comp_val`. We check this by asserting that
        // `0 <= current_comp_val-prev_comp_val-1 < 2^28`.
        //
        // The equivalence of these statements comes from the fact that if
        // `current_comp_val <= prev_comp_val`, then `current_comp_val-prev_comp_val-1 < 0` and will
        // underflow in the prime field, resulting in a value that is `>= 2^28` as long as both
        // `current_comp_val, prev_comp_val` are range-checked to be `<2^28` and as long as we're
        // working in a field larger than `2 * 2^28` (true of the KoalaBear prime, `2^31 - 2^24 + 1`:
        // `2 * 2^28 = 2^29`, comfortably under it).
        let diff_minus_one = current_comp_val - prev_comp_value - Self::Expr::one();

        // Verify that mem_access.ts_diff = mem_access.ts_diff_16bit_limb
        // + mem_access.ts_diff_8bit_limb * 2^16 + mem_access.ts_diff_4bit_limb * 2^24.
        self.eval_range_check_28bits(
            diff_minus_one,
            diff_16bit_limb,
            diff_8bit_limb,
            diff_4bit_limb,
            do_check,
        );
    }

    /// Verifies the inputted value is within 28 bits.
    ///
    /// This method verifies that the input is less than 2^28 by doing a 16 bit, 8 bit, and 4 bit
    /// range check on its limbs.  It will also verify that the limbs are correct.  This method is
    /// needed since the memory access timestamp check (see [Self::eval_memory_access_timestamp])
    /// needs to assume the clk is within 28 bits.
    fn eval_range_check_28bits(
        &mut self,
        value: impl Into<Self::Expr>,
        limb_16: impl Into<Self::Expr> + Clone,
        limb_8: impl Into<Self::Expr> + Clone,
        limb_4: impl Into<Self::Expr> + Clone,
        do_check: impl Into<Self::Expr> + Clone,
    ) {
        // Verify that value = limb_16 + limb_8 * 2^16 + limb_4 * 2^24.
        self.when(do_check.clone()).assert_eq(
            value,
            limb_16.clone().into()
                + limb_8.clone().into() * Self::Expr::from_canonical_u32(1 << 16)
                + limb_4.clone().into() * Self::Expr::from_canonical_u32(1 << 24),
        );

        // Send the range checks for the limbs.
        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            limb_16,
            Self::Expr::zero(),
            Self::Expr::zero(),
            do_check.clone(),
        );

        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            Self::Expr::zero(),
            Self::Expr::zero(),
            limb_8,
            do_check.clone(),
        );

        // `limb_4` must additionally be < 16 (a real 4-bit value), not just a valid byte.
        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            Self::Expr::zero(),
            Self::Expr::zero(),
            limb_4.clone(),
            do_check.clone(),
        );
        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::LTU as u8),
            Self::Expr::one(),
            limb_4,
            Self::Expr::from_canonical_u8(16),
            do_check,
        )
    }
}
