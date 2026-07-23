use std::iter::once;

use p3_field::{Field, FieldAlgebra};
use slop_air::AirBuilder;
use zkm_core_executor::ByteOpcode;
use zkm_hypercube::{
    air::{AirLookup, BaseAirBuilder, ByteAirBuilder, LookupScope, OperationSummaryAirBuilder},
    lookup::LookupKind,
};

use crate::{
    air::WordAirBuilder,
    memory::{
        MemoryAccessCols, MemoryCols, RegisterAccessCols, RegisterAccessTimestamp,
        RegisterWriteAccessCols,
    },
};

pub trait MemoryAirBuilder: BaseAirBuilder {
    /// Constrain a memory read or write.
    ///
    /// This method verifies that a memory access timestamp `(clk_high, clk_low)` is greater than
    /// the previous access's timestamp. It will also add to the memory argument.
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
        let prev_low = mem_access.prev_low.clone().into();
        let prev_values = once(prev_high)
            .chain(once(prev_low))
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
    /// Specifically, if the current and previous access have the same `clk_high`, it ensures the
    /// current's `clk_low` is greater than the previous's. Otherwise, it ensures the current's
    /// `clk_high` is greater than the previous's.
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
        let compare_low: Self::Expr = mem_access.compare_low.clone().into();
        let clk_high: Self::Expr = clk_high.clone().into();
        let prev_high: Self::Expr = mem_access.prev_high.clone().into();
        let prev_low: Self::Expr = mem_access.prev_low.clone().into();
        let clk_low: Self::Expr = clk_low.into();
        let diff_16bit_limb: Self::Expr = mem_access.diff_16bit_limb.clone().into();
        let diff_8bit_limb: Self::Expr = mem_access.diff_8bit_limb.clone().into();

        if self.try_emit_memory_timestamp_summary(
            do_check.clone(),
            clk_high.clone(),
            clk_low.clone(),
            prev_high.clone(),
            prev_low.clone(),
            compare_low.clone(),
            diff_16bit_limb.clone(),
            diff_8bit_limb.clone(),
        ) {
            return;
        }

        // First verify that compare_low's value is correct.
        self.when(do_check.clone()).assert_bool(compare_low.clone());
        self.when(do_check.clone()).when(compare_low.clone()).assert_eq(clk_high.clone(), prev_high);

        // Get the comparison timestamp values for the current and previous memory access.
        let prev_comp_value =
            self.if_else(mem_access.compare_low.clone(), prev_low, mem_access.prev_high.clone());

        let current_comp_val = self.if_else(compare_low.clone(), clk_low, clk_high);

        // Assert `current_comp_val > prev_comp_val`. We check this by asserting that
        // `0 <= current_comp_val-prev_comp_val-1 < 2^24`.
        //
        // The equivalence of these statements comes from the fact that if
        // `current_comp_val <= prev_comp_val`, then `current_comp_val-prev_comp_val-1 < 0` and will
        // underflow in the prime field, resulting in a value that is `>= 2^24` as long as both
        // `current_comp_val, prev_comp_val` are range-checked to be `<2^24` (`clk_low` via
        // `eval_cpu_state`'s `eval_range_check_24bits`; `clk_high` via the `clk_high`-transition
        // chip's own range checks -- see `CpuState`'s doc comment) and as long as we're working in
        // a field larger than `2 * 2^24` (which is true of the KoalaBear and Mersenne31 prime).
        let diff_minus_one = current_comp_val - prev_comp_value - Self::Expr::one();

        // Verify that mem_access.ts_diff = mem_access.ts_diff_16bit_limb
        // + mem_access.ts_diff_8bit_limb * 2^16.
        self.eval_range_check_24bits(diff_minus_one, diff_16bit_limb, diff_8bit_limb, do_check);
    }

    /// Verifies the inputted value is within 24 bits.
    ///
    /// This method verifies that the inputted is less than 2^24 by doing a 16 bit and 8 bit range
    /// check on it's limbs.  It will also verify that the limbs are correct.  This method is needed
    /// since the memory access timestamp check (see [Self::eval_memory_access_timestamp]) needs to assume
    /// the clk is within 24 bits.
    fn eval_range_check_24bits(
        &mut self,
        value: impl Into<Self::Expr>,
        limb_16: impl Into<Self::Expr> + Clone,
        limb_8: impl Into<Self::Expr> + Clone,
        do_check: impl Into<Self::Expr> + Clone,
    ) {
        // Verify that value = limb_16 + limb_8 * 2^16.
        self.when(do_check.clone()).assert_eq(
            value,
            limb_16.clone().into()
                + limb_8.clone().into() * Self::Expr::from_canonical_u32(1 << 16),
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
            do_check,
        )
    }

    /// Constrain a register read, using the read value as the write value.
    ///
    /// Uses [`RegisterAccessCols`] (the cheap register-only timestamp scheme) in place of the
    /// general [`MemoryAccessCols`] used by [`Self::eval_memory_access`] -- this assumes (and
    /// only remains sound because of) the invariant that a register's previous access always
    /// shares this access's `clk_high`, maintained by [`crate::memory::MemoryBumpChip`].
    fn eval_register_access_read<E: Into<Self::Expr> + Clone>(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
        addr: impl Into<Self::Expr>,
        reg_access: &RegisterAccessCols<E>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let clk_high: Self::Expr = clk_high.into();
        let clk_low: Self::Expr = clk_low.into();

        self.assert_bool(do_check.clone());
        self.eval_register_access_timestamp(
            &reg_access.access_timestamp,
            do_check.clone(),
            clk_low.clone(),
        );

        // Defense-in-depth: register words entering the subsystem must remain byte-shaped even
        // if an upstream chip forgot to range check them.
        self.slice_range_check_u8(&reg_access.prev_value.clone().map(Into::into).0, do_check.clone());

        let addr = addr.into();
        let prev_low = reg_access.access_timestamp.prev_low.clone().into();
        let prev_values = once(clk_high.clone())
            .chain(once(prev_low))
            .chain(once(addr.clone()))
            .chain(reg_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(clk_high)
            .chain(once(clk_low))
            .chain(once(addr))
            .chain(reg_access.prev_value.clone().map(Into::into))
            .collect();

        self.send(
            AirLookup::new(prev_values, do_check.clone(), LookupKind::Memory),
            LookupScope::Local,
        );
        self.receive(
            AirLookup::new(current_values, do_check, LookupKind::Memory),
            LookupScope::Local,
        );
    }

    /// Constrain a register write, using [`RegisterWriteAccessCols`]'s witnessed `value` as the
    /// write value (see its doc comment for why this can't instead take an arbitrary caller
    /// expression the way [`Self::eval_memory_access_write`] does: the interaction's value must
    /// stay affine, but the natural "value" a caller wants to write -- e.g. an ALU result masked
    /// for "writes to register 0 are discarded" -- is generally degree 2 or higher). The caller
    /// is responsible for separately asserting `reg_access.value` equals the intended result.
    ///
    /// See [`Self::eval_register_access_read`]'s doc comment for the `clk_high` invariant this
    /// depends on.
    fn eval_register_access_write<E: Into<Self::Expr> + Clone>(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
        addr: impl Into<Self::Expr>,
        reg_access: &RegisterWriteAccessCols<E>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let clk_high: Self::Expr = clk_high.into();
        let clk_low: Self::Expr = clk_low.into();

        self.assert_bool(do_check.clone());
        self.eval_register_access_timestamp(
            &reg_access.access_timestamp,
            do_check.clone(),
            clk_low.clone(),
        );

        self.slice_range_check_u8(&reg_access.prev_value.clone().map(Into::into).0, do_check.clone());
        self.slice_range_check_u8(&reg_access.value.clone().map(Into::into).0, do_check.clone());

        let addr = addr.into();
        let prev_low = reg_access.access_timestamp.prev_low.clone().into();
        let prev_values = once(clk_high.clone())
            .chain(once(prev_low))
            .chain(once(addr.clone()))
            .chain(reg_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(clk_high)
            .chain(once(clk_low))
            .chain(once(addr))
            .chain(reg_access.value.clone().map(Into::into))
            .collect();

        self.send(
            AirLookup::new(prev_values, do_check.clone(), LookupKind::Memory),
            LookupScope::Local,
        );
        self.receive(
            AirLookup::new(current_values, do_check, LookupKind::Memory),
            LookupScope::Local,
        );
    }

    /// Verifies a register access's timestamp is later than its previous access's, using only a
    /// low-limb comparison -- sound only because a register's previous access is guaranteed to
    /// share this access's `clk_high` (see [`Self::eval_register_access_read`]'s doc comment).
    /// The high limb of the difference is *derived* here via a field-inverse, rather than stored
    /// as its own column, saving one more field over the already-cheap 2-column
    /// [`RegisterAccessTimestamp`].
    fn eval_register_access_timestamp(
        &mut self,
        reg_access: &RegisterAccessTimestamp<impl Into<Self::Expr> + Clone>,
        do_check: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let diff_minus_one =
            clk_low.into() - reg_access.prev_low.clone().into() - Self::Expr::one();
        let diff_high_limb = (diff_minus_one - reg_access.diff_low_limb.clone().into())
            * Self::F::from_canonical_u32(1 << 16).inverse();

        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            reg_access.diff_low_limb.clone(),
            Self::Expr::zero(),
            Self::Expr::zero(),
            do_check.clone(),
        );
        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            Self::Expr::zero(),
            Self::Expr::zero(),
            diff_high_limb,
            do_check,
        )
    }
}
