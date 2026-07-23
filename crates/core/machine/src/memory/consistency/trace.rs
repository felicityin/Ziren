use p3_field::PrimeField32;
use zkm_core_executor::events::{
    ByteRecord, MemoryReadRecord, MemoryRecord, MemoryRecordEnum, MemoryWriteRecord,
};

use super::{
    MemoryAccessCols, MemoryReadCols, MemoryReadWriteCols, MemoryWriteCols, RegisterAccessCols,
    RegisterAccessTimestamp, RegisterWriteAccessCols,
};

impl<F: PrimeField32> MemoryWriteCols<F> {
    pub fn populate(&mut self, record: MemoryWriteRecord, output: &mut impl ByteRecord) {
        let current_record = MemoryRecord { value: record.value, timestamp: record.timestamp };
        let prev_record =
            MemoryRecord { value: record.prev_value, timestamp: record.prev_timestamp };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, output);
    }
}

impl<F: PrimeField32> MemoryReadCols<F> {
    pub fn populate(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        let current_record = MemoryRecord { value: record.value, timestamp: record.timestamp };
        let prev_record = MemoryRecord { value: record.value, timestamp: record.prev_timestamp };
        self.access.populate_access(current_record, prev_record, output);
    }
}

impl<F: PrimeField32> MemoryReadWriteCols<F> {
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        match record {
            MemoryRecordEnum::Read(read_record) => self.populate_read(read_record, output),
            MemoryRecordEnum::Write(write_record) => self.populate_write(write_record, output),
        }
    }

    pub fn populate_write(&mut self, record: MemoryWriteRecord, output: &mut impl ByteRecord) {
        let current_record = MemoryRecord { value: record.value, timestamp: record.timestamp };
        let prev_record =
            MemoryRecord { value: record.prev_value, timestamp: record.prev_timestamp };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, output);
    }

    pub fn populate_read(&mut self, record: MemoryReadRecord, output: &mut impl ByteRecord) {
        let current_record = MemoryRecord { value: record.value, timestamp: record.timestamp };
        let prev_record = MemoryRecord { value: record.value, timestamp: record.prev_timestamp };
        self.prev_value = prev_record.value.into();
        self.access.populate_access(current_record, prev_record, output);
    }
}

impl<F: PrimeField32> MemoryAccessCols<F> {
    pub(crate) fn populate_access(
        &mut self,
        current_record: MemoryRecord,
        prev_record: MemoryRecord,
        output: &mut impl ByteRecord,
    ) {
        self.value = current_record.value.into();

        // Match the byte range checks emitted by `eval_memory_access` for both the previous and
        // current memory words.
        output.add_u8_range_checks(&prev_record.value.to_le_bytes());
        output.add_u8_range_checks(&current_record.value.to_le_bytes());

        let prev_high = prev_record.timestamp >> 24;
        let prev_low = prev_record.timestamp & 0xffffff;
        let current_high = current_record.timestamp >> 24;
        let current_low = current_record.timestamp & 0xffffff;

        self.prev_high = F::from_canonical_u64(prev_high);
        self.prev_low = F::from_canonical_u64(prev_low);

        // Fill columns used for verifying current memory access time value is greater than
        // previous's.
        let use_low_comparison = prev_high == current_high;
        self.compare_low = F::from_bool(use_low_comparison);
        let prev_time_value = if use_low_comparison { prev_low } else { prev_high };
        let current_time_value = if use_low_comparison { current_low } else { current_high };

        let diff_minus_one = (current_time_value - prev_time_value).wrapping_sub(1);
        let diff_16bit_limb = (diff_minus_one & 0xffff) as u16;
        self.diff_16bit_limb = F::from_canonical_u16(diff_16bit_limb);
        let diff_8bit_limb = (diff_minus_one >> 16) & 0xff;
        self.diff_8bit_limb = F::from_canonical_u64(diff_8bit_limb);

        // Add a byte table lookup with the 16Range op.
        output.add_u16_range_check(diff_16bit_limb);

        // Add a byte table lookup with the U8Range op.
        output.add_u8_range_check(0, diff_8bit_limb as u8);
    }
}

impl<F: PrimeField32> RegisterAccessCols<F> {
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        let prev_record = record.previous_record();
        let current_record = record.current_record();
        self.prev_value = prev_record.value.into();
        self.access_timestamp.populate(prev_record.timestamp, current_record.timestamp, output);

        // Match the byte range checks emitted by `eval_register_access_read`'s defense-in-depth
        // check (a read's `value` is the same as `prev_value`, so only one is needed).
        output.add_u8_range_checks(&prev_record.value.to_le_bytes());
    }
}

impl<F: PrimeField32> RegisterWriteAccessCols<F> {
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        let prev_record = record.previous_record();
        let current_record = record.current_record();
        self.prev_value = prev_record.value.into();
        self.value = current_record.value.into();

        // Match the byte range checks emitted by `eval_register_access_write`'s defense-in-depth
        // check, for both the previous and new values.
        output.add_u8_range_checks(&prev_record.value.to_le_bytes());
        output.add_u8_range_checks(&current_record.value.to_le_bytes());
        self.access_timestamp.populate(prev_record.timestamp, current_record.timestamp, output);
    }
}

impl<F: PrimeField32> RegisterAccessTimestamp<F> {
    /// Populates the timestamp-consistency columns for a register access, given the previous and
    /// current access's timestamps.
    ///
    /// When this access is itself the one crossing a `clk_high` boundary (i.e. the executor also
    /// emitted a `MemoryBumpChip` event for it, re-stamping the register to `(current_high, 0)`
    /// just before this access, per `Executor::emit_memory_bump_events`), `prev_low` must be
    /// populated as `0` -- not the true previous low limb -- to match what that bump event
    /// established as the register's "current" state on the bus. Using the true previous low
    /// limb here would silently reference a state nothing else ever produced.
    pub fn populate(&mut self, prev_timestamp: u64, current_timestamp: u64, output: &mut impl ByteRecord) {
        let same_epoch = prev_timestamp >> 24 == current_timestamp >> 24;
        let prev_low = if same_epoch { prev_timestamp & 0xffffff } else { 0 };
        let current_low = current_timestamp & 0xffffff;
        self.prev_low = F::from_canonical_u64(prev_low);

        let diff_minus_one = (current_low - prev_low) - 1;
        let diff_low_limb = (diff_minus_one & 0xffff) as u16;
        self.diff_low_limb = F::from_canonical_u16(diff_low_limb);
        let diff_high_limb = (diff_minus_one >> 16) & 0xff;

        // Add byte table lookups matching what `eval_register_access_timestamp` sends: the
        // stored `diff_low_limb` is range-checked directly, and the *derived* `diff_high_limb`
        // (recomputed in-circuit via a field-inverse, not stored) is range-checked too.
        output.add_u16_range_check(diff_low_limb);
        output.add_u8_range_check(0, diff_high_limb as u8);
    }
}
