use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemoryReadRecord, MemoryRecordEnum},
    ByteOpcode, ExecutionRecord, Program, NUM_REGISTERS,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::{MachineAir, ZKMAirBuilder};

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    utils::next_multiple_of_32,
    CoreChipError,
};

use super::MemoryReadCols;

pub(crate) const NUM_MEMORY_BUMP_COLS: usize = size_of::<MemoryBumpCols<u8>>();

/// Re-stamps ("bumps") a register's recorded timestamp whenever it would otherwise cross a
/// `clk_high` boundary, using a full-cost general memory-access check (unlike an ordinary
/// register access, which uses the cheap [`crate::memory::RegisterAccessCols`] scheme and can
/// therefore *only* handle a same-`clk_high` comparison). This is what makes it sound for every
/// other register access to assume its previous access shares the same `clk_high` -- the same
/// role [`crate::adapter::StateBumpChip`] plays for the CPU's own global `clk_high`, just applied
/// per-register instead.
///
/// Populated reactively (see `ExecutionRecord::bump_memory_events`): whenever a real register
/// access's own `clk_high` differs from that register's previous access, the executor emits one
/// of these events, re-stamping straight to the current access's `clk_high` regardless of how
/// many epochs elapsed since the register's last touch -- so this alone is sufficient to
/// maintain the invariant, with no need to periodically "refresh" untouched registers.
#[derive(Default)]
pub struct MemoryBumpChip;

impl MemoryBumpChip {
    pub const fn new() -> Self {
        Self
    }
}

/// The column layout for [`MemoryBumpChip`].
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryBumpCols<T: Copy> {
    /// The full-cost memory access being re-stamped.
    pub access: MemoryReadCols<T>,
    /// The 16-bit limb of the outgoing `clk_high`.
    pub clk_high_16bit_limb: T,
    /// The 8-bit limb of the outgoing `clk_high`.
    pub clk_high_8bit_limb: T,
    /// The 16-bit limb of the outgoing `clk_low`.
    pub clk_low_16bit_limb: T,
    /// The 8-bit limb of the outgoing `clk_low`.
    pub clk_low_8bit_limb: T,
    /// Which register (0..[`NUM_REGISTERS`]) this row bumps.
    pub addr: T,
    /// Whether this row is a real bump event.
    pub is_real: T,
}

impl<F> BaseAir<F> for MemoryBumpChip {
    fn width(&self) -> usize {
        NUM_MEMORY_BUMP_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for MemoryBumpChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "MemoryBump".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.bump_memory_events.len(),
            None,
            <MemoryBumpChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let padded_nb_rows = <MemoryBumpChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_MEMORY_BUMP_COLS];

        values[..input.bump_memory_events.len() * NUM_MEMORY_BUMP_COLS]
            .chunks_mut(NUM_MEMORY_BUMP_COLS)
            .zip(input.bump_memory_events.iter())
            .for_each(|(row, event)| {
                let cols: &mut MemoryBumpCols<F> = row.borrow_mut();
                let mut blu = Vec::new();
                self.event_to_row(event, cols, &mut blu);
            });

        Ok(RowMajorMatrix::new(values, NUM_MEMORY_BUMP_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.bump_memory_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .bump_memory_events
            .chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_MEMORY_BUMP_COLS];
                    let cols: &mut MemoryBumpCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.bump_memory_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl MemoryBumpChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &(MemoryRecordEnum, u32),
        cols: &mut MemoryBumpCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        let (record, addr) = *event;

        // The value never changes across a bump -- only the timestamp is re-stamped -- so this
        // is always a synthetic *read* of whatever the value was immediately before the access
        // being bumped (for a real Write access, that's `prev_value`; for a Read, `value`).
        let value = record.previous_record().value;
        let prev_timestamp = record.previous_record().timestamp;
        let timestamp = record.current_record().timestamp;
        cols.access.populate(MemoryReadRecord::new(value, timestamp, prev_timestamp), blu);
        let clk_high = timestamp >> 24;
        let clk_low = timestamp & 0xffffff;

        cols.clk_high_16bit_limb = F::from_canonical_u64(clk_high & 0xffff);
        cols.clk_high_8bit_limb = F::from_canonical_u64((clk_high >> 16) & 0xff);
        cols.clk_low_16bit_limb = F::from_canonical_u64(clk_low & 0xffff);
        cols.clk_low_8bit_limb = F::from_canonical_u64((clk_low >> 16) & 0xff);
        cols.addr = F::from_canonical_u32(addr);
        cols.is_real = F::ONE;

        blu.add_u16_range_checks(&[(clk_high & 0xffff) as u16, (clk_low & 0xffff) as u16]);
        blu.add_u8_range_checks(&[((clk_high >> 16) & 0xff) as u8, ((clk_low >> 16) & 0xff) as u8]);
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::LTU,
            a1: 1,
            a2: 0,
            b: addr as u8,
            c: NUM_REGISTERS as u8,
        });
    }
}

impl<AB> Air<AB> for MemoryBumpChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryBumpCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // Range check the outgoing clk_high/clk_low limbs.
        builder.slice_range_check_u16(
            &[local.clk_high_16bit_limb, local.clk_low_16bit_limb],
            local.is_real,
        );
        builder.slice_range_check_u8(
            &[local.clk_high_8bit_limb, local.clk_low_8bit_limb],
            local.is_real,
        );

        let clk_high = local.clk_high_16bit_limb.into()
            + local.clk_high_8bit_limb.into() * AB::Expr::from_canonical_u32(1 << 16);
        let clk_low = local.clk_low_16bit_limb.into()
            + local.clk_low_8bit_limb.into() * AB::Expr::from_canonical_u32(1 << 16);

        // Check that `addr` is a valid register address.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            local.addr,
            AB::Expr::from_canonical_u32(NUM_REGISTERS as u32),
            local.is_real,
        );

        // Bump the register's timestamp by doing a full-cost re-read of it.
        builder.eval_memory_access(clk_high, clk_low, local.addr, &local.access, local.is_real);
    }
}
