use core::borrow::{Borrow, BorrowMut};
use std::mem::size_of;

use p3_air::{Air, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use zkm_core_executor::{
    events::PrecompileEvent, syscalls::SyscallCode, ExecutionRecord, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{
    air::{AirLookup, LookupScope, MachineAir, ZKMAirBuilder},
    lookup::LookupKind,
};

use crate::{utils::next_power_of_two, CoreChipError};

/// Brackets a `SHA_EXTEND` syscall's 48-iteration worker chain (`ShaExtendChip`): receives the
/// syscall once, then sends the chain's starting `(shard, clk, w_ptr, i = 16)` state and receives
/// its ending `(shard, clk, w_ptr, i = 64)` state.
#[derive(Default)]
pub struct ShaExtendControlChip;

impl ShaExtendControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

pub const NUM_SHA_EXTEND_CONTROL_COLS: usize = size_of::<ShaExtendControlCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaExtendControlCols<T> {
    pub shard: T,
    pub clk: T,
    pub w_ptr: T,
    pub is_real: T,
}

impl<F> BaseAir<F> for ShaExtendControlChip {
    fn width(&self) -> usize {
        NUM_SHA_EXTEND_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ShaExtendControlChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShaExtendControl".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = input.get_precompile_events(SyscallCode::SHA_EXTEND);

        let mut rows = events
            .iter()
            .map(|(_, event)| {
                let event = if let PrecompileEvent::ShaExtend(event) = event {
                    event
                } else {
                    unreachable!()
                };
                let mut row = [F::ZERO; NUM_SHA_EXTEND_CONTROL_COLS];
                let cols: &mut ShaExtendControlCols<F> = row.as_mut_slice().borrow_mut();
                cols.shard = F::from_canonical_u32(event.shard);
                cols.clk = F::from_canonical_u32(event.clk);
                cols.w_ptr = F::from_canonical_u32(event.w_ptr);
                cols.is_real = F::ONE;
                row
            })
            .collect::<Vec<_>>();

        let nb_rows = rows.len();
        let size_log2 = input.fixed_log2_rows::<F, Self>(self);
        let padded_nb_rows =
            next_power_of_two(nb_rows, size_log2, <Self as MachineAir<F>>::name(self).as_str());
        rows.resize(padded_nb_rows, [F::ZERO; NUM_SHA_EXTEND_CONTROL_COLS]);

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_EXTEND_CONTROL_COLS,
        ))
    }

    fn generate_dependencies(&self, _input: &Self::Record, _output: &mut Self::Record) -> Result<(), Self::Error> {
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SHA_EXTEND).is_empty()
        }
    }
}

impl<AB> Air<AB> for ShaExtendControlChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaExtendControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        builder.receive_syscall(
            local.shard,
            local.clk,
            AB::F::from_canonical_u32(SyscallCode::SHA_EXTEND.syscall_id()),
            local.w_ptr,
            AB::Expr::zero(),
            local.is_real,
            LookupScope::Local,
        );

        // Send the chain's starting state (i = 16), for the worker chip's first row to receive.
        builder.send(
            AirLookup::new(
                vec![
                    local.shard.into(),
                    local.clk.into(),
                    local.w_ptr.into(),
                    AB::Expr::from_canonical_u32(16),
                ],
                local.is_real.into(),
                LookupKind::ShaExtend,
            ),
            LookupScope::Local,
        );
        // Receive the chain's ending state (i = 64), sent by the worker chip's last row.
        builder.receive(
            AirLookup::new(
                vec![
                    local.shard.into(),
                    local.clk.into(),
                    local.w_ptr.into(),
                    AB::Expr::from_canonical_u32(64),
                ],
                local.is_real.into(),
                LookupKind::ShaExtend,
            ),
            LookupScope::Local,
        );
    }
}
