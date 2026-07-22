use core::borrow::{Borrow, BorrowMut};
use std::mem::size_of;

use p3_air::{Air, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    ByteOpcode, ExecutionRecord, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{
    air::{AirLookup, LookupScope, MachineAir, ZKMAirBuilder},
    lookup::LookupKind,
};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, CpuState},
    utils::next_power_of_two,
    CoreChipError,
};

/// Brackets a `SHA_EXTEND` syscall's 48-iteration worker chain (`ShaExtendChip`): receives the
/// syscall once, then sends the chain's starting `(clk_high, clk_low, w_ptr, i = 16)` state and
/// receives its ending `(clk_high, clk_low, w_ptr, i = 64)` state.
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
pub struct ShaExtendControlCols<T: Copy> {
    pub state: CpuState<T>,
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
        output: &mut ExecutionRecord,
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
                cols.state.populate(output, event.clk);
                cols.w_ptr = F::from_canonical_u32(event.w_ptr);
                cols.is_real = F::ONE;
                row
            })
            .collect::<Vec<_>>();

        let nb_rows = rows.len();
        let size_log2 = None;
        let padded_nb_rows =
            next_power_of_two(nb_rows, size_log2, <Self as MachineAir<F>>::name(self).as_str());
        rows.resize(padded_nb_rows, [F::ZERO; NUM_SHA_EXTEND_CONTROL_COLS]);

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_EXTEND_CONTROL_COLS,
        ))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        // `generate_trace`'s own byte-lookup registration (via `cols.state.populate(output,
        // ..)`) is a no-op in production: the trace-generation pass is always called with a
        // throwaway `output` (see `crates/hypercube/src/prover/trace.rs`), so this chip's
        // `eval_cpu_state`-driven `clk_low` range check (2 byte lookups per real row) must be
        // registered here instead, mirroring every other migrated opcode chip.
        let events = input.get_precompile_events(SyscallCode::SHA_EXTEND);
        let mut blu: Vec<ByteLookupEvent> = Vec::new();
        for (_, event) in events {
            let event = if let PrecompileEvent::ShaExtend(event) = event {
                event
            } else {
                unreachable!()
            };
            let clk_16bit_limb = (event.clk & 0xffff) as u16;
            let clk_8bit_limb = ((event.clk >> 16) & 0xff) as u8;
            blu.push(ByteLookupEvent::new(ByteOpcode::U16Range, clk_16bit_limb, 0, 0, 0));
            blu.push(ByteLookupEvent::new(ByteOpcode::U8Range, 0, 0, 0, clk_8bit_limb));
        }
        output.add_byte_lookup_events(blu);
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.get_precompile_events(SyscallCode::SHA_EXTEND).is_empty()
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

        let clk_high = local.state.clk_high;
        let clk_low = clk_low_expr::<AB>(&local.state);
        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

        builder.receive_syscall(
            clk_high,
            clk_low.clone(),
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
                    clk_high.into(),
                    clk_low.clone(),
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
                    clk_high.into(),
                    clk_low,
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
