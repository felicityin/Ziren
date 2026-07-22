use core::borrow::{Borrow, BorrowMut};
use std::mem::size_of;

use p3_air::{Air, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use zkm_core_executor::{events::PrecompileEvent, syscalls::SyscallCode, ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{
    air::{AirLookup, LookupScope, MachineAir, ZKMAirBuilder},
    lookup::LookupKind,
    word::Word,
};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, CpuState},
    utils::next_power_of_two,
    CoreChipError,
};

/// Brackets a `SHA_COMPRESS` syscall's 80-row worker chain (`ShaCompressChip`): receives the
/// syscall once, then sends the chain's starting `(clk_high, clk_low, w_ptr, h_ptr, index = 0, h)` state
/// (the pre-compression `H` values read from memory) and receives its ending
/// `(index = 80, compressed)` state (the post-compression register values, before the finalize
/// phase adds them back onto `H`).
#[derive(Default)]
pub struct ShaCompressControlChip;

impl ShaCompressControlChip {
    pub const fn new() -> Self {
        Self {}
    }
}

pub const NUM_SHA_COMPRESS_CONTROL_COLS: usize = size_of::<ShaCompressControlCols<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct ShaCompressControlCols<T: Copy> {
    pub state: CpuState<T>,
    pub w_ptr: T,
    pub h_ptr: T,
    pub initial_state: [Word<T>; 8],
    pub compressed_state: [Word<T>; 8],
    pub is_real: T,
}

impl<F> BaseAir<F> for ShaCompressControlChip {
    fn width(&self) -> usize {
        NUM_SHA_COMPRESS_CONTROL_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ShaCompressControlChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "ShaCompressControl".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = input.get_precompile_events(SyscallCode::SHA_COMPRESS);

        let mut rows = events
            .iter()
            .map(|(_, event)| {
                let event = if let PrecompileEvent::ShaCompress(event) = event {
                    event
                } else {
                    unreachable!()
                };
                let mut row = [F::ZERO; NUM_SHA_COMPRESS_CONTROL_COLS];
                let cols: &mut ShaCompressControlCols<F> = row.as_mut_slice().borrow_mut();
                cols.state.populate(output, event.clk);
                cols.w_ptr = F::from_canonical_u32(event.w_ptr);
                cols.h_ptr = F::from_canonical_u32(event.h_ptr);
                for i in 0..8 {
                    cols.initial_state[i] = Word::from(event.h[i]);
                    cols.compressed_state[i] =
                        Word::from(event.h_write_records[i].value.wrapping_sub(event.h[i]));
                }
                cols.is_real = F::ONE;
                row
            })
            .collect::<Vec<_>>();

        let nb_rows = rows.len();
        let size_log2 = None;
        let padded_nb_rows =
            next_power_of_two(nb_rows, size_log2, <Self as MachineAir<F>>::name(self).as_str());
        rows.resize(padded_nb_rows, [F::ZERO; NUM_SHA_COMPRESS_CONTROL_COLS]);

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_SHA_COMPRESS_CONTROL_COLS,
        ))
    }

    fn generate_dependencies(
        &self,
        _input: &Self::Record,
        _output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.get_precompile_events(SyscallCode::SHA_COMPRESS).is_empty()
    }
}

impl<AB> Air<AB> for ShaCompressControlChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ShaCompressControlCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let clk_high = local.state.clk_high;
        let clk_low = clk_low_expr::<AB>(&local.state);
        eval_cpu_state(builder, &local.state, clk_low.clone(), local.is_real.into());

        builder.receive_syscall(
            clk_high,
            clk_low.clone(),
            AB::F::from_canonical_u32(SyscallCode::SHA_COMPRESS.syscall_id()),
            local.w_ptr,
            local.h_ptr,
            local.is_real,
            LookupScope::Local,
        );

        let base = [clk_high.into(), clk_low.clone(), local.w_ptr.into(), local.h_ptr.into()];

        // Send the chain's starting state (index = 0): the pre-compression `H` values. Each
        // field is independently verified against genuine memory reads transitively, by whichever
        // worker row along the chain ends up receiving it (see `ShaCompressChip::eval_memory`).
        let send_values = base
            .iter()
            .cloned()
            .chain(core::iter::once(AB::Expr::zero()))
            .chain(local.initial_state.iter().flat_map(|word| word.0.iter().map(|&e| e.into())))
            .collect::<Vec<_>>();
        builder.send(
            AirLookup::new(send_values, local.is_real.into(), LookupKind::ShaCompress),
            LookupScope::Local,
        );

        // Receive the chain's ending state (index = 80): the post-compression, pre-finalize
        // register values.
        let receive_values = base
            .iter()
            .cloned()
            .chain(core::iter::once(AB::Expr::from_canonical_u32(80)))
            .chain(local.compressed_state.iter().flat_map(|word| word.0.iter().map(|&e| e.into())))
            .collect::<Vec<_>>();
        builder.receive(
            AirLookup::new(receive_values, local.is_real.into(), LookupKind::ShaCompress),
            LookupScope::Local,
        );
    }
}
