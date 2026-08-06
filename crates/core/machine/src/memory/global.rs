use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use std::iter::once;

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use zkm_core_executor::events::{
    ByteLookupEvent, ByteRecord, GlobalLookupEvent, MemoryInitializeFinalizeEvent,
};
use zkm_core_executor::{ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
use zkm_hypercube::{
    air::{AirLookup, LookupScope, MachineAir, ZKMAirBuilder},
    lookup::LookupKind,
    word::Word,
};

use crate::{
    air::WordAirBuilder,
    operations::{AssertLtColsBytes, IsZeroOperation, KoalaBearWordRangeChecker},
    utils::next_multiple_of_32,
    CoreChipError,
};

use super::MemoryChipType;

/// A memory chip that can initialize or finalize values in memory.
pub struct MemoryGlobalChip {
    pub kind: MemoryChipType,
}

impl MemoryGlobalChip {
    /// Creates a new memory chip with a certain type.
    pub const fn new(kind: MemoryChipType) -> Self {
        Self { kind }
    }
}

impl<F> BaseAir<F> for MemoryGlobalChip {
    fn width(&self) -> usize {
        NUM_MEMORY_INIT_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for MemoryGlobalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        match self.kind {
            MemoryChipType::Initialize => "MemoryGlobalInit".to_string(),
            MemoryChipType::Finalize => "MemoryGlobalFinalize".to_string(),
        }
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> zkm_hypercube::air::PicusInfo {
        MemoryInitCols::<u8>::picus_info()
    }

    fn generate_dependencies(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<(), Self::Error> {
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };

        let is_receive = match self.kind {
            MemoryChipType::Initialize => false,
            MemoryChipType::Finalize => true,
        };

        match self.kind {
            MemoryChipType::Initialize => {
                output.public_values.global_init_count += memory_events.len() as u32;
            }
            MemoryChipType::Finalize => {
                output.public_values.global_finalize_count += memory_events.len() as u32;
            }
        }

        memory_events.sort_by_key(|event| event.addr);

        // `ByteChip::generate_trace` reads its multiplicities from `input.byte_lookups`, which is
        // only populated by `generate_dependencies` output (not `generate_trace`'s, which the real
        // shard-proving driver discards) -- so the byte lookups this chip's `eval` sends must be
        // registered here too, mirroring the sequential address-chain pass in `generate_trace`.
        let previous_addr = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr,
        };
        let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
        for (i, event) in memory_events.iter().enumerate() {
            let prev_addr = if i == 0 { previous_addr } else { memory_events[i - 1].addr };
            blu.add_u8_range_checks(&event.addr.to_le_bytes());
            blu.add_u8_range_checks(&event.value.to_le_bytes());
            blu.add_u8_range_checks(&prev_addr.to_le_bytes());
            KoalaBearWordRangeChecker::<F>::default().populate(&mut blu, event.addr);

            let is_comp = prev_addr != 0 || i != 0 || event.addr != 0;
            if is_comp {
                let mut row = [F::ZERO; NUM_MEMORY_INIT_COLS];
                let cols: &mut MemoryInitCols<F> = row.as_mut_slice().borrow_mut();
                cols.lt_cols.populate(&mut blu, &prev_addr.to_le_bytes(), &event.addr.to_le_bytes());
            }
        }
        output.add_byte_lookup_events_from_maps(vec![&blu]);

        let events = memory_events.into_iter().map(|event| {
            let lookup_clk_high = if is_receive { (event.timestamp >> 24) as u32 } else { 0 };
            let lookup_clk_low = if is_receive { (event.timestamp & 0xffffff) as u32 } else { 0 };
            GlobalLookupEvent {
                message: [
                    lookup_clk_high,
                    lookup_clk_low,
                    event.addr,
                    (event.value & 255) as u32,
                    ((event.value >> 8) & 255) as u32,
                    ((event.value >> 16) & 255) as u32,
                    ((event.value >> 24) & 255) as u32,
                ],
                is_receive,
                kind: LookupKind::Memory as u8,
            }
        });
        output.global_lookup_events.extend(events);
        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = match self.kind {
            MemoryChipType::Initialize => &input.global_memory_initialize_events,
            MemoryChipType::Finalize => &input.global_memory_finalize_events,
        };
        let nb_rows = events.len();
        let size_log2 = None;
        let padded_nb_rows = next_multiple_of_32(
            nb_rows,
            size_log2,
            <MemoryGlobalChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };

        let previous_addr = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr,
        };

        memory_events.sort_by_key(|event| event.addr);
        let mut rows: Vec<[F; NUM_MEMORY_INIT_COLS]> = memory_events
            .par_iter()
            .map(|event| {
                let MemoryInitializeFinalizeEvent { addr, value, timestamp } = event.to_owned();

                let mut row = [F::ZERO; NUM_MEMORY_INIT_COLS];
                let cols: &mut MemoryInitCols<F> = row.as_mut_slice().borrow_mut();
                cols.addr = Word::from(addr);
                cols.clk_high = F::from_canonical_u64(timestamp >> 24);
                cols.clk_low = F::from_canonical_u64(timestamp & 0xffffff);
                cols.value = Word::from(value);
                cols.is_real = F::one();

                row
            })
            .collect::<Vec<_>>();

        // The strictly-increasing address chain is inherently sequential (each row depends on the
        // previous sorted event's address), so this second pass isn't parallelized like the first.
        let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
        for i in 0..memory_events.len() {
            let addr = memory_events[i].addr;
            let value = memory_events[i].value;
            let cols: &mut MemoryInitCols<F> = rows[i].as_mut_slice().borrow_mut();

            let prev_addr = if i == 0 { previous_addr } else { memory_events[i - 1].addr };

            // Matches `slice_range_check_u8(&local.addr.0/&local.value.0/&local.prev_addr.0, ...)`
            // in `eval()` -- each is a raw witnessed column with no other interaction that would
            // already range-check its bytes, so the matching lookup event must be registered here.
            blu.add_u8_range_checks(&addr.to_le_bytes());
            blu.add_u8_range_checks(&value.to_le_bytes());
            blu.add_u8_range_checks(&prev_addr.to_le_bytes());
            cols.addr_range_checker.populate(&mut blu, addr);

            cols.index = F::from_canonical_u32(i as u32);
            cols.prev_addr = Word::from(prev_addr);
            cols.prev_valid = F::from_bool(!(prev_addr == 0 && i != 0));
            let is_prev_addr_zero = cols.is_prev_addr_zero.populate_from_field_element(
                cols.prev_addr.0[0] + cols.prev_addr.0[1] + cols.prev_addr.0[2] + cols.prev_addr.0[3],
            );
            let is_index_zero = cols.is_index_zero.populate(i as u32);
            cols.is_addr_zero.populate_from_field_element(
                cols.addr.0[0] + cols.addr.0[1] + cols.addr.0[2] + cols.addr.0[3],
            );
            cols.is_prev_addr_and_index_zero =
                F::from_bool(is_prev_addr_zero == 1 && is_index_zero == 1);

            let is_comp = prev_addr != 0 || i != 0 || addr != 0;
            cols.is_comp = F::from_bool(is_comp);
            if is_comp {
                debug_assert!(prev_addr < addr, "prev_addr {prev_addr} < addr {addr}");
                cols.lt_cols.populate(&mut blu, &prev_addr.to_le_bytes(), &addr.to_le_bytes());
            }
        }
        output.add_byte_lookup_events_from_maps(vec![&blu]);

        // Pad the trace to a power of two depending on the proof shape in `input`.
        rows.resize(
            <MemoryGlobalChip as MachineAir<F>>::num_rows(self, input).unwrap(),
            [F::zero(); NUM_MEMORY_INIT_COLS],
        );

        Ok(RowMajorMatrix::new(
            rows.into_iter().flatten().collect::<Vec<_>>(),
            NUM_MEMORY_INIT_COLS,
        ))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        match self.kind {
            MemoryChipType::Initialize => !shard.global_memory_initialize_events.is_empty(),
            MemoryChipType::Finalize => !shard.global_memory_finalize_events.is_empty(),
        }
    }

    fn commit_scope(&self) -> LookupScope {
        LookupScope::Local
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct MemoryInitCols<T: Copy> {
    /// The `clk_high` of the memory access.
    #[cfg_attr(feature = "picus", picus(input, transition_input))]
    pub clk_high: T,

    /// The `clk_low` of the memory access.
    #[cfg_attr(feature = "picus", picus(input, transition_input))]
    pub clk_low: T,

    /// The address of the memory access.
    #[cfg_attr(feature = "picus", picus(input, transition_input))]
    pub addr: Word<T>,

    /// Gadget to verify that `addr` is within the Koala-Bear field -- needed since `addr` is used
    /// as a single reduced field element (in the `Global` interaction and the chain below).
    pub addr_range_checker: KoalaBearWordRangeChecker<T>,

    /// Comparison assertions for address to be strictly increasing.
    pub lt_cols: AssertLtColsBytes<T, 4>,

    /// The value of the memory access.
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub value: Word<T>,

    /// Whether the memory access is a real access.
    pub is_real: T,

    /// This row's position in the chip's own sorted-address sequence. Anchors the
    /// `LookupKind::MemoryGlobalInitControl`/`MemoryGlobalFinalizeControl` chain by value
    /// instead of physical row adjacency.
    pub index: T,

    /// This row's own witnessed previous address (matched by value against whichever row -- or
    /// `ExecutionRecord::eval_public_values`, at index 0 -- sent it). No canonical range-check
    /// needed: only ever used for the is-zero check and the byte-wise comparison below.
    pub prev_addr: Word<T>,

    /// The validity of the previous state received at `index`. False only for the unique
    /// %x0-initializes-once case (mirrors `is_comp`, offset by one row).
    pub prev_valid: T,

    /// A witness to assert whether or not the previous address is zero.
    pub is_prev_addr_zero: IsZeroOperation<T>,

    /// A witness to assert whether or not `index` is zero.
    pub is_index_zero: IsZeroOperation<T>,

    /// A witness to assert whether or not `addr` is zero.
    pub is_addr_zero: IsZeroOperation<T>,

    /// `is_prev_addr_zero.result * is_index_zero.result`, witnessed as its own column so that
    /// `is_comp`'s three-way AND stays within `MAX_CONSTRAINT_DEGREE` (multiplying all three
    /// `IsZeroOperation` results together directly would be degree 4).
    pub is_prev_addr_and_index_zero: T,

    /// Whether or not we are making the assertion `prev_addr < addr`. False only when
    /// `prev_addr == 0`, `index == 0`, and `addr == 0`, i.e. this is the sole initialization of
    /// address 0.
    pub is_comp: T,
}

pub(crate) const NUM_MEMORY_INIT_COLS: usize = size_of::<MemoryInitCols<u8>>();

impl<AB> Air<AB> for MemoryGlobalChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryInitCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // `addr`/`value`/`prev_addr` are raw witnessed columns (not derived via any interaction
        // that would already range-check them elsewhere), so each needs its own explicit
        // byte-range-check.
        builder.slice_range_check_u8(&local.addr.0, local.is_real);
        builder.slice_range_check_u8(&local.value.0, local.is_real);
        builder.slice_range_check_u8(&local.prev_addr.0, local.is_real);

        // Canonicalize padded rows to the default zero trace shape so witness columns cannot
        // drift in extraction modules.
        builder.when_not(local.is_real).assert_zero(local.clk_high);
        builder.when_not(local.is_real).assert_zero(local.clk_low);
        builder.when_not(local.is_real).assert_word_zero(local.addr);
        builder.when_not(local.is_real).assert_word_zero(local.value);
        builder.when_not(local.is_real).assert_word_zero(local.prev_addr);
        for i in 0..4 {
            builder.when_not(local.is_real).assert_zero(local.lt_cols.byte_flags[i]);
        }
        builder.when_not(local.is_real).assert_zero(local.addr_range_checker.high16_lt_top_limb);

        if self.kind == MemoryChipType::Initialize {
            // Send the lookup to the global table.
            builder.send(
                AirLookup::new(
                    vec![
                        AB::Expr::zero(), // shard
                        AB::Expr::zero(), // timestamp
                        local.addr.reduce::<AB>(),
                        local.value.0[0].into(),
                        local.value.0[1].into(),
                        local.value.0[2].into(),
                        local.value.0[3].into(),
                        local.is_real.into(),
                        AB::Expr::zero(),
                        AB::Expr::from_canonical_u8(LookupKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    LookupKind::Global,
                ),
                LookupScope::Local,
            );
        } else {
            // Send the lookup to the global table.
            builder.send(
                AirLookup::new(
                    vec![
                        local.clk_high.into(),
                        local.clk_low.into(),
                        local.addr.reduce::<AB>(),
                        local.value.0[0].into(),
                        local.value.0[1].into(),
                        local.value.0[2].into(),
                        local.value.0[3].into(),
                        AB::Expr::zero(),
                        local.is_real.into(),
                        AB::Expr::from_canonical_u8(LookupKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    LookupKind::Global,
                ),
                LookupScope::Local,
            );
        }

        // Canonically range-check the address into the field so it can be safely used as a
        // single reduced element above and in the chain below.
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.addr,
            local.addr_range_checker,
            local.is_real.into(),
        );

        // Chain this row's own witnessed `prev_addr` against whichever row (or
        // `ExecutionRecord::eval_public_values`, at index 0) sent it as its own address, and this
        // row's own `addr` against whichever row (or the phantom receive, at the final index)
        // receives it as the start of the next segment -- replacing the old row-adjacency
        // `addr < addr'` chaining.
        let interaction_kind = match self.kind {
            MemoryChipType::Initialize => LookupKind::MemoryGlobalInitControl,
            MemoryChipType::Finalize => LookupKind::MemoryGlobalFinalizeControl,
        };
        builder.receive(
            AirLookup::new(
                once(local.index.into())
                    .chain(local.prev_addr.0.iter().map(|&b| b.into()))
                    .chain(once(local.prev_valid.into()))
                    .collect(),
                local.is_real.into(),
                interaction_kind,
            ),
            LookupScope::Local,
        );
        builder.send(
            AirLookup::new(
                once(local.index.into() + AB::Expr::one())
                    .chain(local.addr.0.iter().map(|&b| b.into()))
                    .chain(once(local.is_comp.into()))
                    .collect(),
                local.is_real.into(),
                interaction_kind,
            ),
            LookupScope::Local,
        );

        IsZeroOperation::<AB::F>::eval(
            builder,
            local.prev_addr.0[0] + local.prev_addr.0[1] + local.prev_addr.0[2] + local.prev_addr.0[3],
            local.is_prev_addr_zero,
            local.is_real.into(),
        );
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.index.into(),
            local.is_index_zero,
            local.is_real.into(),
        );
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.addr.0[0] + local.addr.0[1] + local.addr.0[2] + local.addr.0[3],
            local.is_addr_zero,
            local.is_real.into(),
        );

        // Witnessed separately (rather than inlined into the `is_comp` formula below) so that
        // `is_comp`'s three-way AND doesn't exceed `MAX_CONSTRAINT_DEGREE`.
        builder.assert_eq(
            local.is_prev_addr_and_index_zero,
            local.is_prev_addr_zero.result.into() * local.is_index_zero.result.into(),
        );

        // `is_comp` is false only when `prev_addr == 0`, `index == 0`, and `addr == 0`, i.e. this
        // is the sole initialization of address 0 -- the one case with no valid comparison to
        // make. (Requiring `addr == 0` too, not just `prev_addr == 0 && index == 0`, matters
        // whenever a shard's memory-init/finalize chain starts fresh -- `prev_addr` is then the
        // public-values sentinel `0` regardless of what the chain's first real address is -- so
        // omitting it would incorrectly mark every such first row as the degenerate case even
        // when its own `addr` is nonzero, desyncing this row's `is_comp` from the next row's
        // `prev_valid` in the `LookupKind::MemoryGlobalInitControl`/`FinalizeControl` chain.)
        builder.assert_eq(
            local.is_comp,
            local.is_real.into()
                * (AB::Expr::one()
                    - local.is_prev_addr_and_index_zero.into() * local.is_addr_zero.result.into()),
        );
        builder.assert_bool(local.is_comp);

        // If `is_comp`, `prev_addr < addr` must hold.
        local.lt_cols.eval(builder, &local.prev_addr.0, &local.addr.0, local.is_comp);

        // Make assertions for specific types of memory chips.
        if self.kind == MemoryChipType::Initialize {
            builder.when(local.is_real).assert_eq(local.clk_low, AB::F::ONE);
            builder.when(local.is_real).assert_zero(local.clk_high);
        }

        // Constraints related to register %x0.
        //
        // Register %x0 should always be 0. See 2.6 Load and Store Instruction on P.18 of the MIPS
        // spec. When `is_comp` is false (the sole initialization/finalization of address 0), the
        // address and value must both be zero.
        //
        // **Remark**: it is up to the verifier to ensure this happens exactly once, constrained by
        // the public values setting `previous_init_addr`/`previous_finalize_addr` to zero.
        let is_not_comp = local.is_real - local.is_comp;
        builder.when(is_not_comp.clone()).assert_word_zero(local.addr);
        builder.when(is_not_comp).assert_word_zero(local.value);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::programs::tests::simple_program;
    // use crate::{
    //     mips::MipsAir, syscall::precompiles::sha256::extend_tests::sha_extend_program,
    //     utils::setup_logger,
    // };
    use p3_koala_bear::KoalaBear;
    use zkm_core_executor::Executor;
    // `LookupKind`/`LookupScope` are re-imported here (shadowing the `use super::*` glob,
    // which now brings in the new `zkm_hypercube` versions) because `debug_lookups_with_all_chips`
    // and `StarkMachine` below are still the old FRI-backed `zkm_stark` utilities and expect the
    // old-crate types.
    use zkm_stark::{
        // air::LookupScope, debug_lookups_with_all_chips, koala_bear_poseidon2::KoalaBearPoseidon2,
        // LookupKind, StarkMachine,
        ZKMCoreOpts,
    };

    #[test]
    fn test_memory_generate_trace() {
        let program = simple_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.record.clone();

        let chip: MemoryGlobalChip = MemoryGlobalChip::new(MemoryChipType::Initialize);

        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values);

        let chip: MemoryGlobalChip = MemoryGlobalChip::new(MemoryChipType::Finalize);
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values);

        for mem_event in shard.global_memory_finalize_events {
            println!("{mem_event:?}");
        }
    }

    #[test]
    #[ignore = "no zkm-hypercube shard prove/verify driver yet (old FRI-backed run_test/CpuProver removed)"]
    fn test_memory_lookups() {
        // setup_logger();
        // let program = sha_extend_program();
        // let program_clone = program.clone();
        // let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        // runtime.run().unwrap();
        // let machine: StarkMachine<KoalaBearPoseidon2, MipsAir<KoalaBear>> =
        //     MipsAir::machine(KoalaBearPoseidon2::new());
        // let (pkey, _) = machine.setup(&program_clone);
        // let opts = ZKMCoreOpts::default();
        // machine.generate_dependencies(&mut runtime.records, &opts, None).unwrap();
        //
        // let shards = runtime.records;
        // for shard in shards.clone() {
        //     debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
        //         &machine,
        //         &pkey,
        //         &[shard],
        //         vec![LookupKind::Memory],
        //         LookupScope::Local,
        //     );
        // }
        // debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
        //     &machine,
        //     &pkey,
        //     &shards,
        //     vec![LookupKind::Memory],
        //     LookupScope::Global,
        // );
    }

    #[test]
    #[ignore = "no zkm-hypercube shard prove/verify driver yet (old FRI-backed run_test/CpuProver removed)"]
    fn test_byte_lookups() {
        // setup_logger();
        // let program = sha_extend_program();
        // let program_clone = program.clone();
        // let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        // runtime.run().unwrap();
        // let machine = MipsAir::machine(KoalaBearPoseidon2::new());
        // let (pkey, _) = machine.setup(&program_clone);
        // let opts = ZKMCoreOpts::default();
        // machine.generate_dependencies(&mut runtime.records, &opts, None).unwrap();
        //
        // let shards = runtime.records;
        // debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
        //     &machine,
        //     &pkey,
        //     &shards,
        //     vec![LookupKind::Byte],
        //     LookupScope::Global,
        // );
    }
}
