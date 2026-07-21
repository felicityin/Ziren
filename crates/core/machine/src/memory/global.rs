use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use std::{array, iter::once};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use zkm_core_executor::events::{GlobalLookupEvent, MemoryInitializeFinalizeEvent};
use zkm_core_executor::{ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
use zkm_hypercube::{
    air::{AirLookup, LookupScope, MachineAir, ZKMAirBuilder},
    lookup::LookupKind,
};

use crate::{
    operations::{AssertLtColsBits, IsZeroOperation, KoalaBearBitDecomposition},
    utils::next_power_of_two,
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

        let events = memory_events.into_iter().map(|event| {
            let lookup_clk_high = if is_receive { (event.timestamp >> 28) as u32 } else { 0 };
            let lookup_clk_low = if is_receive { (event.timestamp & 0xfff_ffff) as u32 } else { 0 };
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
        let padded_nb_rows = next_power_of_two(
            nb_rows,
            size_log2,
            <MemoryGlobalChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };

        let previous_addr_bits = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr_bits,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr_bits,
        };

        memory_events.sort_by_key(|event| event.addr);
        let mut rows: Vec<[F; NUM_MEMORY_INIT_COLS]> = memory_events
            .par_iter()
            .map(|event| {
                let MemoryInitializeFinalizeEvent { addr, value, timestamp } = event.to_owned();
                let clk_high = (timestamp >> 28) as u32;
                let clk_low = (timestamp & 0xfff_ffff) as u32;

                let mut row = [F::ZERO; NUM_MEMORY_INIT_COLS];
                let cols: &mut MemoryInitCols<F> = row.as_mut_slice().borrow_mut();
                cols.addr = F::from_canonical_u32(addr);
                cols.addr_bits.populate(addr);
                cols.clk_high = F::from_canonical_u32(clk_high);
                cols.timestamp = F::from_canonical_u32(clk_low);
                cols.value = array::from_fn(|i| F::from_canonical_u32((value >> i) & 1));
                cols.is_real = F::one();

                row
            })
            .collect::<Vec<_>>();

        for i in 0..memory_events.len() {
            let addr = memory_events[i].addr;
            let cols: &mut MemoryInitCols<F> = rows[i].as_mut_slice().borrow_mut();

            let prev_addr = if i == 0 {
                previous_addr_bits.iter().enumerate().map(|(j, bit)| bit * (1 << j)).sum::<u32>()
            } else {
                memory_events[i - 1].addr
            };
            let prev_addr_bits: [_; 32] = array::from_fn(|j| (prev_addr >> j) & 1);

            cols.index = F::from_canonical_u32(i as u32);
            cols.prev_addr_bits = prev_addr_bits.map(F::from_canonical_u32);
            cols.prev_valid = F::from_bool(!(prev_addr == 0 && i != 0));
            let is_prev_addr_zero = cols.is_prev_addr_zero.populate(prev_addr);
            let is_index_zero = cols.is_index_zero.populate(i as u32);
            cols.is_addr_zero.populate(addr);
            cols.is_prev_addr_and_index_zero =
                F::from_bool(is_prev_addr_zero == 1 && is_index_zero == 1);

            let is_comp = prev_addr != 0 || i != 0 || addr != 0;
            cols.is_comp = F::from_bool(is_comp);
            if is_comp {
                debug_assert!(prev_addr < addr, "prev_addr {prev_addr} < addr {addr}");
                let addr_bits: [_; 32] = array::from_fn(|j| (addr >> j) & 1);
                cols.lt_cols.populate(&prev_addr_bits, &addr_bits);
            }
        }

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
    /// The clk's high limb (bits above the low 28-bit window) of the memory access. See
    /// `zkm_core_machine::adapter::state::CpuState`'s doc comment.
    #[cfg_attr(feature = "picus", picus(input, transition_input))]
    pub clk_high: T,

    /// The clk (low 28 bits) of the memory access.
    #[cfg_attr(feature = "picus", picus(input, transition_input))]
    pub timestamp: T,

    /// The address of the memory access.
    #[cfg_attr(feature = "picus", picus(input, transition_input))]
    pub addr: T,

    /// Comparison assertions for address to be strictly increasing.
    pub lt_cols: AssertLtColsBits<T, 32>,

    /// A bit decomposition of `addr`.
    pub addr_bits: KoalaBearBitDecomposition<T>,

    /// The value of the memory access.
    #[cfg_attr(feature = "picus", picus(transition_input))]
    pub value: [T; 32],

    /// Whether the memory access is a real access.
    pub is_real: T,

    /// This row's position in the chip's own sorted-address sequence. Anchors the
    /// `LookupKind::MemoryGlobalInitControl`/`MemoryGlobalFinalizeControl` chain by value
    /// instead of physical row adjacency.
    pub index: T,

    /// This row's own witnessed previous address, as bits (matched by value against whichever
    /// row -- or `ExecutionRecord::eval_public_values`, at index 0 -- sent it).
    pub prev_addr_bits: [T; 32],

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
        for i in 0..32 {
            builder.assert_bool(local.value[i]);
        }
        // Canonicalize padded rows to the default zero trace shape so witness columns cannot
        // drift in extraction modules.
        builder.when_not(local.is_real).assert_zero(local.clk_high);
        builder.when_not(local.is_real).assert_zero(local.timestamp);
        builder.when_not(local.is_real).assert_zero(local.addr);
        for i in 0..32 {
            builder.when_not(local.is_real).assert_zero(local.value[i]);
            builder.when_not(local.is_real).assert_zero(local.addr_bits.bits[i]);
            builder.when_not(local.is_real).assert_zero(local.lt_cols.bit_flags[i]);
        }
        builder
            .when_not(local.is_real)
            .assert_zero(local.addr_bits.and_most_sig_byte_decomp_0_to_2);
        builder
            .when_not(local.is_real)
            .assert_zero(local.addr_bits.and_most_sig_byte_decomp_0_to_3);
        builder
            .when_not(local.is_real)
            .assert_zero(local.addr_bits.and_most_sig_byte_decomp_0_to_4);
        builder
            .when_not(local.is_real)
            .assert_zero(local.addr_bits.and_most_sig_byte_decomp_0_to_5);
        builder
            .when_not(local.is_real)
            .assert_zero(local.addr_bits.and_most_sig_byte_decomp_0_to_6);
        builder
            .when_not(local.is_real)
            .assert_zero(local.addr_bits.and_most_sig_byte_decomp_0_to_7);

        let mut byte1 = AB::Expr::zero();
        let mut byte2 = AB::Expr::zero();
        let mut byte3 = AB::Expr::zero();
        let mut byte4 = AB::Expr::zero();
        for i in 0..8 {
            byte1 = byte1.clone() + local.value[i].into() * AB::F::from_canonical_u8(1 << i);
            byte2 = byte2.clone() + local.value[i + 8].into() * AB::F::from_canonical_u8(1 << i);
            byte3 = byte3.clone() + local.value[i + 16].into() * AB::F::from_canonical_u8(1 << i);
            byte4 = byte4.clone() + local.value[i + 24].into() * AB::F::from_canonical_u8(1 << i);
        }
        let value = [byte1, byte2, byte3, byte4];

        if self.kind == MemoryChipType::Initialize {
            // Send the lookup to the global table.
            builder.send(
                AirLookup::new(
                    vec![
                        AB::Expr::zero(), // shard
                        AB::Expr::zero(), // timestamp
                        local.addr.into(),
                        value[0].clone(),
                        value[1].clone(),
                        value[2].clone(),
                        value[3].clone(),
                        local.is_real.into() * AB::Expr::one(),
                        local.is_real.into() * AB::Expr::zero(),
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
                        local.timestamp.into(),
                        local.addr.into(),
                        value[0].clone(),
                        value[1].clone(),
                        value[2].clone(),
                        value[3].clone(),
                        local.is_real.into() * AB::Expr::zero(),
                        local.is_real.into() * AB::Expr::one(),
                        AB::Expr::from_canonical_u8(LookupKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    LookupKind::Global,
                ),
                LookupScope::Local,
            );
        }

        // Canonically decompose the address into bits so we can do comparisons.
        KoalaBearBitDecomposition::<AB::F>::range_check(
            builder,
            local.addr,
            local.addr_bits,
            local.is_real.into(),
        );

        // Chain this row's own witnessed `prev_addr_bits` against whichever row (or
        // `ExecutionRecord::eval_public_values`, at index 0) sent it as its own address, and this
        // row's own `addr_bits` against whichever row (or the phantom receive, at the final index)
        // receives it as the start of the next segment -- replacing the old row-adjacency
        // `addr < addr'` chaining.
        let interaction_kind = match self.kind {
            MemoryChipType::Initialize => LookupKind::MemoryGlobalInitControl,
            MemoryChipType::Finalize => LookupKind::MemoryGlobalFinalizeControl,
        };
        builder.receive(
            AirLookup::new(
                once(local.index.into())
                    .chain(local.prev_addr_bits.iter().map(|&b| b.into()))
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
                    .chain(local.addr_bits.bits.iter().map(|&b| b.into()))
                    .chain(once(local.is_comp.into()))
                    .collect(),
                local.is_real.into(),
                interaction_kind,
            ),
            LookupScope::Local,
        );

        // Since `prev_addr_bits` is either witnessed from a genuine row's already-range-checked
        // `addr_bits`, or from public values (a separate, not-yet-in-scope trust boundary, same
        // status as `CpuChip`'s `start_pc`/`next_pc`), plain booleanity is enough here -- we get an
        // element of the field with no concern for overflow, same as the reconstruction below.
        for i in 0..32 {
            builder.when(local.is_real).assert_bool(local.prev_addr_bits[i]);
        }
        let prev_addr = local
            .prev_addr_bits
            .iter()
            .enumerate()
            .map(|(i, bit)| (*bit).into() * AB::F::from_wrapped_u32(1 << i))
            .sum::<AB::Expr>();

        IsZeroOperation::<AB::F>::eval(
            builder,
            prev_addr,
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
            local.addr.into(),
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
        local.lt_cols.eval(builder, &local.prev_addr_bits, &local.addr_bits.bits, local.is_comp);

        // Make assertions for specific types of memory chips. The genesis sentinel timestamp is
        // 1 (see `MemoryInitializeFinalizeEvent::initialize`), which is entirely within the low
        // 28-bit window.
        if self.kind == MemoryChipType::Initialize {
            builder.when(local.is_real).assert_eq(local.timestamp, AB::F::ONE);
            builder.when(local.is_real).assert_eq(local.clk_high, AB::F::ZERO);
        }

        // Constraints related to register %x0.
        //
        // Register %x0 should always be 0. See 2.6 Load and Store Instruction on P.18 of the MIPS
        // spec. When `is_comp` is false (the sole initialization/finalization of address 0), the
        // address and value must both be zero.
        //
        // **Remark**: it is up to the verifier to ensure this happens exactly once, constrained by
        // the public values setting `previous_init_addr_bits`/`previous_finalize_addr_bits` to zero.
        let is_not_comp = local.is_real - local.is_comp;
        builder.when(is_not_comp.clone()).assert_zero(local.addr);
        for i in 0..32 {
            builder.when(is_not_comp.clone()).assert_zero(local.value[i]);
        }
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
