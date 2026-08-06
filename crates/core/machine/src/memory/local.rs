use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, BaseAir};
use p3_field::FieldAlgebra;
use p3_field::PrimeField32;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use zkm_core_executor::events::{ByteLookupEvent, ByteRecord, GlobalLookupEvent, MemoryLocalEvent};
use zkm_core_executor::{ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{
    air::{AirLookup, LookupScope, MachineAir, ZKMAirBuilder},
    lookup::LookupKind,
    word::Word,
};

use crate::{
    air::WordAirBuilder,
    utils::{next_multiple_of_32, zeroed_f_vec},
    CoreChipError,
};

pub const NUM_LOCAL_MEMORY_ENTRIES_PER_ROW: usize = 1;
pub(crate) const NUM_MEMORY_LOCAL_INIT_COLS: usize = size_of::<MemoryLocalCols<u8>>();

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct SingleMemoryLocal<T: Copy> {
    /// The address of the memory access.
    pub addr: T,

    /// The initial `clk_high` of the memory access.
    pub initial_clk_high: T,

    /// The final `clk_high` of the memory access.
    pub final_clk_high: T,

    /// The initial `clk_low` of the memory access.
    pub initial_low: T,

    /// The final `clk_low` of the memory access.
    pub final_low: T,

    /// The initial value of the memory access.
    pub initial_value: Word<T>,

    /// The final value of the memory access.
    pub final_value: Word<T>,

    /// Whether the memory access is a real access.
    pub is_real: T,
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct MemoryLocalCols<T: Copy> {
    memory_local_entries: [SingleMemoryLocal<T>; NUM_LOCAL_MEMORY_ENTRIES_PER_ROW],
}

pub struct MemoryLocalChip {}

impl MemoryLocalChip {
    /// Creates a new memory chip with a certain type.
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F> BaseAir<F> for MemoryLocalChip {
    fn width(&self) -> usize {
        NUM_MEMORY_LOCAL_INIT_COLS
    }
}

fn nb_rows(count: usize) -> usize {
    if NUM_LOCAL_MEMORY_ENTRIES_PER_ROW > 1 {
        count.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW)
    } else {
        count
    }
}

impl<F: PrimeField32> MachineAir<F> for MemoryLocalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "MemoryLocal".to_string()
    }

    fn generate_dependencies(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<(), Self::Error> {
        let mut events = Vec::new();
        let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();

        input.get_local_mem_events().for_each(|mem_event| {
            let initial_high = (mem_event.initial_mem_access.timestamp >> 24) as u32;
            let initial_low = (mem_event.initial_mem_access.timestamp & 0xffffff) as u32;
            let final_high = (mem_event.final_mem_access.timestamp >> 24) as u32;
            let final_low = (mem_event.final_mem_access.timestamp & 0xffffff) as u32;

            events.push(GlobalLookupEvent {
                message: [
                    initial_high,
                    initial_low,
                    mem_event.addr,
                    mem_event.initial_mem_access.value & 255,
                    (mem_event.initial_mem_access.value >> 8) & 255,
                    (mem_event.initial_mem_access.value >> 16) & 255,
                    (mem_event.initial_mem_access.value >> 24) & 255,
                ],
                is_receive: true,
                kind: LookupKind::Memory as u8,
            });
            events.push(GlobalLookupEvent {
                message: [
                    final_high,
                    final_low,
                    mem_event.addr,
                    mem_event.final_mem_access.value & 255,
                    (mem_event.final_mem_access.value >> 8) & 255,
                    (mem_event.final_mem_access.value >> 16) & 255,
                    (mem_event.final_mem_access.value >> 24) & 255,
                ],
                is_receive: false,
                kind: LookupKind::Memory as u8,
            });

            // Byte range check the eight value limbs (initial and final).
            blu.add_u8_range_checks(&mem_event.initial_mem_access.value.to_le_bytes());
            blu.add_u8_range_checks(&mem_event.final_mem_access.value.to_le_bytes());
        });

        output.global_lookup_events.extend(events);
        output.add_byte_lookup_events_from_maps(vec![&blu]);
        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let count = input.get_local_mem_events().count();
        let nb_rows = nb_rows(count);
        let size_log2 = None;
        Some(next_multiple_of_32(
            nb_rows,
            size_log2,
            <MemoryLocalChip as MachineAir<F>>::name(self).as_str(),
        ))
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let events = input.get_local_mem_events().collect::<Vec<_>>();
        let nb_rows = nb_rows(events.len());
        let padded_nb_rows = <MemoryLocalChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MEMORY_LOCAL_INIT_COLS);
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 0) + 1;

        let mut chunks = values[..nb_rows * NUM_MEMORY_LOCAL_INIT_COLS]
            .chunks_mut(chunk_size * NUM_MEMORY_LOCAL_INIT_COLS)
            .collect::<Vec<_>>();

        chunks.par_iter_mut().enumerate().for_each(|(i, rows)| {
            rows.chunks_mut(NUM_MEMORY_LOCAL_INIT_COLS).enumerate().for_each(|(j, row)| {
                let idx = (i * chunk_size + j) * NUM_LOCAL_MEMORY_ENTRIES_PER_ROW;

                let cols: &mut MemoryLocalCols<F> = row.borrow_mut();
                for k in 0..NUM_LOCAL_MEMORY_ENTRIES_PER_ROW {
                    let cols = &mut cols.memory_local_entries[k];
                    if idx + k < events.len() {
                        let event: &&MemoryLocalEvent = &events[idx + k];
                        let initial_high = event.initial_mem_access.timestamp >> 24;
                        let final_high = event.final_mem_access.timestamp >> 24;
                        let initial_low = event.initial_mem_access.timestamp & 0xffffff;
                        let final_low = event.final_mem_access.timestamp & 0xffffff;

                        cols.addr = F::from_canonical_u32(event.addr);
                        cols.initial_clk_high = F::from_canonical_u64(initial_high);
                        cols.final_clk_high = F::from_canonical_u64(final_high);
                        cols.initial_low = F::from_canonical_u64(initial_low);
                        cols.final_low = F::from_canonical_u64(final_low);

                        cols.initial_value = event.initial_mem_access.value.into();
                        cols.final_value = event.final_mem_access.value.into();
                        cols.is_real = F::ONE;
                    }
                }
            });
        });

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(values, NUM_MEMORY_LOCAL_INIT_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        shard.get_local_mem_events().nth(0).is_some()
    }

    fn commit_scope(&self) -> LookupScope {
        LookupScope::Local
    }
}

impl<AB> Air<AB> for MemoryLocalChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryLocalCols<AB::Var> = (*local).borrow();

        for local in local.memory_local_entries.iter() {
            builder.assert_bool(local.is_real);

            // Defense-in-depth: byte range check all eight value limbs via the byte lookup table.
            builder.slice_range_check_u8(&local.initial_value.0, local.is_real);
            builder.slice_range_check_u8(&local.final_value.0, local.is_real);

            // `initial_clk_high`/`final_clk_high`/`initial_low`/`final_low` need no range check
            // here: they're matched via the `receive`/`send` `LookupKind::Memory` interactions
            // below against the originating instruction's own `clk_high`/`clk_low`, which is
            // already range-checked to 24 bits by `CpuState`'s `eval_cpu_state` (`clk_low`) and the
            // `clk_high`-transition chip (`clk_high`) wherever it was genuinely produced -- see
            // `crate::air::MemoryAirBuilder::eval_memory_access_timestamp`'s doc comment.

            let mut values =
                vec![local.initial_clk_high.into(), local.initial_low.into(), local.addr.into()];
            values.extend(local.initial_value.map(Into::into));
            builder.receive(
                AirLookup::new(values.clone(), local.is_real.into(), LookupKind::Memory),
                LookupScope::Local,
            );

            // Send the lookup to the global table.
            builder.send(
                AirLookup::new(
                    vec![
                        local.initial_clk_high.into(),
                        local.initial_low.into(),
                        local.addr.into(),
                        local.initial_value[0].into(),
                        local.initial_value[1].into(),
                        local.initial_value[2].into(),
                        local.initial_value[3].into(),
                        local.is_real.into() * AB::Expr::zero(),
                        local.is_real.into() * AB::Expr::one(),
                        AB::Expr::from_canonical_u8(LookupKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    LookupKind::Global,
                ),
                LookupScope::Local,
            );

            // Send the lookup to the global table.
            builder.send(
                AirLookup::new(
                    vec![
                        local.final_clk_high.into(),
                        local.final_low.into(),
                        local.addr.into(),
                        local.final_value[0].into(),
                        local.final_value[1].into(),
                        local.final_value[2].into(),
                        local.final_value[3].into(),
                        local.is_real.into() * AB::Expr::one(),
                        local.is_real.into() * AB::Expr::zero(),
                        AB::Expr::from_canonical_u8(LookupKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    LookupKind::Global,
                ),
                LookupScope::Local,
            );

            let mut values =
                vec![local.final_clk_high.into(), local.final_low.into(), local.addr.into()];
            values.extend(local.final_value.map(Into::into));
            builder.send(
                AirLookup::new(values.clone(), local.is_real.into(), LookupKind::Memory),
                LookupScope::Local,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::programs::tests::simple_program;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{ExecutionRecord, Executor};
    // `MachineAir` is the new `zkm_hypercube` trait (needed for `chip.generate_trace(..)` since
    // `MemoryLocalChip` now implements it). `LookupKind`/`LookupScope` stay on the old `zkm_stark`
    // crate because `debug_lookups_with_all_chips` and `StarkMachine` below are still the old
    // FRI-backed utilities and expect the old-crate types.
    use zkm_hypercube::air::MachineAir;
    use zkm_stark::{
        // air::LookupScope, debug_lookups_with_all_chips, koala_bear_poseidon2::KoalaBearPoseidon2,
        // LookupKind, StarkMachine,
        ZKMCoreOpts,
    };

    use crate::{
        memory::MemoryLocalChip,
        // mips::MipsAir,
        // syscall::precompiles::sha256::extend_tests::sha_extend_program, utils::setup_logger,
    };

    #[test]
    fn test_local_memory_generate_trace() {
        let program = simple_program();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        let shard = runtime.records[0].clone();

        let chip: MemoryLocalChip = MemoryLocalChip::new();

        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values);

        for mem_event in shard.global_memory_finalize_events {
            println!("{mem_event:?}");
        }
    }

    #[test]
    #[ignore = "no zkm-hypercube shard prove/verify driver yet (old FRI-backed run_test/CpuProver removed)"]
    fn test_memory_local_defense_in_depth_lookups() {
        // // Uses the inline `simple_program` (no guest ELF) so it can run without the zkVM
        // // toolchain. Verifies that the byte-lookup events recorded in `generate_dependencies`
        // // for the defense-in-depth range checks exactly balance the AIR `send_byte` calls, and
        // // that the memory lookups still balance.
        // setup_logger();
        // let program = simple_program();
        // let program_clone = program.clone();
        // let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        // runtime.run().unwrap();
        //
        // // Sanity check: the program must exercise the memory-local chip for the byte-lookup
        // // balance assertion below to be meaningful.
        // let n_local_events: usize =
        //     runtime.records.iter().map(|r| r.get_local_mem_events().count()).sum();
        // assert!(n_local_events > 0, "expected the test program to produce local memory events");
        //
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
        //     vec![LookupKind::Byte],
        //     LookupScope::Global,
        // );
    }

    #[test]
    #[ignore = "no zkm-hypercube shard prove/verify driver yet (old FRI-backed run_test/CpuProver removed)"]
    fn test_memory_lookup_lookups() {
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
    fn test_byte_lookup_lookups() {
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
        //     vec![LookupKind::Byte],
        //     LookupScope::Global,
        // );
    }

    #[cfg(feature = "sys")]
    fn get_test_execution_record() -> ExecutionRecord {
        use p3_field::PrimeField32;
        use rand::{thread_rng, Rng};
        use zkm_core_executor::events::{MemoryLocalEvent, MemoryRecord};

        let cpu_local_memory_access = (0..=255)
            .flat_map(|_| {
                [{
                    let addr = thread_rng().gen_range(0..KoalaBear::ORDER_U32);
                    let init_value = thread_rng().gen_range(0..u32::MAX);
                    let init_timestamp = thread_rng().gen_range(0..(1u64 << 48));
                    let final_value = thread_rng().gen_range(0..u32::MAX);
                    let final_timestamp = thread_rng().gen_range(0..(1u64 << 48));
                    MemoryLocalEvent {
                        addr,
                        initial_mem_access: MemoryRecord {
                            timestamp: init_timestamp,
                            value: init_value,
                        },
                        final_mem_access: MemoryRecord {
                            timestamp: final_timestamp,
                            value: final_value,
                        },
                    }
                }]
            })
            .collect::<Vec<_>>();
        ExecutionRecord { cpu_local_memory_access, ..Default::default() }
    }

    #[cfg(feature = "sys")]
    #[test]
    fn test_generate_trace_ffi_eq_rust() {
        use p3_matrix::Matrix;

        let record = get_test_execution_record();
        let chip = MemoryLocalChip::new();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&record, &mut ExecutionRecord::default()).unwrap();
        let trace_ffi = generate_trace_ffi(&record, trace.height());

        assert_eq!(trace_ffi, trace);
    }

    #[cfg(feature = "sys")]
    fn generate_trace_ffi(input: &ExecutionRecord, height: usize) -> RowMajorMatrix<KoalaBear> {
        use std::borrow::BorrowMut;

        use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

        use crate::{
            memory::{
                MemoryLocalCols, NUM_LOCAL_MEMORY_ENTRIES_PER_ROW, NUM_MEMORY_LOCAL_INIT_COLS,
            },
            utils::zeroed_f_vec,
        };

        type F = KoalaBear;
        // Generate the trace rows for each event.
        let events = input.get_local_mem_events().collect::<Vec<_>>();
        let nb_rows = events.len().div_ceil(4);
        let padded_nb_rows = height;
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MEMORY_LOCAL_INIT_COLS);
        let chunk_size = std::cmp::max(nb_rows / num_cpus::get(), 0) + 1;

        let mut chunks = values[..nb_rows * NUM_MEMORY_LOCAL_INIT_COLS]
            .chunks_mut(chunk_size * NUM_MEMORY_LOCAL_INIT_COLS)
            .collect::<Vec<_>>();

        chunks.par_iter_mut().enumerate().for_each(|(i, rows)| {
            rows.chunks_mut(NUM_MEMORY_LOCAL_INIT_COLS).enumerate().for_each(|(j, row)| {
                let idx = (i * chunk_size + j) * NUM_LOCAL_MEMORY_ENTRIES_PER_ROW;
                let cols: &mut MemoryLocalCols<F> = row.borrow_mut();
                for k in 0..NUM_LOCAL_MEMORY_ENTRIES_PER_ROW {
                    let cols = &mut cols.memory_local_entries[k];
                    if idx + k < events.len() {
                        unsafe {
                            crate::sys::memory_local_event_to_row_koalabear(events[idx + k], cols);
                        }
                    }
                }
            });
        });

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_MEMORY_LOCAL_INIT_COLS)
    }
}
