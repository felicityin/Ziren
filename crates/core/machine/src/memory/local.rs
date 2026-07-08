use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_field::PrimeField32;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use zkm_core_executor::events::{ByteLookupEvent, ByteRecord, GlobalLookupEvent, MemoryLocalEvent};
use zkm_core_executor::{ExecutionRecord, Program};
use zkm_derive::AlignedBorrow;
use zkm_stark::{
    air::{AirLookup, LookupScope, MachineAir, ZKMAirBuilder},
    LookupKind, Word,
};

use crate::{
    air::{MemoryAirBuilder, WordAirBuilder},
    operations::KoalaBearBitDecomposition,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

pub const NUM_LOCAL_MEMORY_ENTRIES_PER_ROW: usize = 4;
pub(crate) const NUM_MEMORY_LOCAL_INIT_COLS: usize = size_of::<MemoryLocalCols<u8>>();

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct SingleMemoryLocal<T: Copy> {
    /// The address of the memory access.
    pub addr: T,

    /// The bit decomposition of `addr`, used to range check that `addr` is a valid KoalaBear
    /// field element (i.e. strictly less than the modulus `0x7F000001`).
    pub addr_bits: KoalaBearBitDecomposition<T>,

    /// The initial shard of the memory access.
    pub initial_shard: T,

    /// The final shard of the memory access.
    pub final_shard: T,

    /// The initial clk of the memory access.
    pub initial_clk: T,

    /// The final clk of the memory access.
    pub final_clk: T,

    /// The 16-bit limb of `initial_shard`, used for its 16-bit range check.
    pub initial_shard_16bit_limb: T,

    /// The 16-bit limb of `final_shard`, used for its 16-bit range check.
    pub final_shard_16bit_limb: T,

    /// The 16-bit limb of `initial_clk`, used for its 24-bit range check.
    pub initial_clk_16bit_limb: T,
    /// The 8-bit limb of `initial_clk`, used for its 24-bit range check.
    pub initial_clk_8bit_limb: T,

    /// The 16-bit limb of `final_clk`, used for its 24-bit range check.
    pub final_clk_16bit_limb: T,
    /// The 8-bit limb of `final_clk`, used for its 24-bit range check.
    pub final_clk_8bit_limb: T,

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
        // Byte lookups required by the defense-in-depth range checks emitted in `eval`.
        let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();

        input.get_local_mem_events().for_each(|mem_event| {
            events.push(GlobalLookupEvent {
                message: [
                    mem_event.initial_mem_access.shard,
                    mem_event.initial_mem_access.timestamp,
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
                    mem_event.final_mem_access.shard,
                    mem_event.final_mem_access.timestamp,
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

            // 16-bit range checks for shards.
            for value in [mem_event.initial_mem_access.shard, mem_event.final_mem_access.shard] {
                blu.add_u16_range_check(value as u16);
            }

            // 24-bit range checks (16-bit + 8-bit limbs) for the clk fields.
            for value in
                [mem_event.initial_mem_access.timestamp, mem_event.final_mem_access.timestamp]
            {
                blu.add_u16_range_check((value & 0xffff) as u16);
                blu.add_u8_range_check(0, ((value >> 16) & 0xff) as u8);
            }
        });

        output.global_lookup_events.extend(events);
        output.add_byte_lookup_events_from_maps(vec![&blu]);
        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let count = input.get_local_mem_events().count();
        let nb_rows = nb_rows(count);
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        Some(next_power_of_two(
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
                        let initial_shard = event.initial_mem_access.shard;
                        let final_shard = event.final_mem_access.shard;
                        let initial_clk = event.initial_mem_access.timestamp;
                        let final_clk = event.final_mem_access.timestamp;

                        cols.addr = F::from_canonical_u32(event.addr);
                        cols.addr_bits.populate(event.addr);
                        cols.initial_shard = F::from_canonical_u32(initial_shard);
                        cols.final_shard = F::from_canonical_u32(final_shard);
                        cols.initial_clk = F::from_canonical_u32(initial_clk);
                        cols.final_clk = F::from_canonical_u32(final_clk);

                        // Populate the limbs backing the defense-in-depth range checks.
                        for (value, limb_16, limb_8) in [
                            (
                                initial_clk,
                                &mut cols.initial_clk_16bit_limb,
                                &mut cols.initial_clk_8bit_limb,
                            ),
                            (
                                final_clk,
                                &mut cols.final_clk_16bit_limb,
                                &mut cols.final_clk_8bit_limb,
                            ),
                        ] {
                            *limb_16 = F::from_canonical_u32(value & 0xffff);
                            *limb_8 = F::from_canonical_u32((value >> 16) & 0xff);
                        }
                        cols.initial_shard_16bit_limb = F::from_canonical_u32(initial_shard);
                        cols.final_shard_16bit_limb = F::from_canonical_u32(final_shard);

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
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            shard.get_local_mem_events().nth(0).is_some()
        }
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

            // Defense-in-depth: range check `addr` to be a valid KoalaBear field element
            // (strictly less than the modulus `0x7F000001`).
            KoalaBearBitDecomposition::<AB::F>::range_check(
                builder,
                local.addr,
                local.addr_bits,
                local.is_real.into(),
            );

            // Defense-in-depth: range check shards to 16 bits and clocks to 24 bits.
            builder
                .when(local.is_real)
                .assert_eq(local.initial_shard, local.initial_shard_16bit_limb);
            builder.when(local.is_real).assert_eq(local.final_shard, local.final_shard_16bit_limb);
            builder.slice_range_check_u16(
                &[local.initial_shard_16bit_limb, local.final_shard_16bit_limb],
                local.is_real,
            );
            builder.eval_range_check_24bits(
                local.initial_clk,
                local.initial_clk_16bit_limb,
                local.initial_clk_8bit_limb,
                local.is_real,
            );
            builder.eval_range_check_24bits(
                local.final_clk,
                local.final_clk_16bit_limb,
                local.final_clk_8bit_limb,
                local.is_real,
            );

            let mut values =
                vec![local.initial_shard.into(), local.initial_clk.into(), local.addr.into()];
            values.extend(local.initial_value.map(Into::into));
            builder.receive(
                AirLookup::new(values.clone(), local.is_real.into(), LookupKind::Memory),
                LookupScope::Local,
            );

            // Send the lookup to the global table.
            builder.send(
                AirLookup::new(
                    vec![
                        local.initial_shard.into(),
                        local.initial_clk.into(),
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
                        local.final_shard.into(),
                        local.final_clk.into(),
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
                vec![local.final_shard.into(), local.final_clk.into(), local.addr.into()];
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
    use zkm_stark::{
        air::{LookupScope, MachineAir},
        debug_lookups_with_all_chips,
        koala_bear_poseidon2::KoalaBearPoseidon2,
        LookupKind, StarkMachine, ZKMCoreOpts,
    };

    use crate::{
        memory::MemoryLocalChip, mips::MipsAir,
        syscall::precompiles::sha256::extend_tests::sha_extend_program, utils::setup_logger,
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
    fn test_memory_local_defense_in_depth_lookups() {
        // Uses the inline `simple_program` (no guest ELF) so it can run without the zkVM
        // toolchain. Verifies that the byte-lookup events recorded in `generate_dependencies`
        // for the defense-in-depth range checks exactly balance the AIR `send_byte` calls, and
        // that the memory lookups still balance.
        setup_logger();
        let program = simple_program();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();

        // Sanity check: the program must exercise the memory-local chip for the byte-lookup
        // balance assertion below to be meaningful.
        let n_local_events: usize =
            runtime.records.iter().map(|r| r.get_local_mem_events().count()).sum();
        assert!(n_local_events > 0, "expected the test program to produce local memory events");

        let machine: StarkMachine<KoalaBearPoseidon2, MipsAir<KoalaBear>> =
            MipsAir::machine(KoalaBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = ZKMCoreOpts::default();
        machine.generate_dependencies(&mut runtime.records, &opts, None).unwrap();

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
                &machine,
                &pkey,
                &[shard],
                vec![LookupKind::Memory],
                LookupScope::Local,
            );
        }
        debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
            &machine,
            &pkey,
            &shards,
            vec![LookupKind::Byte],
            LookupScope::Global,
        );
    }

    #[test]
    fn test_memory_lookup_lookups() {
        setup_logger();
        let program = sha_extend_program();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        let machine: StarkMachine<KoalaBearPoseidon2, MipsAir<KoalaBear>> =
            MipsAir::machine(KoalaBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = ZKMCoreOpts::default();
        machine.generate_dependencies(&mut runtime.records, &opts, None).unwrap();

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
                &machine,
                &pkey,
                &[shard],
                vec![LookupKind::Memory],
                LookupScope::Local,
            );
        }
        debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
            &machine,
            &pkey,
            &shards,
            vec![LookupKind::Memory],
            LookupScope::Global,
        );
    }

    #[test]
    fn test_byte_lookup_lookups() {
        setup_logger();
        let program = sha_extend_program();
        let program_clone = program.clone();
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        let machine = MipsAir::machine(KoalaBearPoseidon2::new());
        let (pkey, _) = machine.setup(&program_clone);
        let opts = ZKMCoreOpts::default();
        machine.generate_dependencies(&mut runtime.records, &opts, None).unwrap();

        let shards = runtime.records;
        for shard in shards.clone() {
            debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
                &machine,
                &pkey,
                &[shard],
                vec![LookupKind::Memory],
                LookupScope::Local,
            );
        }
        debug_lookups_with_all_chips::<KoalaBearPoseidon2, MipsAir<KoalaBear>>(
            &machine,
            &pkey,
            &shards,
            vec![LookupKind::Byte],
            LookupScope::Global,
        );
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
                    let init_shard = thread_rng().gen_range(0..(1u32 << 16));
                    let init_timestamp = thread_rng().gen_range(0..(1u32 << 24));
                    let final_value = thread_rng().gen_range(0..u32::MAX);
                    let final_timestamp = thread_rng().gen_range(0..(1u32 << 24));
                    let final_shard = thread_rng().gen_range(0..(1u32 << 16));
                    MemoryLocalEvent {
                        addr,
                        initial_mem_access: MemoryRecord {
                            shard: init_shard,
                            timestamp: init_timestamp,
                            value: init_value,
                        },
                        final_mem_access: MemoryRecord {
                            shard: final_shard,
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
