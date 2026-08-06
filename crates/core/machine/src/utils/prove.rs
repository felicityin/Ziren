use crate::mips::MipsAir;
// Used by the commented-out run_test_machine(_with_prover) below.
// use p3_uni_stark::SymbolicAirBuilder;
// use size::Size;
use hashbrown::HashMap;
use std::thread::ScopedJoinHandle;
use std::sync::{mpsc::sync_channel, Arc, Mutex};
use thiserror::Error;
use web_time::Instant;

use p3_koala_bear::KoalaBear;

use crate::{
    io::ZKMStdin,
    utils::{concurrency::TurnBasedSync, trace_budget},
};
use zkm_core_executor::{
    estimate_record_trace_bytes,
    events::{format_table_line, sorted_table_lines, MemoryInitializeFinalizeEvent},
    mips_costs, next_shard, trace_shard,
    ExecutionError, ExecutionRecord, ExecutionReport, Executor, MipsAirId, Program, ShardDriver,
    SplicedMinimalTrace, ZKMContext, CORE_SHARD_HEIGHT_THRESHOLD,
};
use zkm_primitives::io::ZKMPublicValues;

use slop_challenger::IopCtx;
use zkm_hypercube::{
    air::PublicValues,
    config::{default_fri_config, ZkmGlobalContext, ZkmStackedPcs},
    prover::{
        AirProver, PcsProof, ProverPermit, ProverSemaphore, ShardData, TraceGenerator,
        ZkmInnerPcsProver, ZkmShardProver,
    },
    record::MachineRecord,
    ShardContextImpl, ShardProof, ShardVerifier, ZkmSC,
};

/// The concrete shard-proof type produced by Ziren's own (`KoalaBear`, jagged/basefold) shard
/// prover.
pub type ZkmShardProof = ShardProof<
    zkm_hypercube::config::ZkmGlobalContext,
    PcsProof<zkm_hypercube::config::ZkmGlobalContext, ZkmSC<MipsAir<KoalaBear>>>,
>;

/// The shard context used by Ziren's own core shard prover.
type ZkmShardContext = ShardContextImpl<ZkmGlobalContext, ZkmStackedPcs, MipsAir<KoalaBear>>;

/// The main-trace data (plus proving key) needed to prove a single shard, without having proved
/// it yet.
type ZkmShardData = ShardData<ZkmGlobalContext, ZkmShardContext, ZkmInnerPcsProver>;

// Com/MachineAir/MachineProver/OpeningProof/PcsProverData/StarkVerifyingKey are used by the
// commented-out run_test*/run_test_machine* functions below.
// use zkm_hypercube::air::MachineAir;
// use zkm_stark::{Com, MachineProver, OpeningProof, PcsProverData, StarkVerifyingKey};
use zkm_stark::{
    StarkGenericConfig, UniConfig, ZKMCoreOpts, CORE_LOG_STACKING_HEIGHT, CORE_MAX_LOG_ROW_COUNT,
};

#[derive(Error, Debug)]
pub enum ZKMCoreProverError {
    #[error("failed to execute program: {0}")]
    ExecutionError(ExecutionError),
    #[error("traces generation error")]
    TracesGenerationError,
    #[error("dependencies generation error")]
    DependenciesGenerationError,
}

pub fn prove_with_context(
    program: Program,
    stdin: &ZKMStdin,
    opts: ZKMCoreOpts,
    context: ZKMContext,
) -> Result<
    (Vec<ZkmShardProof>, Vec<u8>, u64, zkm_hypercube::MachineVerifyingKey<ZkmGlobalContext>),
    ZKMCoreProverError,
> {
    // `MinimalExecutor` reads plain `stdin.buffer` entries via `HINT_LEN`/`HINT_READ`,
    // `stdin.proofs`/`context.subproof_verifier` via `SYSVERIFY`, and now
    // `context.hook_registry` via `WRITE`'s hook-fd/`FD_HINT` cases too -- only `max_cycles`
    // remains unsupported for real proving (`ZKMProver::execute`'s dry run supports it).
    assert!(context.max_cycles.is_none(), "prove_with_context: a cycle-count limit is not yet supported");
    let stdin_buffer: Arc<[Vec<u8>]> = Arc::from(stdin.buffer.clone());
    let proof_stream = stdin.proofs.clone();
    let subproof_verifier = context.subproof_verifier;
    let deferred_proof_verification_enabled = !context.skip_deferred_proof_verification;
    // Mirrors `Executor::with_context`'s `context.hook_registry.unwrap_or_default()`: the 5
    // built-in hooks stay active by default (matching `ZKMProver::execute`'s identical default),
    // not just when a caller explicitly opts in.
    let hook_registry = Some(context.hook_registry.unwrap_or_default());

    let machine = MipsAir::<KoalaBear>::hypercube_machine();
    // Fixed independently of `opts.shard_size` (the executor's cycle-count ceiling) -- see
    // `zkm_stark::CORE_MAX_LOG_ROW_COUNT`'s doc comment.
    let max_log_row_count = CORE_MAX_LOG_ROW_COUNT;
    let shard_verifier = ShardVerifier::from_basefold_parameters(
        default_fri_config(),
        CORE_LOG_STACKING_HEIGHT,
        max_log_row_count,
        machine,
    );
    let shard_prover = Arc::new(ZkmShardProver::<MipsAir<KoalaBear>>::new(shard_verifier));
    let prover_permits = ProverSemaphore::new(opts.trace_gen_workers.max(1));
    // Per-chip byte costs, for estimating each record's real materialized trace size before
    // admitting it into the process-wide trace-memory budget (`trace_budget::acquire_trace_budget`).
    let trace_byte_costs: Arc<HashMap<MipsAirId, u64>> =
        Arc::new(mips_costs().into_iter().map(|(k, v)| (k, v as u64)).collect());

    let program_arc = Arc::new(program.clone());
    // `MinimalExecutor`'s oracle-log buffer cutoff (a memory-management knob, independent of
    // where shards actually get cut) -- reuses `shard_size`, the same value the legacy executor
    // interprets as its own per-shard cycle-count ceiling.
    let max_trace_size = opts.shard_size as u64;
    // `SplicingVM`'s shard-cut thresholds: `lde_size_threshold` bounds a shard's committed LDE
    // area (a correctness bound tied to the jagged PCS's cell-count rejection);
    // `CORE_SHARD_HEIGHT_THRESHOLD` bounds any single chip's real row count within a shard.
    let element_threshold = opts.lde_size_threshold;
    let height_threshold = CORE_SHARD_HEIGHT_THRESHOLD;

    let setup_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
    let (preprocessed, vk) =
        setup_rt.block_on(shard_prover.setup(program_arc.clone(), ProverSemaphore::new(1)));
    let pk = preprocessed.pk;

    #[cfg(feature = "debug")]
    let (all_records_tx, all_records_rx) = std::sync::mpsc::channel::<Vec<ExecutionRecord>>();

    // Record the start of the process.
    let proving_start = Instant::now();
    let span = tracing::Span::current().clone();
    std::thread::scope(move |s| {
        let _span = span.enter();

        // Spawn the checkpoint generator thread.
        let checkpoint_generator_span = tracing::Span::current().clone();
        let (checkpoints_tx, checkpoints_rx) = sync_channel::<(
            usize,
            SplicedMinimalTrace,
            bool,
            u64,
            u32,
            Option<(Vec<MemoryInitializeFinalizeEvent>, Vec<MemoryInitializeFinalizeEvent>)>,
        )>(opts.checkpoints_channel_capacity);
        let checkpoint_generator_program = program_arc.clone();
        let checkpoint_generator_stdin = stdin_buffer.clone();
        let checkpoint_generator_handle: ScopedJoinHandle<
            Result<(Vec<u8>, ExecutionReport), ZKMCoreProverError>,
        > = s.spawn(move || {
            let _span = checkpoint_generator_span.enter();
            tracing::debug_span!("checkpoint generator").in_scope(|| {
                let mut driver = ShardDriver::new_with_context(
                    checkpoint_generator_program.clone(),
                    max_trace_size,
                    checkpoint_generator_stdin,
                    proof_stream,
                    subproof_verifier,
                    deferred_proof_verification_enabled,
                    hook_registry,
                );
                let mut index = 0;
                loop {
                    // Enter the span.
                    let span = tracing::debug_span!("batch");
                    let _span = span.enter();

                    // Execute the runtime until we reach a shard boundary.
                    let (spliced, done) = next_shard(
                        &mut driver,
                        checkpoint_generator_program.clone(),
                        element_threshold,
                        height_threshold,
                    )
                    .map_err(ZKMCoreProverError::ExecutionError)?;

                    // Only meaningful once the whole run is done -- see
                    // `ShardDriver::global_memory_events`'s doc comment. Sent alongside the final
                    // checkpoint so whichever worker traces it can attach these to its record,
                    // mirroring the legacy `Executor`'s own postprocess-on-final-checkpoint timing.
                    let global_memory_events = done.then(|| driver.global_memory_events());

                    // Send the checkpoint.
                    checkpoints_tx
                        .send((
                            index,
                            spliced,
                            done,
                            driver.clk(),
                            driver.max_syscall_cycles(),
                            global_memory_events,
                        ))
                        .unwrap();
                    tracing::info!(
                        "checkpoint {} generated at {:?} (clk={})",
                        index,
                        proving_start.elapsed(),
                        driver.clk(),
                    );

                    // If we've reached the final checkpoint, break out of the loop.
                    if done {
                        break Ok((
                            driver.public_values_stream().to_vec(),
                            driver.execution_report(),
                        ));
                    }

                    // Update the index.
                    index += 1;
                }
            })
        });

        // Spawn the phase 2 record generator thread.
        let p2_record_gen_sync = Arc::new(TurnBasedSync::new());
        let p2_trace_gen_sync = Arc::new(TurnBasedSync::new());
        let checkpoints_rx = Arc::new(Mutex::new(checkpoints_rx));
        let (p2_records_and_traces_tx, p2_records_and_traces_rx) = sync_channel::<(
            Vec<ExecutionRecord>,
            Vec<ZkmShardData>,
            Vec<ProverPermit>,
            Vec<u32>,
        )>(opts.records_and_traces_channel_capacity);
        let p2_records_and_traces_tx = Arc::new(Mutex::new(p2_records_and_traces_tx));
        // A purely host-side, globally monotonic sequence number, unrelated to any in-circuit
        // public value: `trace_gen_sync`'s turn-based ordering guarantees records are sent into
        // this channel in strict submission order, but phase-2 proving below is itself
        // parallelized across shards and completes out of order, so this tags each record for
        // re-sorting back into submission order afterward.
        let p2_shard_sequence = Arc::new(Mutex::new(0u32));

        let state = Arc::new(Mutex::new(PublicValues::<u32, u32>::default().reset()));
        let deferred = Arc::new(Mutex::new(ExecutionRecord::new(program_arc.clone())));
        let mut p2_record_and_trace_gen_handles = Vec::new();
        for _ in 0..opts.trace_gen_workers {
            let record_gen_sync = Arc::clone(&p2_record_gen_sync);
            let trace_gen_sync = Arc::clone(&p2_trace_gen_sync);
            let records_and_traces_tx = Arc::clone(&p2_records_and_traces_tx);
            let checkpoints_rx = Arc::clone(&checkpoints_rx);

            let state = Arc::clone(&state);
            let deferred = Arc::clone(&deferred);
            let shard_sequence = Arc::clone(&p2_shard_sequence);
            let program = program_arc.clone();
            let shard_prover = Arc::clone(&shard_prover);
            let machine = shard_prover.machine().clone();
            let pk = Arc::clone(&pk);
            let prover_permits = prover_permits.clone();
            let trace_byte_costs = Arc::clone(&trace_byte_costs);
            let async_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();

            let span = tracing::Span::current().clone();

            #[cfg(feature = "debug")]
            let all_records_tx = all_records_tx.clone();

            let handle = s.spawn(move || {
                let _span = span.enter();
                tracing::debug_span!("phase 2 trace generation").in_scope(|| {
                    let _: () = loop {
                        // Receive the latest checkpoint.
                        let received = { checkpoints_rx.lock().unwrap().recv() };
                        if let Ok((
                            index,
                            spliced,
                            done,
                            num_cycles,
                            max_syscall_cycles,
                            global_memory_events,
                        )) = received
                        {
                            // Trace the shard and reconstruct the execution records.
                            let mut record = tracing::debug_span!("trace checkpoint")
                                .in_scope(|| trace_shard(program.clone(), &spliced, max_syscall_cycles))
                                .map_err(ZKMCoreProverError::ExecutionError)?;
                            if let Some((initialize_events, finalize_events)) = global_memory_events {
                                record.global_memory_initialize_events = initialize_events;
                                record.global_memory_finalize_events = finalize_events;
                            }
                            let mut records = vec![record];
                            log::debug!("generated {} records", records.len());

                            // Wait for our turn to update the state.
                            record_gen_sync.wait_for_turn(index);

                            // Update the public values & prover state for the shards which
                            // retired instructions.
                            let mut state = state.lock().unwrap();
                            for record in records.iter_mut() {
                                // `execution_shard` is a raw ordinal the *verifier* only checks
                                // for CPU-bearing shards (must increase by exactly 1, starting at
                                // 1, in submission order -- see `ZKMProver::verify`'s
                                // `current_execution_shard` loop): bump it only when this record
                                // actually retired an instruction, never read it back from the
                                // record itself (a freshly `trace_shard`-produced record has no
                                // opinion on its own global shard ordinal -- unlike
                                // `committed_value_digest`/`deferred_proofs_digest` below, which
                                // *are* self-contained, real values `TracingVM` itself computed).
                                if record.contains_cpu() {
                                    state.execution_shard += 1;
                                }
                                state.is_execution_shard = record.contains_cpu() as u32;
                                if let Some(first_pc) = record.first_instruction_pc {
                                    state.start_pc = first_pc;
                                    state.next_pc = record.last_next_pc;
                                    let first_clk = record.first_instruction_clk.unwrap();
                                    state.initial_clk_high = (first_clk >> 24) as u32;
                                    state.initial_clk_low = (first_clk & 0xFFFFFF) as u32;
                                    // Deliberately *not* `(record.last_timestamp >> 24) as u32`/
                                    // `(record.last_timestamp & 0xFFFFFF) as u32`: those silently
                                    // carry into the next window's `clk_high` (and wrap `clk_low`
                                    // back to a small value) exactly when this shard's last real
                                    // instruction's own increment is what crosses the `clk_high`
                                    // boundary -- but the AIR's closing `receive_state` must match
                                    // that instruction's own unreduced `send_state` bit-for-bit,
                                    // which keeps `clk_high` at *that instruction's own* value and
                                    // sends `clk_low + clk_low_increment` uncorrected, letting it
                                    // spill past 24 bits (see `eval_state_chain`'s doc comment and
                                    // `ExecutionRecord::last_instruction_clk`'s doc comment).
                                    let last_clk_high = record.last_instruction_clk >> 24;
                                    state.last_clk_high = last_clk_high as u32;
                                    state.last_clk_low =
                                        (record.last_timestamp - (last_clk_high << 24)) as u32;
                                }
                                state.committed_value_digest =
                                    record.public_values.committed_value_digest;
                                state.deferred_proofs_digest =
                                    record.public_values.deferred_proofs_digest;
                                record.public_values = *state;
                            }

                            // Defer events that are too expensive to include in every shard.
                            let mut deferred = deferred.lock().unwrap();
                            for record in records.iter_mut() {
                                deferred.append(&mut record.defer());
                            }

                            // We combine the memory init/finalize events if they are "small"
                            // and would affect performance.
                            let mut shape_fixed_records = if done
                                && num_cycles < 1 << 21
                                && deferred.global_memory_initialize_events.len()
                                    < opts.split_opts.combine_memory_threshold
                                && deferred.global_memory_finalize_events.len()
                                    < opts.split_opts.combine_memory_threshold
                            {
                                let mut records_clone = records.clone();
                                let last_record = records_clone.last_mut();
                                // See if any deferred shards are ready to be committed to.
                                let mut deferred =
                                    deferred.split(done, last_record, opts.split_opts);
                                tracing::debug!("deferred {} records", deferred.len());

                                // Update the public values & prover state for the shards which do
                                // not contain "cpu events" before
                                // committing to them.
                                if !done {
                                    state.execution_shard += 1;
                                }
                                for record in deferred.iter_mut() {
                                    state.is_execution_shard = 0;
                                    state.previous_init_addr =
                                        record.public_values.previous_init_addr;
                                    state.last_init_addr =
                                        record.public_values.last_init_addr;
                                    state.previous_finalize_addr =
                                        record.public_values.previous_finalize_addr;
                                    state.last_finalize_addr =
                                        record.public_values.last_finalize_addr;
                                    state.start_pc = state.next_pc;
                                    state.initial_clk_high = state.last_clk_high;
                                    state.initial_clk_low = state.last_clk_low;
                                    record.public_values = *state;
                                }
                                records_clone.append(&mut deferred);

                                // Generate the dependencies.
                                tracing::debug_span!("generate dependencies", index).in_scope(
                                    || -> Result<(), ZKMCoreProverError> {
                                        match machine.generate_dependencies(
                                            records_clone.iter_mut(),
                                            None,
                                        ) {
                                            Ok(()) => Ok(()),
                                            Err(e) => {
                                                tracing::error!(
                                                    "Error generating dependencies: {:?}",
                                                    e
                                                );
                                                Err(ZKMCoreProverError::DependenciesGenerationError)
                                            }
                                        }
                                    },
                                )?;

                                // Let another worker update the state.
                                record_gen_sync.advance_turn();

                                Some(records_clone)
                            } else {
                                None
                            };

                            if shape_fixed_records.is_none() {
                                // See if any deferred shards are ready to be committed to.
                                let mut deferred = deferred.split(done, None, opts.split_opts);
                                log::debug!("deferred {} records", deferred.len());

                                // Update the public values & prover state for the shards which do not
                                // contain "cpu events" before committing to them.
                                if !done {
                                    state.execution_shard += 1;
                                }
                                for record in deferred.iter_mut() {
                                    state.is_execution_shard = 0;
                                    state.previous_init_addr =
                                        record.public_values.previous_init_addr;
                                    state.last_init_addr =
                                        record.public_values.last_init_addr;
                                    state.previous_finalize_addr =
                                        record.public_values.previous_finalize_addr;
                                    state.last_finalize_addr =
                                        record.public_values.last_finalize_addr;
                                    state.start_pc = state.next_pc;
                                    state.initial_clk_high = state.last_clk_high;
                                    state.initial_clk_low = state.last_clk_low;
                                    record.public_values = *state;
                                }
                                records.append(&mut deferred);

                                // Generate the dependencies.
                                tracing::debug_span!("generate dependencies", index).in_scope(
                                    || -> Result<(), ZKMCoreProverError> {
                                        match machine.generate_dependencies(
                                            records.iter_mut(),
                                            None,
                                        ) {
                                            Ok(()) => Ok(()),
                                            Err(e) => {
                                                tracing::error!(
                                                    "Error generating dependencies: {:?}",
                                                    e
                                                );
                                                Err(ZKMCoreProverError::DependenciesGenerationError)
                                            }
                                        }
                                    },
                                )?;

                                // Let another worker update the state.
                                record_gen_sync.advance_turn();

                                shape_fixed_records = Some(records);
                            }

                            let records = shape_fixed_records.unwrap();

                            #[cfg(feature = "debug")]
                            all_records_tx.send(records.clone()).unwrap();

                            trace_gen_sync.wait_for_turn(index);

                            // Generate each record's traces and send it to the phase 2 prover
                            // immediately, one record at a time, rather than generating the
                            // whole checkpoint's traces up front and sending them as chunked
                            // batches afterwards. `ProverSemaphore::new(opts.trace_gen_workers
                            // .max(1))` gives this worker exactly one permit; `generate_main_traces`
                            // holds a permit for as long as its returned `MainTraceData` is alive,
                            // i.e. until the `ZkmShardData` wrapping it is sent downstream and
                            // dropped. Generating a later record's traces (and so acquiring its
                            // permit) while an earlier record's permit in the same checkpoint is
                            // still held -- because it hasn't been sent yet, as the old
                            // collect-then-chunk-then-send structure did -- deadlocks against that
                            // single-permit capacity whenever a checkpoint yields more than one
                            // record (e.g. a shard's own CPU execution plus a deferred precompile
                            // shard split off in the same checkpoint).
                            tracing::debug_span!("generate main traces", index).in_scope(|| {
                                for record in records {
                                    // Admission into the process-wide trace-memory budget, sized
                                    // by this record's *exact* estimated materialized-trace
                                    // bytes -- independent of `trace_gen_workers`'s thread-count-
                                    // based `prover_permits` gate below, and shared by every
                                    // concurrent caller of `prove_with_context` in this process
                                    // (see `trace_budget`'s doc comment).
                                    let estimated_bytes =
                                        estimate_record_trace_bytes(&record, &trace_byte_costs);
                                    let budget_permit = async_rt
                                        .block_on(trace_budget::acquire_trace_budget(estimated_bytes));

                                    let main_trace_data =
                                        async_rt.block_on(shard_prover.trace_generator().generate_main_traces(
                                            record.clone(),
                                            shard_prover.max_log_row_count(),
                                            prover_permits.clone(),
                                        ));
                                    let shard_data = ZkmShardData { pk: Arc::clone(&pk), main_trace_data };
                                    let sequence = {
                                        let mut sequence = shard_sequence.lock().unwrap();
                                        let this_sequence = *sequence;
                                        *sequence += 1;
                                        this_sequence
                                    };
                                    records_and_traces_tx
                                        .lock()
                                        .unwrap()
                                        .send((
                                            vec![record],
                                            vec![shard_data],
                                            vec![budget_permit],
                                            vec![sequence],
                                        ))
                                        .unwrap();
                                }
                            });

                            trace_gen_sync.advance_turn();
                        } else {
                            break;
                        }
                    };
                    Ok(())
                })
            });
            p2_record_and_trace_gen_handles.push(handle);
        }
        drop(p2_records_and_traces_tx);
        #[cfg(feature = "debug")]
        drop(all_records_tx);

        // Spawn phase 2 prover worker threads, sized by `prove_workers` -- a separate knob
        // from `trace_gen_workers` (see `ZKMCoreOpts::prove_workers`'s doc comment), since
        // trace generation and shard proving are different workloads with different scaling
        // characteristics (trace generation is lighter/more memory-bound; shard proving's
        // `commit_traces` step is heavier/more CPU-bound). Previously this was a single
        // thread draining `p2_records_and_traces_rx` one item at a time -- since each
        // channel message holds exactly one shard's data (the trace-gen workers send
        // immediately, one record at a time; see the comment above on why), that meant
        // shards were proven strictly sequentially no matter how many trace-gen workers fed
        // the channel. Mirror the same "wrap the receiver in Arc<Mutex<_>>, spawn N workers
        // that lock only for the brief recv()" pattern already used for `checkpoints_rx`
        // above, so multiple shards' `prove_shard_with_data` calls (each itself already
        // using rayon internally) can run concurrently, sharing rayon's global thread pool
        // rather than competing with each other for whole worker threads. Proofs finish out
        // of order across workers, so tag each with the host-side submission sequence number
        // assigned above and sort by it afterward -- downstream verification requires proofs
        // in strictly increasing shard order.
        let p2_prover_span = tracing::Span::current().clone();
        let p2_records_and_traces_rx = Arc::new(Mutex::new(p2_records_and_traces_rx));
        let all_shard_proofs_unordered = Arc::new(Mutex::new(Vec::new()));
        let mut p2_prover_handles = Vec::new();
        for _ in 0..opts.prove_workers.max(1) {
            let span = p2_prover_span.clone();
            let shard_prover = Arc::clone(&shard_prover);
            let rx = Arc::clone(&p2_records_and_traces_rx);
            let all_shard_proofs_unordered = Arc::clone(&all_shard_proofs_unordered);
            let handle = s.spawn(move || {
                let _span = span.enter();
                tracing::debug_span!("phase 2 prover").in_scope(|| loop {
                    let received = { rx.lock().unwrap().recv() };
                    let Ok((records, shard_data, budget_permits, sequences)) = received else {
                        break;
                    };
                    for (((_record, data), budget_permit), shard_index) in
                        records.into_iter().zip(shard_data).zip(budget_permits).zip(sequences)
                    {
                        let shard_start = Instant::now();
                        let mut challenger = ZkmGlobalContext::default_challenger();
                        data.pk.vk.observe_into(&mut challenger);
                        let (proof, _permit) = shard_prover.prove_shard_with_data(data, challenger);
                        // The trace-memory budget permit stays held through the whole
                        // commit_traces/GKR/zerocheck/prove_evaluation_claims call above --
                        // `data`'s padded main traces are what it was sized to admit -- and is
                        // only released here, once this shard no longer needs them resident.
                        drop(budget_permit);
                        tracing::info!(
                            "shard {} proved in {:?} (total elapsed {:?})",
                            shard_index,
                            shard_start.elapsed(),
                            proving_start.elapsed(),
                        );
                        let mut all_shard_proofs_unordered = all_shard_proofs_unordered.lock().unwrap();
                        all_shard_proofs_unordered.push((shard_index, proof));
                        tracing::info!(
                            "shards proved so far: {} (total elapsed {:?})",
                            all_shard_proofs_unordered.len(),
                            proving_start.elapsed(),
                        );
                    }
                });
            });
            p2_prover_handles.push(handle);
        }

        // Wait until the checkpoint generator handle has fully finished.
        let (public_values_stream, execution_report) = checkpoint_generator_handle.join().unwrap()?;

        // Wait until the records and traces have been fully generated for phase 2.
        for handle in p2_record_and_trace_gen_handles {
            handle.join().unwrap()?;
        }

        // Wait until all phase 2 prover workers have finished, then restore shard order.
        for handle in p2_prover_handles {
            handle.join().unwrap();
        }
        let mut all_shard_proofs_unordered = match Arc::try_unwrap(all_shard_proofs_unordered) {
            Ok(mutex) => mutex.into_inner().unwrap(),
            Err(_) => panic!("all_shard_proofs_unordered still has outstanding references"),
        };
        all_shard_proofs_unordered.sort_by_key(|(shard_index, _)| *shard_index);
        let all_shard_proofs: Vec<_> =
            all_shard_proofs_unordered.into_iter().map(|(_, proof)| proof).collect();

        // Log some of the `ExecutionReport` information.
        tracing::info!(
            "execution report (totals): total_cycles={}, total_syscall_cycles={}, touched_memory_addresses={}",
            execution_report.total_instruction_count(),
            execution_report.total_syscall_count(),
            execution_report.touched_memory_addresses,
        );

        // Print the opcode and syscall count tables like `du`: sorted by count (descending) and
        // with the count in the first column.
        tracing::info!("execution report (opcode counts):");
        let (width, lines) = sorted_table_lines(execution_report.opcode_counts.as_ref());
        for (label, count) in lines {
            if *count > 0 {
                tracing::info!("  {}", format_table_line(&width, &label, count));
            } else {
                tracing::debug!("  {}", format_table_line(&width, &label, count));
            }
        }

        tracing::info!("execution report (syscall counts):");
        let (width, lines) = sorted_table_lines(execution_report.syscall_counts.as_ref());
        for (label, count) in lines {
            if *count > 0 {
                tracing::info!("  {}", format_table_line(&width, &label, count));
            } else {
                tracing::debug!("  {}", format_table_line(&width, &label, count));
            }
        }

        let cycles = execution_report.total_instruction_count();

        // Print the summary.
        let proving_time = proving_start.elapsed().as_secs_f64();
        tracing::info!(
            "summary: cycles={}, e2e={}s, khz={:.2}, shards={}",
            cycles,
            proving_time,
            (cycles as f64 / (proving_time * 1000.0) as f64),
            all_shard_proofs.len(),
        );

        Ok((all_shard_proofs, public_values_stream, cycles))
    })
    .map(|(all_shard_proofs, public_values_stream, cycles)| (all_shard_proofs, public_values_stream, cycles, vk))
}

/// Runs a program and returns the public values stream.
pub fn run_test_io(
    program: Program,
    inputs: ZKMStdin,
) -> Result<ZKMPublicValues, ZKMCoreProverError> {
    let runtime = tracing::debug_span!("runtime.run(...)").in_scope(|| {
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.write_vecs(&inputs.buffer);
        runtime.run().unwrap();
        runtime
    });
    let public_values = ZKMPublicValues::from(&runtime.state.public_values_stream);

    let _ = run_test_core(runtime, inputs)?;
    Ok(public_values)
}

pub fn run_test(program: Program) -> Result<Vec<ZkmShardProof>, ZKMCoreProverError> {
    let runtime = tracing::debug_span!("runtime.run(...)").in_scope(|| {
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        runtime
    });
    run_test_core(runtime, ZKMStdin::new())
}

pub fn run_test_core(
    runtime: Executor,
    inputs: ZKMStdin,
) -> Result<Vec<ZkmShardProof>, ZKMCoreProverError> {
    let (shard_proofs, _public_values_stream, _cycles, vk) = prove_with_context(
        Program::clone(&runtime.program),
        &inputs,
        ZKMCoreOpts::default(),
        ZKMContext::default(),
    )?;

    let machine = MipsAir::<KoalaBear>::hypercube_machine();
    let max_log_row_count = CORE_MAX_LOG_ROW_COUNT;
    let shard_verifier = ShardVerifier::from_basefold_parameters(
        default_fri_config(),
        CORE_LOG_STACKING_HEIGHT,
        max_log_row_count,
        machine,
    );

    for proof in &shard_proofs {
        let mut challenger = ZkmGlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        shard_verifier.verify_shard(&vk, proof, &mut challenger).unwrap();
    }

    Ok(shard_proofs)
}

#[cfg(debug_assertions)]
#[cfg(not(doctest))]
pub fn uni_stark_prove<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<SC::Val>,
) -> Proof<UniConfig<SC>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::ProverConstraintFolder<'a, UniConfig<SC>>>
        + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, SC::Val>>,
{
    p3_uni_stark::prove(&UniConfig(config.clone()), air, challenger, trace, &vec![])
}

#[cfg(not(debug_assertions))]
pub fn uni_stark_prove<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<SC::Val>,
) -> Proof<UniConfig<SC>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::ProverConstraintFolder<'a, UniConfig<SC>>>,
{
    p3_uni_stark::prove(&UniConfig(config.clone()), air, challenger, trace, &vec![])
}

#[cfg(debug_assertions)]
#[cfg(not(doctest))]
pub fn uni_stark_verify<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    proof: &Proof<UniConfig<SC>>,
) -> Result<(), p3_uni_stark::VerificationError<p3_uni_stark::PcsError<UniConfig<SC>>>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::VerifierConstraintFolder<'a, UniConfig<SC>>>
        + for<'a> Air<p3_uni_stark::DebugConstraintBuilder<'a, SC::Val>>,
{
    p3_uni_stark::verify(&UniConfig(config.clone()), air, challenger, proof, &vec![])
}

#[cfg(not(debug_assertions))]
pub fn uni_stark_verify<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    proof: &Proof<UniConfig<SC>>,
) -> Result<(), p3_uni_stark::VerificationError<p3_uni_stark::PcsError<UniConfig<SC>>>>
where
    SC: StarkGenericConfig,
    A: Air<p3_uni_stark::SymbolicAirBuilder<SC::Val>>
        + for<'a> Air<p3_uni_stark::VerifierConstraintFolder<'a, UniConfig<SC>>>,
{
    p3_uni_stark::verify(&UniConfig(config.clone()), air, challenger, proof, &vec![])
}

use p3_air::Air;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::Proof;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::programs::tests::{
        halt_only_program, hello_world_program, hook_fp_inverse_program, simple_memory_program,
        simple_program,
    };

    #[test]
    fn run_test_core_smoke() {
        let program = simple_program();
        let runtime = Executor::new(program, ZKMCoreOpts::default());
        run_test_core(runtime, ZKMStdin::new()).unwrap();
    }

    #[test]
    fn run_test_simple_memory_smoke() {
        let program = simple_memory_program();
        let runtime = Executor::new(program, ZKMCoreOpts::default());
        run_test_core(runtime, ZKMStdin::new()).unwrap();
    }

    // `run_test_halt_only_smoke` used to reproduce a `GkrVerificationFailed(CumulativeSumMismatch(..))`
    // that occurred on any program executing a real `SYSCALL` instruction. Root-caused to two bugs
    // in `CpuChip::eval_state_chain`'s (`crates/core/machine/src/cpu/air/mod.rs`) `LookupKind::State`
    // chain: (1) a halting row's `next_pc` is forced to the public sentinel `0`, which the
    // predecessor's chain prediction doesn't know about, and (2) a row immediately following a
    // taken branch/jump (i.e. sitting in the delay slot) inherits its predecessor's resolved
    // `next_next_pc` as its own `next_pc` (see `Executor::execute_operation`'s carry-forward of
    // `self.state.next_pc`) rather than `pc + 4`. Fixed by witnessing a dedicated
    // `state_chain_next_pc` column that resolves to `pc + 4` only on a halting row and to
    // `local.next_pc` otherwise. Found via `zkm_hypercube::lookup::debug_interactions_with_all_chips`,
    // a ported-from-upstream interaction-imbalance debugger (`crates/hypercube/src/lookup/debug.rs`)
    // that nets each chip's send/receive multiplicities per lookup key.
    #[test]
    fn run_test_halt_only_smoke() {
        let program = halt_only_program();
        let runtime = Executor::new(program, ZKMCoreOpts::default());
        run_test_core(runtime, ZKMStdin::new()).unwrap();
    }

    #[test]
    fn run_test_hello_world_real_elf() {
        let program = hello_world_program();
        run_test(program).unwrap();
    }

    #[test]
    fn run_test_fibonacci_real_elf() {
        let program = Program::from(test_artifacts::FIBONACCI_ELF).unwrap();
        run_test(program).unwrap();
    }

    /// Real proof/verify over the `ec_add` precompile (via `SECP256R1_ADD`).
    #[test]
    fn run_test_secp256r1_add_real_elf() {
        let program = crate::programs::tests::secp256r1_add_program();
        run_test(program).unwrap();
    }

    /// Real proof/verify exercising a hook fd during real proving, not just `execute_fast`'s dry
    /// run: `hook_fp_inverse_program` invokes the real built-in `fp_inverse` hook (default-active,
    /// same as `ZKMProver::execute`'s default context -- see `prove_with_context`'s
    /// `unwrap_or_default()` comment), reads its spliced result back via `SYSHINTLEN`/
    /// `SYSHINTREAD`, and commits it as public values. Confirms the whole chain works through the
    /// real pipeline: `MinimalExecutor`'s real hook invocation, `SYSHINTLEN`'s new oracle-logged
    /// replay (`CoreVM`/`TracingVM` no longer recompute it from a fixed `stdin` snapshot), and the
    /// hinted word materializing correctly into the shard's own trace via `LW`.
    #[test]
    fn run_test_hook_fp_inverse_real_proof() {
        let program = hook_fp_inverse_program();
        let (shard_proofs, public_values_stream, _cycles, vk) = prove_with_context(
            program.clone(),
            &ZKMStdin::new(),
            ZKMCoreOpts::default(),
            ZKMContext::default(),
        )
        .unwrap();

        // 3 * 5 = 15 = 2*7 + 1, so 5 is the modular inverse of 3 mod 7.
        assert_eq!(public_values_stream, vec![5, 0, 0, 0], "wrong fp_inverse hook result");

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        let max_log_row_count = CORE_MAX_LOG_ROW_COUNT;
        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            CORE_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        for proof in &shard_proofs {
            let mut challenger = ZkmGlobalContext::default_challenger();
            vk.observe_into(&mut challenger);
            shard_verifier.verify_shard(&vk, proof, &mut challenger).unwrap();
        }
    }

    /// Serial (single-threaded, no channels) re-implementation of `prove_with_context` +
    /// `run_test_core`'s verify loop, proving and verifying each record immediately as it's
    /// produced instead of pipelining trace generation/proving/verification across worker
    /// threads. Manual diagnostic: pinpoints exactly which record fails without the multi-
    /// threaded pipeline's log interleaving, and gives a much shorter call stack to add ad hoc
    /// instrumentation to while investigating a `GkrVerificationFailed` mismatch.
    #[test]
    #[ignore = "manual diagnostic, run explicitly while investigating a GkrVerificationFailed mismatch"]
    fn debug_serial_prove_verify_secp256r1_add() {
        use std::sync::Arc;
        use zkm_hypercube::{
            air::PublicValues,
            prover::{ProverSemaphore, ZkmShardProver},
            ShardVerifier,
        };

        let program = Arc::new(crate::programs::tests::secp256r1_add_program());

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        let max_log_row_count = CORE_MAX_LOG_ROW_COUNT;
        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            CORE_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        let shard_prover = ZkmShardProver::<MipsAir<KoalaBear>>::new(shard_verifier.clone());
        let async_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (preprocessed, vk) =
            async_rt.block_on(shard_prover.setup(program.clone(), ProverSemaphore::new(1)));
        let pk = Arc::new(preprocessed.pk);
        let prover_permits = ProverSemaphore::new(1);

        let opts = ZKMCoreOpts::default();
        let element_threshold = opts.lde_size_threshold;
        let height_threshold = CORE_SHARD_HEIGHT_THRESHOLD;

        let mut driver = ShardDriver::new(program.clone(), opts.shard_size as u64);
        let mut state = PublicValues::<u32, u32>::default().reset();
        let mut deferred = ExecutionRecord::new(program.clone());
        let mut index = 0u32;
        let mut record_num = 0usize;
        loop {
            let (spliced, done) =
                next_shard(&mut driver, program.clone(), element_threshold, height_threshold).unwrap();
            let mut record =
                trace_shard(program.clone(), &spliced, driver.max_syscall_cycles())
                    .unwrap();
            if done {
                let (init, fin) = driver.global_memory_events();
                record.global_memory_initialize_events = init;
                record.global_memory_finalize_events = fin;
            }

            state.execution_shard = index + 1;
            state.is_execution_shard = record.contains_cpu() as u32;
            if let Some(first_pc) = record.first_instruction_pc {
                state.start_pc = first_pc;
                state.next_pc = record.last_next_pc;
                let first_clk = record.first_instruction_clk.unwrap();
                state.initial_clk_high = (first_clk >> 24) as u32;
                state.initial_clk_low = (first_clk & 0xFFFFFF) as u32;
                let last_clk_high = record.last_instruction_clk >> 24;
                state.last_clk_high = last_clk_high as u32;
                state.last_clk_low = (record.last_timestamp - (last_clk_high << 24)) as u32;
            }
            state.committed_value_digest = record.public_values.committed_value_digest;
            state.deferred_proofs_digest = record.public_values.deferred_proofs_digest;
            record.public_values = state;

            deferred.append(&mut record.defer());
            let mut records = vec![record];
            let mut split_records = deferred.split(done, records.last_mut(), opts.split_opts);

            if !done {
                state.execution_shard += 1;
            }
            for split_record in &mut split_records {
                state.is_execution_shard = 0;
                state.previous_init_addr = split_record.public_values.previous_init_addr;
                state.last_init_addr = split_record.public_values.last_init_addr;
                state.previous_finalize_addr = split_record.public_values.previous_finalize_addr;
                state.last_finalize_addr = split_record.public_values.last_finalize_addr;
                state.start_pc = state.next_pc;
                state.initial_clk_high = state.last_clk_high;
                state.initial_clk_low = state.last_clk_low;
                split_record.public_values = state;
            }
            records.append(&mut split_records);

            for mut record in records {
                shard_prover.machine().generate_dependencies(std::iter::once(&mut record), None).unwrap();

                let main_trace_data = async_rt.block_on(shard_prover.trace_generator().generate_main_traces(
                    record.clone(),
                    shard_prover.max_log_row_count(),
                    prover_permits.clone(),
                ));
                let shard_data = zkm_hypercube::prover::ShardData { pk: Arc::clone(&pk), main_trace_data };

                let mut prove_challenger = ZkmGlobalContext::default_challenger();
                shard_data.pk.vk.observe_into(&mut prove_challenger);
                let (proof, _permit) = shard_prover.prove_shard_with_data(shard_data, prove_challenger);

                let mut verify_challenger = ZkmGlobalContext::default_challenger();
                vk.observe_into(&mut verify_challenger);
                let result = shard_verifier.verify_shard(&vk, &proof, &mut verify_challenger);
                println!(
                    "record {record_num} (checkpoint {index}, is_execution_shard={}, contains_cpu={}): {}",
                    record.public_values.is_execution_shard,
                    record.contains_cpu(),
                    if result.is_ok() { "OK".to_string() } else { format!("FAILED: {:?}", result.unwrap_err()) }
                );
                record_num += 1;
            }

            if done {
                break;
            }
            index += 1;
        }
    }

    /// Same shape as `debug_serial_prove_verify_secp256r1_add`, but over `sha3_chain_program`
    /// (plain ALU/branch/load-store instructions only, no precompile syscalls) with a tiny
    /// `shard_size` to force many shards -- diagnostic for the `GkrVerificationFailed
    /// (CumulativeSumMismatch)` found via a real multi-shard SDK proof
    /// (`zkm_sdk::tests::test_e2e_core_many_shards`), to pinpoint exactly which record/shard
    /// boundary the imbalance first appears at.
    #[test]
    #[ignore = "manual diagnostic, run explicitly while investigating a GkrVerificationFailed mismatch"]
    fn debug_serial_prove_verify_sha3_chain_many_shards() {
        use std::sync::Arc;
        use zkm_hypercube::{
            air::PublicValues,
            prover::{ProverSemaphore, ZkmShardProver},
            ShardVerifier,
        };

        let program = Arc::new(crate::programs::tests::sha3_chain_program());

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        let max_log_row_count = CORE_MAX_LOG_ROW_COUNT;
        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            CORE_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        let shard_prover = ZkmShardProver::<MipsAir<KoalaBear>>::new(shard_verifier.clone());
        let async_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (preprocessed, vk) =
            async_rt.block_on(shard_prover.setup(program.clone(), ProverSemaphore::new(1)));
        let pk = Arc::new(preprocessed.pk);
        let prover_permits = ProverSemaphore::new(1);

        let mut opts = ZKMCoreOpts::default();
        opts.shard_size = 4096;
        let element_threshold = opts.lde_size_threshold;
        let height_threshold = CORE_SHARD_HEIGHT_THRESHOLD;

        let mut driver = ShardDriver::new(program.clone(), opts.shard_size as u64);
        let mut state = PublicValues::<u32, u32>::default().reset();
        let mut deferred = ExecutionRecord::new(program.clone());
        let mut index = 0u32;
        let mut record_num = 0usize;
        loop {
            let (spliced, done) =
                next_shard(&mut driver, program.clone(), element_threshold, height_threshold).unwrap();
            let mut record =
                trace_shard(program.clone(), &spliced, driver.max_syscall_cycles())
                    .unwrap();
            if done {
                let (init, fin) = driver.global_memory_events();
                record.global_memory_initialize_events = init;
                record.global_memory_finalize_events = fin;
            }

            if record.contains_cpu() {
                state.execution_shard += 1;
            }
            state.is_execution_shard = record.contains_cpu() as u32;
            if let Some(first_pc) = record.first_instruction_pc {
                state.start_pc = first_pc;
                state.next_pc = record.last_next_pc;
                let first_clk = record.first_instruction_clk.unwrap();
                state.initial_clk_high = (first_clk >> 24) as u32;
                state.initial_clk_low = (first_clk & 0xFFFFFF) as u32;
                let last_clk_high = record.last_instruction_clk >> 24;
                state.last_clk_high = last_clk_high as u32;
                state.last_clk_low = (record.last_timestamp - (last_clk_high << 24)) as u32;
            }
            state.committed_value_digest = record.public_values.committed_value_digest;
            state.deferred_proofs_digest = record.public_values.deferred_proofs_digest;
            record.public_values = state;

            deferred.append(&mut record.defer());
            let mut records = vec![record];
            let mut split_records = deferred.split(done, records.last_mut(), opts.split_opts);

            for split_record in &mut split_records {
                state.is_execution_shard = 0;
                state.previous_init_addr = split_record.public_values.previous_init_addr;
                state.last_init_addr = split_record.public_values.last_init_addr;
                state.previous_finalize_addr = split_record.public_values.previous_finalize_addr;
                state.last_finalize_addr = split_record.public_values.last_finalize_addr;
                state.start_pc = state.next_pc;
                state.initial_clk_high = state.last_clk_high;
                state.initial_clk_low = state.last_clk_low;
                split_record.public_values = state;
            }
            records.append(&mut split_records);

            for mut record in records {
                shard_prover.machine().generate_dependencies(std::iter::once(&mut record), None).unwrap();

                let main_trace_data = async_rt.block_on(shard_prover.trace_generator().generate_main_traces(
                    record.clone(),
                    shard_prover.max_log_row_count(),
                    prover_permits.clone(),
                ));
                let shard_data = zkm_hypercube::prover::ShardData { pk: Arc::clone(&pk), main_trace_data };

                let mut prove_challenger = ZkmGlobalContext::default_challenger();
                shard_data.pk.vk.observe_into(&mut prove_challenger);
                let (proof, _permit) = shard_prover.prove_shard_with_data(shard_data, prove_challenger);

                let mut verify_challenger = ZkmGlobalContext::default_challenger();
                vk.observe_into(&mut verify_challenger);
                let result = shard_verifier.verify_shard(&vk, &proof, &mut verify_challenger);
                println!(
                    "record {record_num} (checkpoint {index}, is_execution_shard={}, contains_cpu={}, execution_shard={}): {}",
                    record.public_values.is_execution_shard,
                    record.contains_cpu(),
                    record.public_values.execution_shard,
                    if result.is_ok() { "OK".to_string() } else { format!("FAILED: {:?}", result.unwrap_err()) }
                );
                record_num += 1;
            }

            if done {
                break;
            }
            index += 1;
        }
    }

    /// Differential counterpart to `debug_serial_prove_verify_secp256r1_add`: same serial
    /// generate_dependencies+generate_main_traces+prove_shard_with_data+verify_shard harness and
    /// the same defer/split/state-threading logic, but starting from the *legacy* `Executor`'s
    /// own record instead of the new MinimalExecutor/ShardDriver pipeline's. If this passes while
    /// the new-pipeline version fails, the bug is confirmed to be in the new pipeline's record
    /// construction (`TracingVM`/`MinimalExecutor`), not in the shared prover/verifier code both
    /// versions call identically.
    #[test]
    #[ignore = "manual diagnostic, run explicitly while investigating a GkrVerificationFailed mismatch"]
    fn debug_serial_prove_verify_secp256r1_add_legacy_executor() {
        use std::sync::Arc;
        use zkm_hypercube::{
            air::PublicValues,
            prover::{ProverSemaphore, ZkmShardProver},
            ShardVerifier,
        };

        let program = Arc::new(crate::programs::tests::secp256r1_add_program());

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        let max_log_row_count = CORE_MAX_LOG_ROW_COUNT;
        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            CORE_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        let shard_prover = ZkmShardProver::<MipsAir<KoalaBear>>::new(shard_verifier.clone());
        let async_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (preprocessed, vk) =
            async_rt.block_on(shard_prover.setup(program.clone(), ProverSemaphore::new(1)));
        let pk = Arc::new(preprocessed.pk);
        let prover_permits = ProverSemaphore::new(1);

        let opts = ZKMCoreOpts::default();

        let mut runtime = Executor::new(Program::clone(&program), opts);
        runtime.run().unwrap();
        let mut legacy_records = runtime.records;
        println!("legacy executor produced {} checkpoint record(s)", legacy_records.len());

        let mut state = PublicValues::<u32, u32>::default().reset();
        let mut deferred = ExecutionRecord::new(program.clone());
        let mut record_num = 0usize;
        let num_checkpoints = legacy_records.len();
        for (index, mut record) in legacy_records.drain(..).enumerate() {
            let done = index + 1 == num_checkpoints;

            state.execution_shard = record.public_values.execution_shard;
            state.is_execution_shard = record.contains_cpu() as u32;
            if let Some(first_pc) = record.first_instruction_pc {
                state.start_pc = first_pc;
                state.next_pc = record.last_next_pc;
                let first_clk = record.first_instruction_clk.unwrap();
                state.initial_clk_high = (first_clk >> 24) as u32;
                state.initial_clk_low = (first_clk & 0xFFFFFF) as u32;
                let last_clk_high = record.last_instruction_clk >> 24;
                state.last_clk_high = last_clk_high as u32;
                state.last_clk_low = (record.last_timestamp - (last_clk_high << 24)) as u32;
            }
            state.committed_value_digest = record.public_values.committed_value_digest;
            state.deferred_proofs_digest = record.public_values.deferred_proofs_digest;
            record.public_values = state;

            deferred.append(&mut record.defer());
            let mut records = vec![record];
            let mut split_records = deferred.split(done, records.last_mut(), opts.split_opts);

            if !done {
                state.execution_shard += 1;
            }
            for split_record in &mut split_records {
                state.is_execution_shard = 0;
                state.previous_init_addr = split_record.public_values.previous_init_addr;
                state.last_init_addr = split_record.public_values.last_init_addr;
                state.previous_finalize_addr = split_record.public_values.previous_finalize_addr;
                state.last_finalize_addr = split_record.public_values.last_finalize_addr;
                state.start_pc = state.next_pc;
                state.initial_clk_high = state.last_clk_high;
                state.initial_clk_low = state.last_clk_low;
                split_record.public_values = state;
            }
            records.append(&mut split_records);

            for mut record in records {
                shard_prover.machine().generate_dependencies(std::iter::once(&mut record), None).unwrap();

                let main_trace_data = async_rt.block_on(shard_prover.trace_generator().generate_main_traces(
                    record.clone(),
                    shard_prover.max_log_row_count(),
                    prover_permits.clone(),
                ));
                let shard_data = zkm_hypercube::prover::ShardData { pk: Arc::clone(&pk), main_trace_data };

                let mut prove_challenger = ZkmGlobalContext::default_challenger();
                shard_data.pk.vk.observe_into(&mut prove_challenger);
                let (proof, _permit) = shard_prover.prove_shard_with_data(shard_data, prove_challenger);

                let mut verify_challenger = ZkmGlobalContext::default_challenger();
                vk.observe_into(&mut verify_challenger);
                let result = shard_verifier.verify_shard(&vk, &proof, &mut verify_challenger);
                println!(
                    "record {record_num} (checkpoint {index}, is_execution_shard={}, contains_cpu={}): {}",
                    record.public_values.is_execution_shard,
                    record.contains_cpu(),
                    if result.is_ok() { "OK".to_string() } else { format!("FAILED: {:?}", result.unwrap_err()) }
                );
                record_num += 1;
            }
        }
    }
}
