use crate::mips::MipsAir;
// Used by the commented-out run_test_machine(_with_prover) below.
// use p3_uni_stark::SymbolicAirBuilder;
// use size::Size;
use hashbrown::HashMap;
use std::thread::ScopedJoinHandle;
use std::{
    io::{self},
    sync::{mpsc::sync_channel, Arc, Mutex},
};
use thiserror::Error;
use web_time::Instant;

use p3_koala_bear::KoalaBear;

use crate::io::ZKMStdin;
use zkm_core_executor::{
    mips_costs,
    minimal::{Chunk, MinimalRunner},
    splicing::{SplicedChunk, SplicingVM},
    tracing_chunk::{self, TracingVM},
    ExecutionError, ExecutionRecord, Executor, MipsAirId, Program, ZKMContext,
};
use zkm_primitives::io::ZKMPublicValues;

use slop_challenger::IopCtx;
use zkm_hypercube::{
    air::PublicValues,
    config::{default_fri_config, ZkmGlobalContext},
    prover::{AirProver, PcsProof, ProverSemaphore, ZkmShardProver},
    record::MachineRecord,
    ShardProof, ShardVerifier, ZkmSC,
};

/// The concrete shard-proof type produced by Ziren's own (`KoalaBear`, jagged/basefold) shard
/// prover.
pub type ZkmShardProof = ShardProof<
    zkm_hypercube::config::ZkmGlobalContext,
    PcsProof<zkm_hypercube::config::ZkmGlobalContext, ZkmSC<MipsAir<KoalaBear>>>,
>;


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
    #[error("io error: {0}")]
    IoError(io::Error),
    #[error("serialization error: {0}")]
    SerializationError(bincode::Error),
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
    // Per-chip byte costs, fed into `SplicingVM`'s own shard-cut cost model
    // (`estimate_mips_lde_size`, used by `should_cut_shard`) via Stage B below.
    let trace_byte_costs: Arc<HashMap<MipsAirId, u64>> =
        Arc::new(mips_costs().into_iter().map(|(k, v)| (k, v as u64)).collect());

    let setup_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
    let program_arc = Arc::new(program.clone());
    let (preprocessed, vk) =
        setup_rt.block_on(shard_prover.setup(program_arc.clone(), ProverSemaphore::new(1)));
    let pk = preprocessed.pk;

    // `generate_records`'s `MinimalRunner` setup. `stdin.proofs` (deferred-proof witnesses for
    // `VERIFY_ZKM_PROOF`/`SYSVERIFY`) is not yet wired into `MinimalRunner`; see
    // `SyscallRuntime::verify_deferred_proof`'s doc comment.
    let mut minimal_runner =
        MinimalRunner::new(program_arc.clone(), opts.minimal_trace_chunk_threshold);
    for buf in &stdin.buffer {
        minimal_runner.with_input(buf);
    }
    let context = context;
    let _ = context; // `ZKMContext`'s subproof verifier is likewise not yet threaded through.

    // Record the start of the process.
    let proving_start = Instant::now();
    let span = tracing::Span::current().clone();
    std::thread::scope(move |s| {
        let _span = span.enter();

        let (p2_records_tx, p2_records_rx) =
            sync_channel::<ExecutionRecord>(opts.records_and_traces_channel_capacity);

        // Stage A (raw execution): `MinimalRunner`'s own sequential walk. It never builds typed
        // events, so there's no event-construction work here to parallelize -- this stays cheap
        // and single-threaded, dispatching each `Chunk` round-robin to Stage B's splicing
        // workers.
        let splicing_worker_count = opts.splicing_workers.max(1);
        let mut splicing_chunk_txs = Vec::with_capacity(splicing_worker_count);
        let mut splicing_chunk_rxs = Vec::with_capacity(splicing_worker_count);
        for _ in 0..splicing_worker_count {
            let (tx, rx) = sync_channel::<Chunk>(1);
            splicing_chunk_txs.push(tx);
            splicing_chunk_rxs.push(rx);
        }
        // Stage C (see below) needs `minimal_runner`'s final registers/memory for `emit_globals`
        // once it reaches the last shard -- sent wholesale (not cloned; `PagedMemory` can be
        // large) once Stage A's loop finishes with it.
        let (final_runner_tx, final_runner_rx) = sync_channel::<MinimalRunner>(1);

        let stage_a_span = tracing::Span::current().clone();
        let stage_a_handle: ScopedJoinHandle<Result<(Vec<u8>, u64), ZKMCoreProverError>> =
            s.spawn(move || {
                let _span = stage_a_span.enter();
                tracing::debug_span!("minimal_execution").in_scope(|| {
                    let mut chunk_index = 0usize;
                    let mut cycles = 0u64;
                    loop {
                        let Some(chunk) = minimal_runner
                            .try_next_chunk()
                            .map_err(ZKMCoreProverError::ExecutionError)?
                        else {
                            break;
                        };
                        let chunk_done = chunk.done;
                        splicing_chunk_txs[chunk_index % splicing_worker_count]
                            .send(chunk)
                            .unwrap();
                        chunk_index += 1;
                        if chunk_done {
                            cycles = minimal_runner.global_clk();
                            break;
                        }
                    }
                    let public_values_stream = minimal_runner.public_values_stream().to_vec();
                    final_runner_tx.send(minimal_runner).unwrap();
                    Ok((public_values_stream, cycles))
                })
            });

        // Stage B (splicing): decides real shard cuts. Each worker replays `Chunk`s
        // independently via its own `SplicingVM` config clone (`SplicingVM` carries no state
        // across calls -- see its module doc comment for why that's safe: every chunk boundary
        // forces a shard cut if the cost threshold hasn't naturally been hit yet). Dispatched
        // round-robin from Stage A and consumed round-robin by Stage C below, so static
        // partitioning keeps chunk order aligned on both ends without a reorder buffer -- each
        // worker's own input and output channels stay in the order it received/produced them.
        let mut splicing_result_txs = Vec::with_capacity(splicing_worker_count);
        let mut splicing_result_rxs = Vec::with_capacity(splicing_worker_count);
        for _ in 0..splicing_worker_count {
            let (tx, rx) = sync_channel::<(Vec<SplicedChunk>, bool)>(1);
            splicing_result_txs.push(tx);
            splicing_result_rxs.push(rx);
        }

        let stage_b_span = tracing::Span::current().clone();
        let mut stage_b_handles = Vec::with_capacity(splicing_worker_count);
        for (chunk_rx, result_tx) in splicing_chunk_rxs.into_iter().zip(splicing_result_txs) {
            let span = stage_b_span.clone();
            let splicing_program = program_arc.clone();
            let splicing_trace_byte_costs = Arc::clone(&trace_byte_costs);
            let max_syscall_cycles = 0; // see `ZKMCoreOpts::minimal_trace_chunk_threshold`'s doc comment
            let shard_size = (opts.shard_size as u32) * 4;
            let lde_size_threshold = opts.lde_size_threshold;
            let handle: ScopedJoinHandle<Result<(), ZKMCoreProverError>> = s.spawn(move || {
                let _span = span.enter();
                tracing::debug_span!("splicing").in_scope(|| {
                    let splicing = SplicingVM::new(
                        splicing_program,
                        max_syscall_cycles,
                        shard_size,
                        lde_size_threshold,
                        (*splicing_trace_byte_costs).clone(),
                    );
                    loop {
                        let Ok(chunk) = chunk_rx.recv() else { break };
                        let chunk_done = chunk.done;
                        let spliced = splicing
                            .splice_chunk(chunk)
                            .map_err(ZKMCoreProverError::ExecutionError)?;
                        result_tx.send((spliced, chunk_done)).unwrap();
                    }
                    Ok(())
                })
            });
            stage_b_handles.push(handle);
        }

        // Stage C (dispatch): reads Stage B's results round-robin (the same order Stage A
        // dispatched chunks in) and assigns each `SplicedChunk` its true global shard index --
        // `SplicedChunk::shard` coming out of Stage B is only chunk-local (workers don't know the
        // global count -- see its doc comment), and this must happen before tracing, since
        // `TracingVM::new()` bakes `chunk.shard` into every typed event at construction time.
        // Otherwise just forwards each shard-patched `SplicedChunk` round-robin to Stage C'.
        let tracing_worker_count = opts.tracing_workers.max(1);
        let mut tracing_chunk_txs = Vec::with_capacity(tracing_worker_count);
        let mut tracing_chunk_rxs = Vec::with_capacity(tracing_worker_count);
        for _ in 0..tracing_worker_count {
            let (tx, rx) = sync_channel::<SplicedChunk>(1);
            tracing_chunk_txs.push(tx);
            tracing_chunk_rxs.push(rx);
        }

        let stage_c_span = tracing::Span::current().clone();
        let stage_c_handle: ScopedJoinHandle<Result<(), ZKMCoreProverError>> = s.spawn(move || {
            let _span = stage_c_span.enter();
            tracing::debug_span!("shard_dispatch").in_scope(|| {
                // Starts at 1, not 0: matches `CoreVM::new()`'s own `current_shard: 1` convention,
                // which is what `SplicingVM`'s shard numbering used before this stage split.
                let mut execution_shard_counter = 1u32;
                let mut chunk_index = 0usize;
                let mut dispatch_index = 0usize;

                loop {
                    let Ok((spliced_pieces, chunk_done)) =
                        splicing_result_rxs[chunk_index % splicing_worker_count].recv()
                    else {
                        break;
                    };
                    chunk_index += 1;

                    for mut spliced in spliced_pieces {
                        spliced.shard = execution_shard_counter;
                        execution_shard_counter += 1;
                        tracing_chunk_txs[dispatch_index % tracing_worker_count]
                            .send(spliced)
                            .unwrap();
                        dispatch_index += 1;
                    }

                    if chunk_done {
                        break;
                    }
                }
                Ok(())
            })
        });

        // Stage C' (tracing): builds the real typed `ExecutionRecord` for each shard. Each
        // worker replays a `SplicedChunk` independently -- `TracingVM::new()` needs nothing
        // threaded in from a previous shard's trace call (`HINT_LEN`/`HINT_READ` resolve against
        // the same oracle stream as everything else now, see `MinimalExecutor::resolve_hint_len`'s
        // doc comment) -- so this is safe the same way Stage B's worker pool is. Dispatched
        // round-robin from Stage C and consumed round-robin by Stage D below, same reasoning as
        // every other round-robin handoff in this pipeline.
        let mut tracing_result_txs = Vec::with_capacity(tracing_worker_count);
        let mut tracing_result_rxs = Vec::with_capacity(tracing_worker_count);
        for _ in 0..tracing_worker_count {
            let (tx, rx) = sync_channel::<tracing_chunk::TracedShard>(1);
            tracing_result_txs.push(tx);
            tracing_result_rxs.push(rx);
        }

        let stage_c_prime_span = tracing::Span::current().clone();
        let mut stage_c_prime_handles = Vec::with_capacity(tracing_worker_count);
        for (chunk_rx, result_tx) in tracing_chunk_rxs.into_iter().zip(tracing_result_txs) {
            let span = stage_c_prime_span.clone();
            let tracing_program = program_arc.clone();
            let handle: ScopedJoinHandle<Result<(), ZKMCoreProverError>> = s.spawn(move || {
                let _span = span.enter();
                tracing::debug_span!("tracing").in_scope(|| {
                    loop {
                        let Ok(spliced) = chunk_rx.recv() else { break };
                        let tracer = TracingVM::new(tracing_program.clone(), spliced);
                        let traced = tracer.trace().map_err(ZKMCoreProverError::ExecutionError)?;
                        result_tx.send(traced).unwrap();
                    }
                    Ok(())
                })
            });
            stage_c_prime_handles.push(handle);
        }

        // Stage D (state/deferred bookkeeping): reads Stage C's traced shards round-robin (the
        // same order Stage C dispatched them in) and finalizes each record's `public_values`.
        // Stays sequential -- `deferred`/`state`'s cross-shard chaining is order-dependent (see
        // the long comment below), so this can't be split across a worker pool the way Stage C'
        // was. Feeds the existing Stage 2 pool (dependency generation/trace generation/proving)
        // via `p2_records_tx`, unchanged from before this stage split.
        let stage_d_span = tracing::Span::current().clone();
        let reducer_program = program_arc.clone();
        let stage_d_handle: ScopedJoinHandle<Result<(), ZKMCoreProverError>> = s.spawn(move || {
            let _span = stage_d_span.enter();
            tracing::debug_span!("bookkeeping").in_scope(|| {
                let mut deferred = ExecutionRecord::new(reducer_program.clone());
                let mut state = PublicValues::<u32, u32>::default().reset();
                let mut result_index = 0usize;

                loop {
                    let Ok(traced) =
                        tracing_result_rxs[result_index % tracing_worker_count].recv()
                    else {
                        break;
                    };
                    result_index += 1;

                    let mut record = traced.record;
                    let done = traced.done;
                    // `TracingVM::new()` already set `record.public_values.shard` from the shard
                    // index Stage C patched onto its `SplicedChunk` before dispatch -- read it
                    // back here instead of needing it threaded separately.
                    let execution_shard = record.public_values.shard;

                    if done {
                        let minimal_runner_final = final_runner_rx.recv().unwrap();
                        tracing_chunk::emit_globals(
                            minimal_runner_final.registers(),
                            minimal_runner_final.registers_touched(),
                            minimal_runner_final.memory(),
                            minimal_runner_final.uninitialized_memory(),
                            minimal_runner_final.program(),
                            &mut record,
                        );
                    }

                    // Update the public values & prover state for this (execution)
                    // shard, then propagate to the record -- sequential, so (unlike the
                    // old multi-worker design) no turn-based synchronization is needed.
                    state.shard += 1;
                    state.execution_shard = execution_shard;
                    state.is_execution_shard = record.contains_cpu() as u32;
                    if let Some(first_pc) = record.first_instruction_pc {
                        state.start_pc = first_pc;
                        state.next_pc = record.last_next_pc;
                        let initial_clk = record.first_instruction_clk.unwrap();
                        // `clk_high` is constant across a shard (see
                        // `SplicingVM::should_cut_shard`'s window-boundary rule), so the
                        // shard's first row's high limb is this shard's only value.
                        state.clk_high = (initial_clk >> 28) as u32;
                        state.initial_timestamp = (initial_clk & 0xfff_ffff) as u32;
                        // Unlike `initial_timestamp`, deliberately relative to this shard's
                        // own `clk_high` window rather than masked to it: the AIR's
                        // `send_state` for the shard's last row predicts its successor's
                        // incoming clk as the unreduced expression `clk_low + clk_increment`
                        // (see `eval_state_chain`'s doc comment), which the window-boundary
                        // shard-cut rule allows to spill past `1 << 28` by up to one
                        // instruction's `clk_increment` exactly when it triggers the cut.
                        // `eval_public_values`'s closing `receive_state` must match that
                        // same unreduced value bit-for-bit -- masking to 28 bits would wrap
                        // it back into the shard's own window and never match.
                        state.last_timestamp =
                            (record.last_timestamp - (u64::from(state.clk_high) << 28))
                                as u32;
                    }
                    state.committed_value_digest = record.public_values.committed_value_digest;
                    state.deferred_proofs_digest = record.public_values.deferred_proofs_digest;
                    record.public_values = state;

                    // Defer events that are too expensive to include in every shard.
                    deferred.append(&mut record.defer());

                    let mut records = vec![record];
                    let mut split_records =
                        deferred.split(done, records.last_mut(), opts.split_opts);

                    // Update the public values & prover state for the shards which do
                    // not contain "cpu events" before committing to them.
                    if !done {
                        state.execution_shard += 1;
                    }
                    for split_record in &mut split_records {
                        state.shard += 1;
                        state.is_execution_shard = 0;
                        state.previous_init_addr_bits =
                            split_record.public_values.previous_init_addr_bits;
                        state.last_init_addr_bits = split_record.public_values.last_init_addr_bits;
                        state.previous_finalize_addr_bits =
                            split_record.public_values.previous_finalize_addr_bits;
                        state.last_finalize_addr_bits =
                            split_record.public_values.last_finalize_addr_bits;
                        state.start_pc = state.next_pc;
                        state.initial_timestamp = state.last_timestamp;
                        split_record.public_values = state;
                    }
                    records.append(&mut split_records);

                    // Dependency generation, trace-matrix generation, and proving
                    // (Stage 2, below) are all per-record-independent once
                    // `public_values` is finalized -- `Machine::generate_dependencies`
                    // allocates a fresh scratch record per (record, chip) and only ever
                    // reads/writes that one record, with no shared or cross-record
                    // accumulator. Hand each record off immediately so Stage 2 can work
                    // on shard N's dependency generation/trace generation/proving
                    // concurrently with this thread producing and tracing shard N+1
                    // onward.
                    for record in records {
                        p2_records_tx.send(record).unwrap();
                    }

                    if done {
                        break;
                    }
                }

                Ok(())
            })
        });

        // Stage 2 (dependency generation, trace-matrix generation, and shard proving): a fixed
        // pool of `prove_workers` threads -- a separate knob from `trace_gen_workers` (see
        // `ZKMCoreOpts::prove_workers`'s doc comment), since trace generation and shard proving
        // are different workloads with different scaling characteristics (trace generation is
        // lighter/more memory-bound; shard proving's `commit_traces` step is heavier/more
        // CPU-bound). The receiver is wrapped in `Arc<Mutex<_>>` and shared across `prove_workers`
        // threads that each lock only for the brief `recv()`, so multiple shards' work (each
        // itself already using rayon internally, e.g. some chips' own `generate_dependencies`)
        // can run concurrently, sharing rayon's global thread pool rather than competing with
        // each other for whole worker threads. Shards finish out of order across workers, so tag
        // each with its `ExecutionRecord`'s `shard` index and sort by it afterward -- downstream
        // verification requires proofs in strictly increasing shard order.
        let p2_prover_span = tracing::Span::current().clone();
        let p2_records_rx = Arc::new(Mutex::new(p2_records_rx));
        let all_shard_proofs_unordered = Arc::new(Mutex::new(Vec::new()));
        let mut p2_prover_handles = Vec::new();
        for _ in 0..opts.prove_workers.max(1) {
            let span = p2_prover_span.clone();
            let machine = shard_prover.machine().clone();
            let shard_prover = Arc::clone(&shard_prover);
            let rx = Arc::clone(&p2_records_rx);
            let all_shard_proofs_unordered = Arc::clone(&all_shard_proofs_unordered);
            let pk = Arc::clone(&pk);
            let prover_permits = prover_permits.clone();
            let handle: ScopedJoinHandle<Result<(), ZKMCoreProverError>> = s.spawn(move || {
                let _span = span.enter();
                tracing::debug_span!("shard prover").in_scope(|| -> Result<(), ZKMCoreProverError> {
                    let async_rt =
                        tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
                    loop {
                        let received = { rx.lock().unwrap().recv() };
                        let Ok(mut record) = received else {
                            break;
                        };

                        machine.generate_dependencies(std::iter::once(&mut record), None).map_err(
                            |e| {
                                tracing::error!("Error generating dependencies: {:?}", e);
                                ZKMCoreProverError::DependenciesGenerationError
                            },
                        )?;

                        let shard_index = record.public_values.shard;
                        let shard_start = Instant::now();
                        let (proof, _permit) = async_rt.block_on(shard_prover.prove_shard_with_pk(
                            Arc::clone(&pk),
                            record,
                            prover_permits.clone(),
                        ));
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
                    Ok(())
                })
            });
            p2_prover_handles.push(handle);
        }

        // Wait until Stages A/B/C/C'/D have fully finished.
        let (public_values_stream, cycles) = stage_a_handle.join().unwrap()?;
        for handle in stage_b_handles {
            handle.join().unwrap()?;
        }
        stage_c_handle.join().unwrap()?;
        for handle in stage_c_prime_handles {
            handle.join().unwrap()?;
        }
        stage_d_handle.join().unwrap()?;

        // Wait until all shard-proving workers have finished, then restore shard order.
        for handle in p2_prover_handles {
            handle.join().unwrap()?;
        }
        let mut all_shard_proofs_unordered = match Arc::try_unwrap(all_shard_proofs_unordered) {
            Ok(mutex) => mutex.into_inner().unwrap(),
            Err(_) => panic!("all_shard_proofs_unordered still has outstanding references"),
        };
        all_shard_proofs_unordered.sort_by_key(|(shard_index, _)| *shard_index);
        let all_shard_proofs: Vec<_> =
            all_shard_proofs_unordered.into_iter().map(|(_, proof)| proof).collect();

        // NOTE: a per-opcode/per-syscall `ExecutionReport` breakdown is not produced here, since
        // `MinimalRunner`/`SplicingVM`/`TracingVM` don't duplicate `Executor::report`'s
        // bookkeeping. Cycle count alone is still exact, from `MinimalRunner::global_clk()`.

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
    // A successful run always retires at least one shard; zero shards means the producer
    // pipeline silently never executed anything (verified nothing below), not a real pass.
    assert!(!shard_proofs.is_empty(), "prove_with_context produced zero shards");

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
        halt_only_program, hello_world_program, simple_memory_program, simple_program,
    };

    /// Drives the same `MinimalRunner` -> `SplicingVM` -> `TracingVM` pipeline
    /// `prove_with_context` uses, then checks local-scope interaction balance per record via
    /// `debug_interactions_with_all_chips` -- much faster than a full GKR proof, and pinpoints
    /// which `LookupKind` is imbalanced instead of just "verification failed".
    fn debug_generate_records_interactions_balance(program: Program) -> bool {
        use p3_koala_bear::KoalaBear;
        use slop_air::BaseAir;
        use slop_multilinear::{Mle, PaddedMle};
        use std::sync::Arc;
        use zkm_core_executor::{minimal::MinimalRunner, splicing::SplicingVM, tracing_chunk::TracingVM};
        use zkm_hypercube::{
            air::{LookupScope, MachineAir},
            lookup::{debug_interactions_with_all_chips, LookupKind},
            prover::Traces,
            record::MachineRecord,
        };

        let opts = ZKMCoreOpts::default();
        // `Executor::new` normalizes the raw program (e.g. appends a halt safety net) --
        // `run_test_core` always runs this first (`Program::clone(&runtime.program)`), and a bare
        // `simple_program()`-style program with no explicit halt instruction relies on it.
        let program = Executor::new(program, opts.clone()).program;
        let mut minimal_runner =
            MinimalRunner::new(program.clone(), opts.minimal_trace_chunk_threshold);
        let splicing = SplicingVM::new(
            program.clone(),
            0,
            (opts.shard_size as u32) * 4,
            opts.lde_size_threshold,
            mips_costs().into_iter().map(|(k, v)| (k, v as u64)).collect(),
        );
        let mut deferred = ExecutionRecord::new(program.clone());
        let mut state = PublicValues::<u32, u32>::default().reset();
        let mut all_records = Vec::new();

        loop {
            let Some(chunk) = minimal_runner.try_next_chunk().unwrap() else { break };
            let chunk_done = chunk.done;
            for spliced in splicing.splice_chunk(chunk).unwrap() {
                let execution_shard = spliced.shard;
                let tracer = TracingVM::new(program.clone(), spliced);
                let traced = tracer.trace().unwrap();
                let mut record = traced.record;
                let done = traced.done;

                if done {
                    tracing_chunk::emit_globals(
                        minimal_runner.registers(),
                        minimal_runner.registers_touched(),
                        minimal_runner.memory(),
                        minimal_runner.uninitialized_memory(),
                        minimal_runner.program(),
                        &mut record,
                    );
                }

                state.shard += 1;
                state.execution_shard = execution_shard;
                state.is_execution_shard = record.contains_cpu() as u32;
                if let Some(first_pc) = record.first_instruction_pc {
                    state.start_pc = first_pc;
                    state.next_pc = record.last_next_pc;
                    let initial_clk = record.first_instruction_clk.unwrap();
                    state.clk_high = (initial_clk >> 28) as u32;
                    state.initial_timestamp = (initial_clk & 0xfff_ffff) as u32;
                    state.last_timestamp =
                        (record.last_timestamp - (u64::from(state.clk_high) << 28)) as u32;
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
                    state.shard += 1;
                    state.is_execution_shard = 0;
                    state.previous_init_addr_bits = split_record.public_values.previous_init_addr_bits;
                    state.last_init_addr_bits = split_record.public_values.last_init_addr_bits;
                    state.previous_finalize_addr_bits =
                        split_record.public_values.previous_finalize_addr_bits;
                    state.last_finalize_addr_bits = split_record.public_values.last_finalize_addr_bits;
                    state.start_pc = state.next_pc;
                    state.initial_timestamp = state.last_timestamp;
                    split_record.public_values = state;
                }
                records.append(&mut split_records);
                all_records.append(&mut records);
            }
            if chunk_done {
                break;
            }
        }
        println!("produced {} record(s)", all_records.len());

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        machine.generate_dependencies(all_records.iter_mut(), None).unwrap();
        let chips = machine.chips().to_vec();
        let max_log_row_count = 22u32;

        let mut all_balanced = true;
        for (i, record) in all_records.into_iter().enumerate() {
            let mut preprocessed_named = std::collections::BTreeMap::new();
            let mut main_named = std::collections::BTreeMap::new();
            for chip in &chips {
                let chip_name = MachineAir::<KoalaBear>::name(chip);
                let pre_mle = match chip.generate_preprocessed_trace(&program) {
                    Some(t) => PaddedMle::padded_with_zeros(Arc::new(Mle::from(t)), max_log_row_count),
                    None => PaddedMle::zeros(0, max_log_row_count),
                };
                preprocessed_named.insert(chip_name.clone(), pre_mle);

                let main_mle = if chip.included(&record) {
                    let trace = chip.generate_trace(&record, &mut Default::default()).unwrap();
                    PaddedMle::padded_with_zeros(Arc::new(Mle::from(trace)), max_log_row_count)
                } else {
                    PaddedMle::zeros(BaseAir::<KoalaBear>::width(chip), max_log_row_count)
                };
                main_named.insert(chip_name, main_mle);
            }
            let preprocessed_traces = Traces { named_traces: preprocessed_named };
            let traces = Traces { named_traces: main_named };
            let public_values = record.public_values::<KoalaBear>();

            println!("== record {i} ==");
            let balanced = debug_interactions_with_all_chips(
                &chips,
                &preprocessed_traces,
                &traces,
                public_values,
                LookupKind::all_kinds(),
                LookupScope::Local,
            );
            all_balanced &= balanced;
            if !balanced {
                println!("record {i} is imbalanced -- stopping early instead of checking the rest");
                break;
            }
        }
        all_balanced
    }

    #[test]
    fn debug_generate_records_pipeline_balance_smoke() {
        assert!(
            debug_generate_records_interactions_balance(simple_program()),
            "local-scope send/receive interactions don't balance"
        );
    }

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

    /// Mirrors `examples/fibonacci/host/src/main.rs` exactly (real `n = 1000` written to stdin,
    /// read back by the guest via `zkm_zkvm::io::read`/`HINT_READ`), unlike
    /// `run_test_fibonacci_real_elf` above, which runs with empty stdin -- `HINT_READ` against an
    /// empty stream returns `0`, so that test's guest loop never actually executes.
    #[test]
    fn run_test_fibonacci_real_stdin() {
        let program = Program::from(test_artifacts::FIBONACCI_ELF).unwrap();
        let mut stdin = ZKMStdin::new();
        stdin.write(&1000u32);
        run_test_io(program, stdin).unwrap();
    }

    /// Forces many small shards (a tiny `shard_size` against thousands of repeated `ADD`s, each
    /// touching the same register), then proves and verifies all of them for real -- exercising
    /// the cross-shard memory-consistency argument (`MemoryLocalChip`/`MemoryGlobalChip`) far more
    /// densely than the other smoke tests here, which mostly stay within one shard. See
    /// `register-refresh-rolled-back` project memory: a structurally similar cross-shard change
    /// previously passed every local `debug_interactions` check yet failed real GKR verification,
    /// so this exercises `prove_with_context`'s real challenge-based protocol end to end rather
    /// than stopping at trace generation.
    #[test]
    fn run_test_many_small_shards() {
        use crate::programs::tests::many_adds_program;
        use zkm_core_executor::ZKMContext;
        use zkm_stark::ZKMCoreOpts;

        let program = many_adds_program(4096);
        let opts = ZKMCoreOpts { shard_size: 1 << 8, ..Default::default() };
        let (shard_proofs, _public_values_stream, _cycles, vk) =
            prove_with_context(program, &ZKMStdin::new(), opts, ZKMContext::default()).unwrap();

        assert!(shard_proofs.len() > 1, "expected more than one shard, got {}", shard_proofs.len());

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

    /// A tiny `minimal_trace_chunk_threshold` forces `MinimalRunner` to yield many `Chunk`s well
    /// before any shard's cost threshold is hit, so `SplicingVM::splice_chunk` must repeatedly
    /// force-close a shard at a chunk boundary rather than at a natural `should_cut_shard()` cut
    /// -- the new code path this stage split added. None of the other smoke tests here exercise
    /// it: `many_adds_program` never touches the oracle at all (pure register ops), and every
    /// other test's `minimal_trace_chunk_threshold` stays at the default (sized well above a
    /// typical shard's own oracle-entry count), so a shard always closes before its chunk does.
    #[test]
    fn run_test_chunk_boundary_mid_shard() {
        use crate::programs::tests::fibonacci_program;
        use zkm_core_executor::ZKMContext;
        use zkm_stark::ZKMCoreOpts;

        let program = fibonacci_program();
        // Small enough to force many chunks within fibonacci's own (small, likely single-shard)
        // execution, but not so small that per-chunk channel/replay overhead dominates runtime --
        // `shard_size` stays at its default so this doesn't also multiply shard count on top of
        // chunk count.
        let opts = ZKMCoreOpts { minimal_trace_chunk_threshold: 64, ..Default::default() };
        let (shard_proofs, _public_values_stream, _cycles, vk) =
            prove_with_context(program, &ZKMStdin::new(), opts, ZKMContext::default()).unwrap();

        assert!(!shard_proofs.is_empty());

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

    /// Combines real `HINT_LEN`/`HINT_READ` usage (via real stdin) with a tiny
    /// `minimal_trace_chunk_threshold`, so a synthetic hint-length oracle entry (see
    /// `MinimalExecutor::resolve_hint_len`'s doc comment) is likely to land at or near a forced
    /// chunk boundary. Neither `run_test_fibonacci_real_stdin` (real stdin, but default -- large
    /// -- chunk threshold, so it never crosses a chunk boundary) nor
    /// `run_test_chunk_boundary_mid_shard` (forces a chunk boundary, but with empty stdin, so it
    /// never calls `HINT_READ` at all) exercises both at once.
    #[test]
    fn run_test_hint_read_chunk_boundary() {
        use zkm_core_executor::ZKMContext;
        use zkm_stark::ZKMCoreOpts;

        let program = Program::from(test_artifacts::FIBONACCI_ELF).unwrap();
        let mut stdin = ZKMStdin::new();
        stdin.write(&1000u32);
        let opts = ZKMCoreOpts { minimal_trace_chunk_threshold: 64, ..Default::default() };
        let (shard_proofs, _public_values_stream, _cycles, vk) =
            prove_with_context(program, &stdin, opts, ZKMContext::default()).unwrap();

        assert!(!shard_proofs.is_empty());

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
}
