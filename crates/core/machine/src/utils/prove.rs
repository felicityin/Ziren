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

use crate::{io::ZKMStdin, utils::trace_budget};
use zkm_core_executor::{
    estimate_record_trace_bytes, mips_costs, minimal::MinimalRunner, splicing::SplicingVM,
    tracing_chunk::TracingVM, ExecutionError, ExecutionRecord, Executor, MipsAirId, Program,
    ZKMContext,
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
    // Per-chip byte costs, for estimating each record's real materialized trace size before
    // admitting it into the process-wide trace-memory budget (`trace_budget::acquire_trace_budget`).
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

        // `generate_records` producer: a single sequential pass through `MinimalRunner` ->
        // `SplicingVM` -> `TracingVM`, with no worker threads for record generation. Runs on its
        // own thread only so it can pipeline concurrently with the prover workers below via the
        // channel.
        let (p2_records_and_traces_tx, p2_records_and_traces_rx) = sync_channel::<(
            Vec<ExecutionRecord>,
            Vec<ZkmShardData>,
            Vec<ProverPermit>,
        )>(opts.records_and_traces_channel_capacity);

        let producer_span = tracing::Span::current().clone();
        let producer_program = program_arc.clone();
        let producer_machine = shard_prover.machine().clone();
        let producer_shard_prover = Arc::clone(&shard_prover);
        let producer_pk = Arc::clone(&pk);
        let producer_prover_permits = prover_permits.clone();
        let producer_trace_byte_costs = Arc::clone(&trace_byte_costs);
        let producer_handle: ScopedJoinHandle<Result<(Vec<u8>, u64), ZKMCoreProverError>> =
            s.spawn(move || {
                let _span = producer_span.enter();
                tracing::debug_span!("generate_records").in_scope(|| {
                    let async_rt =
                        tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();

                    let max_syscall_cycles = 0; // see `ZKMCoreOpts::minimal_trace_chunk_threshold`'s doc comment
                    let mut splicing = SplicingVM::new(
                        producer_program.clone(),
                        max_syscall_cycles,
                        (opts.shard_size as u32) * 4,
                        opts.shape_check_frequency,
                        opts.lde_size_threshold,
                        (*producer_trace_byte_costs).clone(),
                    );
                    // `SplicingVM`/`TracingVM` replay `HINT_LEN`/`HINT_READ` syscalls against
                    // this same stream (see `SyscallRuntime::seed_uninitialized`'s doc comment:
                    // only the *values* are oracle-carried, not the syscall's own bookkeeping),
                    // so they need the same stdin, consumed in the same order `minimal_runner`
                    // already consumed it in above.
                    for buf in &stdin.buffer {
                        splicing.with_input(buf);
                    }
                    let mut tracing_tags = HashMap::new();
                    let mut tracing_input_stream: std::collections::VecDeque<Vec<u8>> =
                        stdin.buffer.iter().cloned().collect();
                    let mut deferred = ExecutionRecord::new(producer_program.clone());
                    let mut state = PublicValues::<u32, u32>::default().reset();
                    let mut cycles = 0u64;

                    loop {
                        let Some(chunk) = minimal_runner
                            .try_next_chunk()
                            .map_err(ZKMCoreProverError::ExecutionError)?
                        else {
                            break;
                        };
                        let chunk_done = chunk.done;

                        let spliced_pieces = splicing
                            .splice_chunk(chunk)
                            .map_err(ZKMCoreProverError::ExecutionError)?;

                        for spliced in spliced_pieces {
                            let execution_shard = spliced.shard;
                            let tracer = TracingVM::new(
                                producer_program.clone(),
                                spliced,
                                std::mem::take(&mut tracing_tags),
                                std::mem::take(&mut tracing_input_stream),
                            );
                            let traced = tracer.trace().map_err(ZKMCoreProverError::ExecutionError)?;
                            tracing_tags = traced.tags;
                            tracing_input_stream = traced.input_stream;
                            let mut record = traced.record;
                            let done = traced.done;

                            if done {
                                minimal_runner.emit_globals(&mut record);
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

                            producer_machine.generate_dependencies(records.iter_mut(), None).map_err(
                                |e| {
                                    tracing::error!("Error generating dependencies: {:?}", e);
                                    ZKMCoreProverError::DependenciesGenerationError
                                },
                            )?;

                            // Generate each record's traces and send it to the shard-proving
                            // workers immediately, one record at a time -- see the long-form
                            // comment on this same pattern further down.
                            for record in records {
                                let estimated_bytes =
                                    estimate_record_trace_bytes(&record, &producer_trace_byte_costs);
                                let budget_permit =
                                    async_rt.block_on(trace_budget::acquire_trace_budget(estimated_bytes));
                                let main_trace_data =
                                    async_rt.block_on(producer_shard_prover.trace_generator().generate_main_traces(
                                        record.clone(),
                                        producer_shard_prover.max_log_row_count(),
                                        producer_prover_permits.clone(),
                                    ));
                                let shard_data =
                                    ZkmShardData { pk: Arc::clone(&producer_pk), main_trace_data };
                                p2_records_and_traces_tx
                                    .send((vec![record], vec![shard_data], vec![budget_permit]))
                                    .unwrap();
                            }
                        }

                        if chunk_done {
                            cycles = minimal_runner.global_clk();
                            break;
                        }
                    }

                    tracing::info!(
                        "generate_records finished at {:?} (cycles={})",
                        proving_start.elapsed(),
                        cycles,
                    );

                    Ok((minimal_runner.public_values_stream().to_vec(), cycles))
                })
            });
        // Spawn shard-proving worker threads, sized by `prove_workers` -- a separate knob from
        // `trace_gen_workers` (see `ZKMCoreOpts::prove_workers`'s doc comment), since trace
        // generation and shard proving are different workloads with different scaling
        // characteristics (trace generation is lighter/more memory-bound; shard proving's
        // `commit_traces` step is heavier/more CPU-bound). The receiver is wrapped in
        // `Arc<Mutex<_>>` and shared across `prove_workers` threads that each lock only for the
        // brief `recv()`, so multiple shards' `prove_shard_with_data` calls (each itself already
        // using rayon internally) can run concurrently, sharing rayon's global thread pool rather
        // than competing with each other for whole worker threads. Proofs finish out of order
        // across workers, so tag each with its `ExecutionRecord`'s `shard` index and sort by it
        // afterward -- downstream verification requires proofs in strictly increasing shard
        // order.
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
                tracing::debug_span!("shard prover").in_scope(|| loop {
                    let received = { rx.lock().unwrap().recv() };
                    let Ok((records, shard_data, budget_permits)) = received else {
                        break;
                    };
                    for ((record, data), budget_permit) in
                        records.into_iter().zip(shard_data).zip(budget_permits)
                    {
                        let shard_index = record.public_values.shard;
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

        // Wait until the sequential producer has fully finished.
        let (public_values_stream, cycles) = producer_handle.join().unwrap()?;

        // Wait until all shard-proving workers have finished, then restore shard order.
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
}
