use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use slop_air::BaseAir;
use slop_algebra::Field;
use slop_alloc::{Backend, CanCopyFrom, CpuBackend, GLOBAL_CPU_BACKEND};
use slop_multilinear::{Mle, PaddedMle};
use slop_tensor::Tensor;
use tokio::sync::oneshot;
use tracing::Instrument;

use crate::{air::MachineAir, record::MachineRecord, Chip, Machine};

use super::{ProverPermit, ProverSemaphore};

/// A collection of per-chip traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Tensor<F, B>: Serialize, F: Serialize, B: Serialize, "))]
#[serde(bound(deserialize = "Tensor<F, B>: Deserialize<'de>, F: Deserialize<'de>, B: Deserialize<'de>, "))]
pub struct Traces<F, B: Backend> {
    pub named_traces: BTreeMap<String, PaddedMle<F, B>>,
}

impl<F, B: Backend> IntoIterator for Traces<F, B> {
    type Item = (String, PaddedMle<F, B>);
    type IntoIter = <BTreeMap<String, PaddedMle<F, B>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.named_traces.into_iter()
    }
}

impl<F, B: Backend> Deref for Traces<F, B> {
    type Target = BTreeMap<String, PaddedMle<F, B>>;

    fn deref(&self) -> &Self::Target {
        &self.named_traces
    }
}

impl<F, B: Backend> DerefMut for Traces<F, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.named_traces
    }
}

/// The main traces for a program, with a permit.
pub struct MainTraceData<F: Field, A: MachineAir<F>, B: Backend> {
    /// The traces.
    pub traces: Traces<F, B>,
    /// The public values.
    pub public_values: Vec<F>,
    /// The shape cluster corresponding to the traces.
    pub shard_chips: BTreeSet<Chip<F, A>>,
    /// A permit for a prover resource.
    pub permit: ProverPermit,
}

/// The total trace data for a shard.
pub struct TraceData<F: Field, A: MachineAir<F>, B: Backend> {
    /// The preprocessed traces.
    pub preprocessed_traces: Traces<F, B>,
    /// The main traces.
    pub main_trace_data: MainTraceData<F, A, B>,
}

/// The preprocessed traces for a program.
pub struct PreprocessedTraceData<F: Field, B: Backend> {
    /// The preprocessed traces.
    pub preprocessed_traces: Traces<F, B>,
    /// A permit for a prover resource.
    pub permit: ProverPermit,
}

/// A trace generator for a given machine.
///
/// The trace generator is responsible for producing the preprocessed traces from a program and the
/// traces from an execution record.
pub trait TraceGenerator<F: Field, A: MachineAir<F>, B: Backend>: 'static + Send + Sync {
    /// Get a handle for the machine.
    fn machine(&self) -> &Machine<F, A>;

    /// Get the allocator for the traces.
    fn allocator(&self) -> &B;

    /// Generate the preprocessed traces for the given program.
    fn generate_preprocessed_traces(
        &self,
        program: Arc<A::Program>,
        max_log_row_count: usize,
        setup_permits: ProverSemaphore,
    ) -> impl Future<Output = PreprocessedTraceData<F, B>> + Send;

    /// Generate the main traces for the given execution record.
    fn generate_main_traces(
        &self,
        record: A::Record,
        max_log_row_count: usize,
        prover_permits: ProverSemaphore,
    ) -> impl Future<Output = MainTraceData<F, A, B>> + Send;

    /// Generate all traces for the given program and execution record.
    fn generate_traces(
        &self,
        program: Arc<A::Program>,
        record: A::Record,
        max_log_row_count: usize,
        prover_permits: ProverSemaphore,
    ) -> impl Future<Output = TraceData<F, A, B>> + Send;
}

/// A trace generator that used the default methods on chips for generating traces.
pub struct DefaultTraceGenerator<F: Field, A, B = CpuBackend> {
    machine: Machine<F, A>,
    trace_allocator: B,
}

impl<F: Field, A: MachineAir<F>, B: Backend> DefaultTraceGenerator<F, A, B> {
    /// Create a new trace generator.
    #[must_use]
    pub fn new_in(machine: Machine<F, A>, trace_allocator: B) -> Self {
        Self { machine, trace_allocator }
    }
}

impl<F: Field, A: MachineAir<F>> DefaultTraceGenerator<F, A, CpuBackend> {
    /// Create a new trace generator on the CPU.
    #[must_use]
    pub fn new(machine: Machine<F, A>) -> Self {
        Self { machine, trace_allocator: GLOBAL_CPU_BACKEND }
    }
}

impl<F: Field, A: MachineAir<F>> TraceGenerator<F, A, CpuBackend>
    for DefaultTraceGenerator<F, A, CpuBackend>
{
    fn machine(&self) -> &Machine<F, A> {
        &self.machine
    }

    fn allocator(&self) -> &CpuBackend {
        &self.trace_allocator
    }

    async fn generate_main_traces(
        &self,
        record: A::Record,
        max_log_row_count: usize,
        prover_permits: ProverSemaphore,
    ) -> MainTraceData<F, A, CpuBackend> {
        let airs = self.machine.chips().to_vec();
        let (tx, rx) = oneshot::channel();
        // Spawn a rayon task to generate the traces on the CPU.
        slop_futures::rayon::spawn(move || {
            let chips_and_traces = airs
                .into_par_iter()
                .filter(|air| air.included(&record))
                .map(|air| {
                    let trace = air.generate_trace(&record, &mut A::Record::default());
                    let trace = Mle::from(trace.unwrap());
                    (air, trace)
                })
                .collect::<BTreeMap<_, _>>();

            // Get the public values from the record.
            let public_values = record.public_values::<F>();

            tx.send((chips_and_traces, public_values)).ok().unwrap();
            // Emphasize that we are dropping the record after sending the traces.
            drop(record);
        });
        // Wait for the traces to be generated and copy them to the target backend.
        let (chips_and_traces, public_values) = rx.await.unwrap();

        let chip_set = chips_and_traces.keys().cloned().collect::<BTreeSet<_>>();
        let shard_chips = self.machine.smallest_cluster(&chip_set).unwrap().clone();

        // Wait for a prover to be available.
        let permit = prover_permits
            .acquire()
            .instrument(tracing::debug_span!("acquire prover"))
            .await
            .unwrap();
        // Copy the traces to the target backend.

        // Make the padded traces.
        let padded_traces = shard_chips
            .iter()
            .filter(|chip| !chips_and_traces.contains_key(*chip))
            .map(|chip| {
                let num_polynomials = chip.width();
                (
                    chip.name().to_string(),
                    PaddedMle::zeros(num_polynomials, max_log_row_count as u32),
                )
            })
            .collect::<BTreeMap<_, _>>();

        // Copy the real traces to the target backend.
        let real_traces = chips_and_traces
            .into_iter()
            .map(|(chip, trace)| {
                let trace = self.trace_allocator.copy_into(trace).unwrap();
                let mle = Arc::new(trace);
                (
                    chip.name().to_string(),
                    PaddedMle::padded_with_zeros(mle, max_log_row_count as u32),
                )
            })
            .collect::<Vec<_>>();

        let mut traces = padded_traces;

        for (name, trace) in real_traces {
            traces.insert(name, trace);
        }

        let traces = Traces { named_traces: traces };

        MainTraceData { traces, public_values, shard_chips, permit }
    }

    async fn generate_preprocessed_traces(
        &self,
        program: Arc<A::Program>,
        max_log_row_count: usize,
        setup_permits: ProverSemaphore,
    ) -> PreprocessedTraceData<F, CpuBackend> {
        // Generate the traces on the CPU.
        let airs = self.machine.chips().iter().map(|chip| chip.air.clone()).collect::<Vec<_>>();
        let (tx, rx) = oneshot::channel();
        // Spawn a rayon task to generate the traces on the CPU.
        slop_futures::rayon::spawn(move || {
            let named_preprocessed_traces = airs
                .par_iter()
                .filter_map(|air| {
                    let name = air.name().to_string();
                    let trace = air.generate_preprocessed_trace(&program);
                    trace.map(Mle::from).map(|tr| (name, tr))
                })
                .collect::<BTreeMap<_, _>>();
            tx.send(named_preprocessed_traces).ok().unwrap();
        });

        // Wait for the traces to be generated and copy them to the target backend.
        // Wait for traces.
        let named_preprocessed_traces = rx.await.unwrap();

        // Wait for a permit to be available to copy the traces to the target backend.
        let permit = setup_permits
            .acquire()
            .instrument(tracing::debug_span!("acquire setup"))
            .await
            .unwrap();

        // Copy the traces to the target backend.
        let named_traces = named_preprocessed_traces
            .into_iter()
            .map(|(name, trace)| {
                let trace = self.trace_allocator.copy_into(trace).unwrap();
                let padded_mle =
                    PaddedMle::padded_with_zeros(Arc::new(trace), max_log_row_count as u32);
                (name, padded_mle)
            })
            .collect::<BTreeMap<_, _>>();

        let traces = Traces { named_traces };

        PreprocessedTraceData { preprocessed_traces: traces, permit }
    }

    async fn generate_traces(
        &self,
        program: Arc<A::Program>,
        record: A::Record,
        max_log_row_count: usize,
        prover_permits: ProverSemaphore,
    ) -> TraceData<F, A, CpuBackend> {
        let airs = self.machine.chips().to_vec();
        let (tx, rx) = oneshot::channel();
        // Spawn a rayon task to generate the traces on the CPU.
        slop_futures::rayon::spawn(move || {
            let named_preprocessed_traces = airs
                .par_iter()
                .filter_map(|air| {
                    let name = air.name().to_string();
                    let trace = air.generate_preprocessed_trace(&program);
                    trace.map(Mle::from).map(|tr| (name, tr))
                })
                .collect::<BTreeMap<_, _>>();

            let chips_and_traces = airs
                .into_par_iter()
                .filter(|air| air.included(&record))
                .map(|air| {
                    let trace = air.generate_trace(&record, &mut A::Record::default());
                    let trace = Mle::from(trace.unwrap());
                    (air, trace)
                })
                .collect::<BTreeMap<_, _>>();

            // Get the public values from the record.
            let public_values = record.public_values::<F>();
            tx.send((named_preprocessed_traces, chips_and_traces, public_values)).ok().unwrap();
            // Emphasize that we are dropping the record after sending the traces.
            drop(record);
        });
        // Wait for the traces to be generated and copy them to the target backend.
        let (named_preprocessed_traces, chips_and_traces, public_values) = rx.await.unwrap();

        let chip_set = chips_and_traces.keys().cloned().collect::<BTreeSet<_>>();
        let shard_chips = self.machine.smallest_cluster(&chip_set).unwrap().clone();

        // Make the padded traces.
        let padded_traces = shard_chips
            .iter()
            .filter(|chip| !chips_and_traces.contains_key(*chip))
            .map(|chip| {
                let num_polynomials = chip.width();
                (
                    chip.name().to_string(),
                    PaddedMle::zeros(num_polynomials, max_log_row_count as u32),
                )
            })
            .collect::<BTreeMap<_, _>>();

        // Wait for a prover to be available.
        let permit = prover_permits
            .acquire()
            .instrument(tracing::debug_span!("acquire prover"))
            .await
            .unwrap();

        // Copy the preprocessed traces to the target backend.
        let preprocessed_traces = named_preprocessed_traces
            .into_iter()
            .map(|(name, trace)| {
                let trace = self.trace_allocator.copy_into(trace).unwrap();
                let padded_mle =
                    PaddedMle::padded_with_zeros(Arc::new(trace), max_log_row_count as u32);
                (name, padded_mle)
            })
            .collect::<BTreeMap<_, _>>();

        let preprocessed_traces = Traces { named_traces: preprocessed_traces };

        // Copy the real traces to the target backend.
        let real_traces = chips_and_traces
            .into_iter()
            .map(|(chip, trace)| {
                let trace = self.trace_allocator.copy_into(trace).unwrap();
                let mle = Arc::new(trace);
                (
                    chip.name().to_string(),
                    PaddedMle::padded_with_zeros(mle, max_log_row_count as u32),
                )
            })
            .collect::<Vec<_>>();

        let mut traces = padded_traces;

        for (name, trace) in real_traces {
            traces.insert(name, trace);
        }

        let traces = Traces { named_traces: traces };

        let main_trace_data = MainTraceData { traces, public_values, shard_chips, permit };

        TraceData { preprocessed_traces, main_trace_data }
    }
}
