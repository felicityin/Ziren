//! # Async Pipeline
//!
//! A flexible and efficient asynchronous task execution pipeline for Rust.
//!
//! This module provides a framework for building composable asynchronous pipelines
//! that can process tasks through multiple stages with worker pools and capacity management.
//!
//! ## Features
//!
//! - **Worker Pools**: Manage pools of workers that can execute tasks concurrently
//! - **Capacity Control**: Use semaphore-based permits to limit concurrent task execution
//! - **Pipeline Composition**: Chain multiple pipelines together to create complex workflows
//! - **Task Weighting**: Support for weighted tasks that consume multiple permits
//! - **Error Handling**: Comprehensive error types for different failure scenarios
//!
//! ## Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use tokio::sync::Semaphore;
//!
//! // Create workers and engine
//! let workers = vec![MyWorker::new(); 4];
//! let permits = Arc::new(Semaphore::new(10));
//! let engine = AsyncEngine::new(workers, permits);
//!
//! // Submit a task
//! let handle = engine.submit(my_task).await?;
//! let result = handle.await?;
//! ```

use core::marker::PhantomData;
use std::{
    fmt,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use tracing::Instrument;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use thiserror::Error;

use crate::queue::{self, AcquireWorkerError, WorkerQueue};

/// A trait for task inputs that can be processed by the pipeline.
///
/// Tasks must have a static lifetime and be thread-safe. This trait is automatically implemented
/// for all static lifetime types that are `Send` and `Sync`.
pub trait TaskInput: 'static + Send + Sync {}

impl<T: 'static + Send + Sync> TaskInput for T {}

/// Error returned when a task submission fails.
///
/// This error indicates that the engine has been closed and is no longer accepting new tasks.
#[derive(Error, Debug)]
#[error("Engine closed")]
pub struct SubmitError;

/// Error returned when a non-blocking task submission fails.
///
/// This error can occur for two reasons:
/// - The engine has been closed
/// - No capacity is currently available (all permits are in use)
#[derive(Error)]
#[error("failed to submit task")]
pub enum TrySubmitError<T> {
    /// The engine has been closed and is no longer accepting tasks
    #[error("engine closed")]
    Closed,
    /// No capacity is currently available for new tasks
    #[error("no capacity available")]
    NoCapacity(T),
}

impl<T> fmt::Debug for TrySubmitError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TrySubmitError<{}>", std::any::type_name::<T>())
    }
}

/// Error returned when a task submission fails.
///
/// This error indicates that the engine has been closed and is no longer accepting new tasks.
#[derive(Error, Debug)]
pub enum RunError {
    #[error("failed to submit task")]
    SubmitError(#[from] SubmitError),
    #[error("task execution failed")]
    TaskFailed(#[from] TaskJoinError),
}

/// Error that can occur when waiting for a task to complete.
///
/// This error type represents various failure modes that can occur
/// during task execution or when acquiring workers from the pool.
#[derive(Error, Debug)]
pub enum TaskJoinError {
    /// The task failed during execution (e.g., panicked)
    #[error("execution error")]
    ExecutionError(#[from] tokio::task::JoinError),
    /// Failed to acquire a worker from the pool
    #[error("failed to acquire worker")]
    PopWorker(#[from] AcquireWorkerError),
}

/// A handle to a running task that can be awaited for its result.
///
/// This handle is returned when a task is submitted to the pipeline
/// and can be used to:
/// - Wait for the task to complete and retrieve its result
/// - Abort the task if it's no longer needed
///
/// # Example
///
/// ```ignore
/// let handle = engine.submit(my_task).await?;
///
/// // Option 1: Wait for completion
/// match handle.await {
///     Ok(result) => println!("Task completed: {:?}", result),
///     Err(e) => eprintln!("Task failed: {}", e),
/// }
///
/// // Option 2: Abort the task
/// handle.abort();
/// ```
pub struct TaskHandle<T> {
    inner: tokio::task::JoinHandle<Result<T, TaskJoinError>>,
}

impl<T> TaskHandle<T> {
    /// Aborts the task associated with this handle.
    ///
    /// This will cause the task to stop executing as soon as possible.
    /// Any work already completed by the task will be lost.
    pub fn abort(&self) {
        self.inner.abort();
    }
}

impl<T> Drop for TaskHandle<T> {
    fn drop(&mut self) {
        self.abort();
    }
}

impl<T> Future for TaskHandle<T> {
    type Output = Result<T, TaskJoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let pin = Pin::new(&mut self.inner);
        pin.poll(cx).map(|res| res.map_err(TaskJoinError::from)).map(|res| match res {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(error)) => Err(error),
            Err(error) => Err(error),
        })
    }
}

pub type SubmitHandle<P> = PipelineHandle<<P as Pipeline>::Resource, <P as Pipeline>::Output>;

pub struct PipelineHandle<R, O> {
    handle: TaskHandle<(R, O)>,
}

impl<R, O> PipelineHandle<R, O> {
    pub fn new(handle: TaskHandle<(R, O)>) -> Self {
        Self { handle }
    }

    pub fn abort(&self) {
        self.handle.abort();
    }

    fn into_inner(self) -> TaskHandle<(R, O)> {
        self.handle
    }
}

impl<R, O> Future for PipelineHandle<R, O> {
    type Output = Result<O, TaskJoinError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let pin = Pin::new(&mut self.handle);
        pin.poll(cx).map(|res| res.map(|(_, output)| output))
    }
}

/// A trait representing an asynchronous processing pipeline.
///
/// Pipelines accept input tasks and produce output results asynchronously.
/// They can be composed together to create complex processing workflows.
///
/// # Type Parameters
///
/// - `Input`: The type of input tasks the pipeline accepts
/// - `Output`: The type of results the pipeline produces
///
/// # Required Methods
///
/// - `submit`: Asynchronously submit a task, waiting if necessary for capacity
/// - `try_submit`: Try to submit a task without waiting
pub trait Pipeline: 'static + Send + Sync {
    /// The input type that this pipeline accepts
    type Input: 'static + Send + Sync;
    /// The output type that this pipeline produces
    type Output: 'static + Send + Sync;
    /// The resource type that this pipeline uses
    type Resource: 'static + Send + Sync;

    /// Submit a task to the pipeline, waiting if necessary for capacity.
    ///
    /// This method will wait until there is capacity available in the pipeline before submitting
    /// the task.
    fn submit(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<SubmitHandle<Self>, SubmitError>> + Send;

    /// Try to submit a task without waiting.
    ///
    /// This method returns immediately with an error if there is no capacity
    /// available in the pipeline.
    fn try_submit(
        &self,
        input: Self::Input,
    ) -> Result<SubmitHandle<Self>, TrySubmitError<Self::Input>>;

    /// Run the pipeline on an input task and wait for the output.
    ///     
    /// This method will submit the task to the pipeline and wait for the output.
    fn run(
        &self,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output, RunError>> + Send {
        async move {
            let handle = self.submit(input).await?;
            let output = handle.await.map_err(RunError::from)?;
            Ok(output)
        }
    }

    fn blocking_submit(&self, input: Self::Input) -> Result<SubmitHandle<Self>, SubmitError> {
        let mut last_input = input;
        loop {
            match self.try_submit(last_input) {
                Ok(handle) => {
                    return Ok(handle);
                }
                Err(TrySubmitError::NoCapacity(input)) => {
                    last_input = input;
                    std::hint::spin_loop();
                }
                Err(TrySubmitError::Closed) => {
                    return Err(SubmitError);
                }
            }
        }
    }
}

/// A trait for workers that can process tasks asynchronously.
///
/// Workers are the units of execution in the pipeline. They receive
/// input tasks and produce output results asynchronously.
///
/// # Example
///
/// ```ignore
/// #[derive(Debug, Clone)]
/// struct MyWorker {
///     config: WorkerConfig,
/// }
///
/// impl AsyncWorker<MyTask, MyResult> for MyWorker {
///     async fn call(&self, input: MyTask) -> MyResult {
///         // Process the task...
///         MyResult::new()
///     }
/// }
/// ```
pub trait AsyncWorker<Input, Output>: 'static + Send + Sync {
    /// Process an input task and produce an output result.
    ///
    /// This method is called by the engine when a worker is assigned
    /// to process a task.
    fn call(&self, input: Input) -> impl Future<Output = Output> + Send;
}

/// An asynchronous execution engine that manages a pool of workers.
///
/// The `AsyncEngine` orchestrates task execution using:
/// - A pool of workers that process tasks
/// - A semaphore-based permit system for capacity control
/// - Task weighting support for resource management
///
/// # Type Parameters
///
/// - `Input`: The task input type (must implement `TaskInput`)
/// - `Output`: The result type produced by workers
/// - `Worker`: The worker type that processes tasks
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use tokio::sync::Semaphore;
///
/// // Create a pool of 4 workers with capacity for 10 concurrent tasks
/// let workers = vec![MyWorker::new(); 4];
/// let permits = Arc::new(Semaphore::new(10));
/// let engine = AsyncEngine::new(workers, permits);
///
/// // Submit tasks to the engine
/// let handle = engine.submit(my_task).await?;
/// let result = handle.await?;
/// ```
#[derive(Debug, Clone)]
pub struct AsyncEngine<Input, Output, Worker> {
    task_permits: Arc<Semaphore>,
    workers: Arc<WorkerQueue<Worker>>,
    _marker: PhantomData<(Input, Output)>,
}

impl<Input, Output, Worker> AsyncEngine<Input, Output, Worker>
where
    Input: TaskInput,
    Worker: AsyncWorker<Input, Output>,
    Output: 'static + Send + Sync,
{
    /// Creates a new `AsyncEngine` with the specified workers and permit semaphore.
    ///
    /// # Arguments
    ///
    /// - `workers`: A vector of workers that will process tasks
    /// - `input_buffer_size`: The size of the input buffer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let workers = vec![MyWorker::new(); 4];
    /// let engine = AsyncEngine::new(workers, 10);
    /// ```
    pub fn new(workers: Vec<Worker>, input_buffer_size: usize) -> Self {
        Self {
            workers: Arc::new(WorkerQueue::new(workers)),
            task_permits: Arc::new(Semaphore::new(input_buffer_size)),
            _marker: PhantomData,
        }
    }

    /// Create a new `AsyncEngine` with a single permit per worker.
    ///
    /// # Arguments
    ///
    /// - `workers`: A vector of workers that will process tasks
    ///
    /// # Example
    ///
    /// ```ignore
    /// let workers = vec![MyWorker::new(); 4];
    /// let engine = AsyncEngine::single_permit_per_worker(workers);
    /// ```
    pub fn single_permit_per_worker(workers: Vec<Worker>) -> Self {
        let num_workers = workers.len();
        Self::new(workers, num_workers)
    }

    fn spawn(
        &self,
        input: Input,
        permit: OwnedSemaphorePermit,
    ) -> TaskHandle<(queue::Worker<Worker>, Output)> {
        let workers = self.workers.clone();
        let handle = tokio::spawn(
            async move {
                let permit = permit;
                let worker = workers
                    .pop()
                    .instrument(tracing::debug_span!("waiting for a worker"))
                    .await
                    .map_err(TaskJoinError::from)?;
                // Drop the permit to release the input queue task slot.
                drop(permit);
                // Process the task.
                let output = worker.call(input).await;
                // Return the worker and output.
                Ok((worker, output))
            }
            .in_current_span(),
        );
        TaskHandle { inner: handle }
    }
}

/// Implementation of `Pipeline` for `AsyncEngine`.
///
/// This allows the async engine to be used as a pipeline component,
/// enabling it to be composed with other pipelines.
impl<Input, Output, Worker> Pipeline for AsyncEngine<Input, Output, Worker>
where
    Input: TaskInput,
    Worker: AsyncWorker<Input, Output>,
    Output: 'static + Send + Sync,
{
    type Input = Input;
    type Output = Output;
    type Resource = queue::Worker<Worker>;

    async fn submit(&self, input: Self::Input) -> Result<SubmitHandle<Self>, SubmitError> {
        let permit = self
            .task_permits
            .clone()
            .acquire_owned()
            .instrument(tracing::debug_span!("waiting to enter input queue"))
            .await
            .map_err(|_| SubmitError)?;
        Ok(PipelineHandle::new(self.spawn(input, permit)))
    }

    fn try_submit(
        &self,
        input: Self::Input,
    ) -> Result<SubmitHandle<Self>, TrySubmitError<Self::Input>> {
        let permit_result = self.task_permits.clone().try_acquire_owned();
        match permit_result {
            Ok(permit) => Ok(PipelineHandle::new(self.spawn(input, permit))),
            Err(TryAcquireError::NoPermits) => Err(TrySubmitError::NoCapacity(input)),
            Err(TryAcquireError::Closed) => Err(TrySubmitError::Closed),
        }
    }
}

/// A trait for workers that process tasks synchronously.
///
/// This trait is similar to `AsyncWorker` but for synchronous blocking tasks. It can be useful
/// when wrapping blocking operations or integrating with non-async code.
///
/// # Example
///
/// ```ignore
/// struct BlockingWorker;
///
/// impl BlockingWorker<ComputeTask, ComputeResult> for BlockingWorker {
///     fn call(&self, input: ComputeTask) -> ComputeResult {
///         // Perform a potentially blocking calculation
///         ComputeResult::wait_for_result(input)
///     }
/// }
/// ```
pub trait BlockingWorker<Input, Output>: 'static + Send + Sync {
    /// Process an input task synchronously and produce an output result.
    fn call(&self, input: Input) -> Output;
}

/// A trait for workers that process tasks synchronously.
///
/// This trait is similar to `AsyncWorker` but for synchronous cpu-intensive tasks. It can be useful
/// when wrapping blocking operations or integrating with non-async code.
///
/// # Example
///
/// ```ignore
/// struct CpuIntensiveWorker;
///
/// impl RayonWorker<ComputeTask, ComputeResult> for CpuIntensiveWorker {
///     fn call(&self, input: ComputeTask) -> ComputeResult {
///         // Perform CPU-intensive calculation
///         ComputeResult::calculate(input)
///     }
/// }
/// ```
pub trait RayonWorker<Input, Output>: 'static + Send + Sync {
    /// Process an input task synchronously and produce an output result.
    fn call(&self, input: Input) -> Output;
}

/// A blocking execution engine that manages a pool of workers for blocking tasks.
///
/// The `BlockingEngine` is similar to `AsyncEngine` but designed for synchronous, blocking tasks.
/// It executes blocking tasks on the tokio runtime to avoid blocking the async runtime.
///
/// # Type Parameters
///
/// - `Input`: The task input type (must implement `TaskInput`)
/// - `Output`: The result type produced by workers
/// - `Worker`: The worker type that processes tasks synchronously
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use tokio::sync::Semaphore;
///
/// // Create a pool of workers for CPU-intensive tasks
/// let workers = vec![ComputeWorker::new(); 4];
/// let permits = Arc::new(Semaphore::new(10));
/// let engine = BlockingEngine::new(workers, permits);
///
/// // Submit CPU-intensive tasks
/// let handle = engine.submit(compute_task).await?;
/// let result = handle.await?;
/// ```
#[derive(Debug, Clone)]
pub struct BlockingEngine<Input, Output, Worker> {
    task_permits: Arc<Semaphore>,
    workers: Arc<WorkerQueue<Worker>>,
    _marker: PhantomData<(Input, Output)>,
}

impl<Input, Output, Worker> BlockingEngine<Input, Output, Worker>
where
    Input: TaskInput,
    Worker: BlockingWorker<Input, Output>,
    Output: 'static + Send + Sync,
{
    /// Creates a new `BlockingEngine` with the specified workers and permit semaphore.
    ///
    /// # Arguments
    ///
    /// - `workers`: A vector of workers that will process tasks
    /// - `permits`: A semaphore controlling the maximum number of concurrent tasks
    ///
    /// # Example
    ///
    /// ```ignore
    /// let workers = vec![MyWorker::new(); 4];
    /// let engine = BlockingEngine::new(workers, 10);
    /// ```
    pub fn new(workers: Vec<Worker>, input_buffer_size: usize) -> Self {
        Self {
            workers: Arc::new(WorkerQueue::new(workers)),
            task_permits: Arc::new(Semaphore::new(input_buffer_size)),
            _marker: PhantomData,
        }
    }

    /// Create a new `BlockingEngine` with a single permit per worker.
    ///
    /// # Arguments
    ///
    /// - `workers`: A vector of workers that will process tasks
    ///
    /// # Example
    ///
    /// ```ignore
    /// let workers = vec![MyWorker::new(); 4];
    /// let engine = BlockingEngine::single_permit_per_worker(workers);
    /// ```
    pub fn single_permit_per_worker(workers: Vec<Worker>) -> Self {
        let num_workers = workers.len();
        Self::new(workers, num_workers)
    }

    fn spawn(
        &self,
        input: Input,
        permit: OwnedSemaphorePermit,
    ) -> TaskHandle<(queue::Worker<Worker>, Output)> {
        let workers = self.workers.clone();
        let handle = tokio::spawn(
            async move {
                let permit = permit;
                // Wait for a worker to become available.
                let worker = workers
                    .pop()
                    .instrument(tracing::debug_span!("waiting for a worker"))
                    .await
                    .map_err(TaskJoinError::from)?;
                // Drop the permit to release the input queue task slot.
                drop(permit);
                let span = tracing::Span::current();
                let (worker, output) = tokio::task::spawn_blocking(move || {
                    let _guard = span.enter();
                    let output = worker.call(input);
                    (worker, output)
                })
                .await
                .unwrap();
                Ok((worker, output))
            }
            .in_current_span(),
        );
        TaskHandle { inner: handle }
    }

    pub fn blocking_submit(
        &self,
        input: Input,
    ) -> Result<TaskHandle<(queue::Worker<Worker>, Output)>, SubmitError> {
        let permit = loop {
            match self.task_permits.clone().try_acquire_owned() {
                Ok(permit) => break permit,
                Err(TryAcquireError::NoPermits) => {
                    std::hint::spin_loop();
                }
                Err(TryAcquireError::Closed) => {
                    return Err(SubmitError);
                }
            }
        };
        Ok(self.spawn(input, permit))
    }
}

/// Implementation of `Pipeline` for `BlockingEngine`.
///
/// This allows the blocking engine to be used as a pipeline component,
/// enabling it to be composed with other pipelines.
impl<Input, Output, Worker> Pipeline for BlockingEngine<Input, Output, Worker>
where
    Input: TaskInput,
    Worker: BlockingWorker<Input, Output>,
    Output: 'static + Send + Sync,
{
    type Input = Input;
    type Output = Output;
    type Resource = queue::Worker<Worker>;

    async fn submit(&self, input: Self::Input) -> Result<SubmitHandle<Self>, SubmitError> {
        let permit = self
            .task_permits
            .clone()
            .acquire_owned()
            .instrument(tracing::debug_span!("waiting to enter input queue"))
            .await
            .map_err(|_| SubmitError)?;
        Ok(PipelineHandle::new(self.spawn(input, permit)))
    }

    fn try_submit(
        &self,
        input: Self::Input,
    ) -> Result<SubmitHandle<Self>, TrySubmitError<Self::Input>> {
        let permit_result = self.task_permits.clone().try_acquire_owned();
        match permit_result {
            Ok(permit) => Ok(PipelineHandle::new(self.spawn(input, permit))),
            Err(TryAcquireError::NoPermits) => Err(TrySubmitError::NoCapacity(input)),
            Err(TryAcquireError::Closed) => Err(TrySubmitError::Closed),
        }
    }
}

/// An execution engine that manages a pool of workers for CPU-intensive tasks using `rayon`.
///
/// The `RayonEngine` is similar to `BlockinEngine` but designed for synchronous, CPU-intensive
/// workloads. It executes blocking tasks on a Rayon thread pool to avoid blocking the async
/// runtime.
///
/// # Type Parameters
///
/// - `Input`: The task input type (must implement `TaskInput`)
/// - `Output`: The result type produced by workers
/// - `Worker`: The worker type that processes tasks synchronously
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use tokio::sync::Semaphore;
///
/// // Create a pool of workers for CPU-intensive tasks
/// let workers = vec![ComputeWorker::new(); 4];
/// let permits = Arc::new(Semaphore::new(10));
/// let engine = BlockingEngine::new(workers, permits);
///
/// // Submit CPU-intensive tasks
/// let handle = engine.submit(compute_task).await?;
/// let result = handle.await?;
/// ```
#[derive(Debug, Clone)]
pub struct RayonEngine<Input, Output, Worker> {
    task_permits: Arc<Semaphore>,
    workers: Arc<WorkerQueue<Worker>>,
    _marker: PhantomData<(Input, Output)>,
}

impl<Input, Output, Worker> RayonEngine<Input, Output, Worker>
where
    Input: TaskInput,
    Worker: RayonWorker<Input, Output>,
    Output: 'static + Send + Sync,
{
    /// Creates a new `RayonEngine` with the specified workers and permit semaphore.
    ///
    /// # Arguments
    ///
    /// - `workers`: A vector of workers that will process tasks
    /// - `permits`: A semaphore controlling the maximum number of concurrent tasks
    ///
    /// # Example
    ///
    /// ```ignore
    /// let workers = vec![MyWorker::new(); 4];
    /// let permits = Arc::new(Semaphore::new(10));
    /// let engine = RayonEngine::new(workers, permits);
    /// ```
    pub fn new(workers: Vec<Worker>, permits: Arc<Semaphore>) -> Self {
        Self {
            workers: Arc::new(WorkerQueue::new(workers)),
            task_permits: permits,
            _marker: PhantomData,
        }
    }

    /// Create a new `RayonEngine` with a single permit per worker.
    ///
    /// # Arguments
    ///
    /// - `workers`: A vector of workers that will process tasks
    ///
    /// # Example
    ///
    /// ```ignore
    /// let workers = vec![MyWorker::new(); 4];
    /// let engine = RayonEngine::single_permit_per_worker(workers);
    /// ```
    pub fn single_permit_per_worker(workers: Vec<Worker>) -> Self {
        let num_workers = workers.len();
        Self::new(workers, Arc::new(Semaphore::new(num_workers)))
    }

    fn spawn(
        &self,
        input: Input,
        permit: OwnedSemaphorePermit,
    ) -> TaskHandle<(queue::Worker<Worker>, Output)> {
        let workers = self.workers.clone();
        let handle = tokio::spawn(
            async move {
                let permit = permit;
                // Wait for a worker to become available.
                let worker = workers
                    .pop()
                    .instrument(tracing::debug_span!("waiting for a worker"))
                    .await
                    .map_err(TaskJoinError::from)?;
                // Drop the permit to release the input queue task slot.
                drop(permit);
                // Spawn the blocking task on the rayon thread pool
                let ret = crate::rayon::spawn(move || {
                    let output = worker.call(input);
                    (worker, output)
                })
                .await
                .unwrap();
                Ok(ret)
            }
            .in_current_span(),
        );
        TaskHandle { inner: handle }
    }
}

/// Implementation of `Pipeline` for `BlockingEngine`.
///
/// This allows the blocking engine to be used as a pipeline component,
/// enabling it to be composed with other pipelines.
impl<Input, Output, Worker> Pipeline for RayonEngine<Input, Output, Worker>
where
    Input: TaskInput,
    Worker: RayonWorker<Input, Output>,
    Output: 'static + Send + Sync,
{
    type Input = Input;
    type Output = Output;
    type Resource = queue::Worker<Worker>;
    async fn submit(&self, input: Self::Input) -> Result<SubmitHandle<Self>, SubmitError> {
        let permit = self
            .task_permits
            .clone()
            .acquire_owned()
            .instrument(tracing::debug_span!("waiting to enter input queue"))
            .await
            .map_err(|_| SubmitError)?;
        Ok(PipelineHandle::new(self.spawn(input, permit)))
    }

    fn try_submit(
        &self,
        input: Self::Input,
    ) -> Result<SubmitHandle<Self>, TrySubmitError<Self::Input>> {
        let permit_result = self.task_permits.clone().try_acquire_owned();
        match permit_result {
            Ok(permit) => Ok(PipelineHandle::new(self.spawn(input, permit))),
            Err(TryAcquireError::NoPermits) => Err(TrySubmitError::NoCapacity(input)),
            Err(TryAcquireError::Closed) => Err(TrySubmitError::Closed),
        }
    }
}

/// A composite pipeline that chains two pipelines together.
///
/// `Chain` allows you to create complex processing workflows by connecting the output of one
/// pipeline to the input of another. The output type of the first pipeline must be convertible
/// to the input type of the second pipeline.
///
/// # Type Parameters
///
/// - `First`: The first pipeline in the chain
/// - `Second`: The second pipeline in the chain
///
/// # Example
///
/// ```ignore
/// // Create two pipelines
/// let preprocessing = PreprocessingPipeline::new();
/// let processing = ProcessingPipeline::new();
///
/// // Chain them together
/// let chain = Chain::new(preprocessing, processing);
///
/// // Submit tasks to the chained pipeline
/// let result = chain.submit(raw_data).await?;
/// ```
#[derive(Clone, Debug, Copy)]
pub struct Chain<First, Second> {
    first: First,
    second: Second,
}

impl<First, Second> Chain<First, Second>
where
    First: Pipeline + Clone,
    Second: Pipeline + Clone,
    First::Output: Into<Second::Input>,
{
    /// Creates a new chain from two pipelines.
    ///
    /// # Arguments
    ///
    /// - `first`: The first pipeline that will process the input
    /// - `second`: The second pipeline that will process the output from the first
    ///
    /// # Example
    ///
    /// ```ignore
    /// let chain = Chain::new(first_pipeline, second_pipeline);
    /// ```
    pub fn new(first: First, second: Second) -> Self {
        Self { first, second }
    }

    /// Get a reference to the first pipeline in the chain.
    ///
    /// This is useful for being able to submit tasks to the first pipeline directly, without having
    /// to go through the second pipeline if there is no need to.
    pub fn first(&self) -> &First {
        &self.first
    }

    /// Get a reference to the second pipeline in the chain.
    ///
    /// This is useful for being able to submit tasks to the second pipeline directly, without
    /// having to go through the first pipeline if there is no need to.
    pub fn second(&self) -> &Second {
        &self.second
    }

    fn spawn(
        &self,
        first_handle: TaskHandle<(First::Resource, First::Output)>,
    ) -> TaskHandle<(Second::Resource, Second::Output)> {
        let second = self.second.clone();
        let handle = tokio::spawn(
            async move {
                let first_handle = first_handle;
                let (first_resource, first_output) = first_handle.await?;
                let second_input: Second::Input = first_output.into();
                // Submit the second task to the second pipeline.
                let second_handle =
                    second.submit(second_input).await.expect("failed to submit second task");
                // Once the task is in the second pipeline, we can release the first resource.
                drop(first_resource);
                // Wait for the second task to complete with it's resource.
                let second_handle = second_handle.into_inner();
                second_handle.await
            }
            .in_current_span(),
        );
        TaskHandle { inner: handle }
    }
}

/// Implementation of `Pipeline` for `Chain<First, Second>`.
///
/// This implementation allows chains to be used as pipelines themselves,
/// enabling further composition and nesting of processing workflows.
impl<First, Second> Pipeline for Chain<First, Second>
where
    First: Pipeline + Clone,
    Second: Pipeline + Clone,
    First::Output: Into<Second::Input>,
{
    type Input = First::Input;
    type Output = Second::Output;
    type Resource = Second::Resource;
    /// Submit a task to the chained pipeline.
    ///
    /// The task will be processed by the first pipeline, and its output
    /// will automatically be fed as input to the second pipeline.
    ///
    /// # Arguments
    ///
    /// - `input`: The initial input for the first pipeline
    ///
    /// # Returns
    ///
    /// A handle to the final result from the second pipeline
    async fn submit(&self, input: Self::Input) -> Result<SubmitHandle<Self>, SubmitError> {
        let first_handle = self.first.submit(input).await?;
        Ok(PipelineHandle::new(self.spawn(first_handle.into_inner())))
    }

    /// Try to submit a task to the chained pipeline without blocking.
    ///
    /// # Arguments
    ///
    /// - `input`: The initial input for the first pipeline
    ///
    /// # Returns
    ///
    /// - `Ok(TaskHandle)` if the task was successfully submitted to the first pipeline
    /// - `Err(TrySubmitError)` if submission failed
    fn try_submit(
        &self,
        input: Self::Input,
    ) -> Result<SubmitHandle<Self>, TrySubmitError<Self::Input>> {
        let first_handle = self.first.try_submit(input)?;
        Ok(PipelineHandle::new(self.spawn(first_handle.into_inner())))
    }
}

/// Implementation of `Pipeline` for `Arc<P>` where `P` implements `Pipeline`.
///
/// This allows pipelines to be shared across multiple threads efficiently.
/// The `Arc` wrapper enables cheap cloning and thread-safe sharing of the
/// underlying pipeline.
///
/// # Example
///
/// ```ignore
/// let pipeline = MyPipeline::new();
/// let shared_pipeline = Arc::new(pipeline);
///
/// // Can now clone and share across threads
/// let pipeline_clone = shared_pipeline.clone();
/// tokio::spawn(async move {
///     let result = pipeline_clone.submit(task).await?;
/// });
/// ```
impl<P: Pipeline> Pipeline for Arc<P> {
    type Input = P::Input;
    type Output = P::Output;
    type Resource = P::Resource;

    #[inline]
    async fn submit(&self, input: Self::Input) -> Result<SubmitHandle<Self>, SubmitError> {
        self.as_ref().submit(input).await
    }

    #[inline]
    fn try_submit(
        &self,
        input: Self::Input,
    ) -> Result<SubmitHandle<Self>, TrySubmitError<Self::Input>> {
        self.as_ref().try_submit(input)
    }
}

#[derive(Debug, Clone)]
pub struct PipelineBuilder<P = ()> {
    pipeline: P,
}

impl PipelineBuilder {
    pub fn new<P: Pipeline>(pipeline: P) -> PipelineBuilder<P> {
        PipelineBuilder { pipeline }
    }
}

impl<P: Pipeline> PipelineBuilder<P> {
    /// Build the pipeline.
    ///
    /// # Returns
    ///
    /// The built pipeline
    pub fn build(self) -> P {
        self.pipeline
    }

    /// Chain the pipeline with another pipeline.
    ///
    /// # Arguments
    ///
    /// - `pipeline`: The pipeline to chain with
    ///
    /// # Returns
    ///
    /// A new pipeline builder with the chained pipeline
    pub fn through<Q>(self, pipeline: Q) -> PipelineBuilder<Chain<P, Q>>
    where
        P: Clone,
        Q: Pipeline + Clone,
        P::Output: Into<Q::Input>,
    {
        PipelineBuilder { pipeline: Chain::new(self.pipeline, pipeline) }
    }
}

#[cfg(test)]
mod tests {
    use futures::{prelude::*, stream::FuturesOrdered};
    use rand::Rng;
    use std::time::Duration;
    use tokio::task::JoinSet;

    use super::*;

    #[derive(Debug, Clone)]
    struct TestWorker;

    #[derive(Debug, Clone)]
    struct TestTask {
        time: Duration,
        hanging_probability: f64,
    }

    impl AsyncWorker<TestTask, ()> for TestWorker {
        async fn call(&self, input: TestTask) {
            tokio::time::sleep(input.time).await;

            let should_hang = rand::thread_rng().gen_bool(input.hanging_probability);
            if should_hang {
                loop {
                    tokio::task::yield_now().await;
                }
            }
        }
    }

    #[tokio::test]
    #[allow(clippy::print_stdout)]
    async fn test_async_engine() {
        let num_workers = 5;
        let task_queue_length = 5;
        let num_tasks_spawned = 10;
        let wait_duration = Duration::from_millis(10);

        let workers = (0..num_workers).map(|_| TestWorker).collect();
        let engine = Arc::new(AsyncEngine::new(workers, task_queue_length));

        let tasks = (0..num_tasks_spawned)
            .map(|_| TestTask { time: wait_duration, hanging_probability: 0.0 })
            .collect::<Vec<_>>();

        // Submit all tasks concurrently and wait for them to complete
        let mut join_set = JoinSet::new();
        let time = tokio::time::Instant::now();
        for task in tasks {
            let e = engine.clone();
            join_set.spawn(async move { e.submit(task).await.unwrap().await.unwrap() });
        }
        join_set.join_all().await;
        let duration = time.elapsed();
        println!("Time taken for async engine: {:?}", duration);

        // Compare this with the case of complete parallelism
        let mut join_set = JoinSet::new();
        let tasks_per_worker = num_tasks_spawned / num_workers;
        let time = tokio::time::Instant::now();
        for _ in 0..num_workers {
            join_set.spawn(async move {
                for _ in 0..tasks_per_worker {
                    tokio::time::sleep(wait_duration).await;
                }
            });
        }
        join_set.join_all().await;
        let duration = time.elapsed();
        println!("Time taken for complete parallelism: {:?}", duration);
    }

    #[tokio::test]
    #[allow(clippy::print_stdout)]
    async fn test_hanging_task_async_engine() {
        let num_workers = 1;
        let task_queue_length = 2;
        let num_tasks_spawned = 100;
        let wait_duration = Duration::from_millis(1);
        let hanging_probability = 0.5;
        let timeout = Duration::from_millis(100);

        let workers = (0..num_workers).map(|_| TestWorker).collect();
        let engine = Arc::new(AsyncEngine::new(workers, task_queue_length));

        let tasks = (0..num_tasks_spawned)
            .map(|_| TestTask { time: wait_duration, hanging_probability })
            .collect::<Vec<_>>();

        // Submit all tasks concurrently and wait for them to complete
        let mut join_set = JoinSet::new();
        let time = tokio::time::Instant::now();
        for task in tasks {
            let handle = engine.submit(task).await.unwrap();
            let future = async move { handle.await.unwrap() };
            join_set.spawn(async move { tokio::time::timeout(timeout, future).await });
        }

        let mut success_count = 0;
        while let Some(result) = join_set.join_next().await {
            let result = result.unwrap();
            if result.is_ok() {
                success_count += 1;
            }
        }
        let duration = time.elapsed();
        println!("Time taken for async engine: {:?}, success count: {success_count}", duration);
    }

    #[tokio::test]
    #[allow(clippy::print_stdout)]
    async fn test_blocking_engine() {
        #[derive(Debug, Clone)]
        struct SummingWorker;

        #[derive(Debug, Clone)]
        struct SummingTask {
            summands: Vec<u32>,
        }

        impl BlockingWorker<SummingTask, u32> for SummingWorker {
            fn call(&self, input: SummingTask) -> u32 {
                input.summands.iter().sum()
            }
        }

        let num_workers = 10;
        let task_queue_length = 20;
        let num_tasks_spawned = 10;
        let max_summands = 20;

        let workers = (0..num_workers).map(|_| SummingWorker).collect();
        let engine = Arc::new(BlockingEngine::new(workers, task_queue_length));

        let mut rng = rand::thread_rng();
        let tasks = (0..num_tasks_spawned)
            .map(|_| SummingTask { summands: vec![1; rng.gen_range(1..=max_summands)] })
            .collect::<Vec<_>>();

        // Submit all tasks concurrently and wait for them to complete
        let mut results = FuturesOrdered::new();
        for task in tasks.iter() {
            results.push_back(engine.submit(task.clone()).await.unwrap());
        }
        let results = results.collect::<Vec<_>>().await;
        for (task, result) in tasks.iter().zip(results) {
            let result = result.unwrap();
            let expected = task.summands.iter().sum();
            assert_eq!(result, expected);
        }
    }

    #[tokio::test]
    #[allow(clippy::print_stdout)]
    #[should_panic]
    async fn test_async_failing_engine() {
        #[derive(Debug, Clone)]
        struct FailingWorker;

        #[derive(Debug, Clone)]
        struct TestTask {
            time: Duration,
        }

        impl AsyncWorker<TestTask, ()> for FailingWorker {
            async fn call(&self, input: TestTask) {
                if input.time > Duration::from_millis(50) {
                    panic!("not interested to wait for this long");
                }
                tokio::time::sleep(input.time).await;
            }
        }
        let num_workers = 10;
        let task_queue_length = 20;
        let wait_duration = 100;

        let workers = (0..num_workers).map(|_| FailingWorker).collect();
        let engine = Arc::new(AsyncEngine::new(workers, task_queue_length));

        let tasks = (0..wait_duration)
            .map(|i| TestTask { time: Duration::from_millis(i) })
            .collect::<Vec<_>>();

        // Submit all tasks concurrently and wait for them to complete
        let mut join_set = JoinSet::new();
        let time = tokio::time::Instant::now();
        for task in tasks {
            let e = engine.clone();
            join_set.spawn(async move { e.submit(task).await.unwrap().await.unwrap() });
        }
        join_set.join_all().await;
        let duration = time.elapsed();
        println!("Time taken for async engine: {:?}", duration);
    }

    #[tokio::test]
    #[allow(clippy::print_stdout)]
    async fn test_chained_pipelines() {
        #[derive(Debug, Clone)]
        struct FirstTask;

        #[derive(Debug, Clone)]
        struct FirstWorker;

        impl BlockingWorker<FirstTask, SecondTask> for FirstWorker {
            fn call(&self, _input: FirstTask) -> SecondTask {
                let mut rng = rand::thread_rng();
                SecondTask { value: rng.gen_range(200..=1000) }
            }
        }

        #[derive(Debug, Clone)]
        struct SecondWorker;

        #[derive(Debug, Clone)]
        struct SecondTask {
            value: u64,
        }

        impl AsyncWorker<SecondTask, u64> for SecondWorker {
            async fn call(&self, input: SecondTask) -> u64 {
                tokio::time::sleep(Duration::from_millis(input.value)).await;
                input.value
            }
        }

        let first_workers = (0..10).map(|_| FirstWorker).collect();
        let first_pipeline = Arc::new(BlockingEngine::single_permit_per_worker(first_workers));
        let second_workers = (0..10).map(|_| SecondWorker).collect();
        let second_pipeline = Arc::new(AsyncEngine::single_permit_per_worker(second_workers));
        let chain = Chain::new(first_pipeline, second_pipeline);

        let handles = (0..10)
            .map(|_| chain.submit(FirstTask))
            .collect::<FuturesOrdered<_>>()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        for handle in handles {
            let _result = handle.await.unwrap();
        }
    }

    #[tokio::test]
    #[allow(clippy::print_stdout)]
    async fn test_timing_chained_pipelines() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct SleepTask {
            duration: Duration,
        }

        #[derive(Debug, Clone)]
        struct SleepWorker;

        impl AsyncWorker<SleepTask, SleepTask> for SleepWorker {
            async fn call(&self, input: SleepTask) -> SleepTask {
                let sleep_duration = input.duration;
                tokio::time::sleep(sleep_duration).await;
                input
            }
        }

        let num_workers = 10;

        let workers = (0..num_workers).map(|_| SleepWorker).collect::<Vec<_>>();
        let make_engine =
            |workers: Vec<SleepWorker>| Arc::new(AsyncEngine::single_permit_per_worker(workers));

        let pipeline = PipelineBuilder::new(make_engine(workers.clone()))
            .through(make_engine(workers.clone()))
            .through(make_engine(workers.clone()))
            .through(make_engine(workers.clone()))
            .through(make_engine(workers.clone()))
            .build();

        let chain_input_task = SleepTask { duration: Duration::from_millis(100) };
        let single_input_task = SleepTask { duration: Duration::from_millis(500) };

        let time = tokio::time::Instant::now();
        let chain_result = pipeline.submit(chain_input_task).await.unwrap().await.unwrap();
        let chain_duration = time.elapsed();
        println!("Chain duration: {:?}", chain_duration);
        assert_eq!(chain_result, chain_input_task);

        let single_engine = make_engine(workers.clone());
        let time = tokio::time::Instant::now();
        let single_result = single_engine.submit(single_input_task).await.unwrap().await.unwrap();
        let single_duration = time.elapsed();
        println!("Single duration: {:?}", single_duration);
        assert_eq!(single_result, single_input_task);
    }
}
