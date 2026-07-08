use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use slop_alloc::{Backend, CpuBackend};
use slop_multilinear::{Mle, MleEval, Point};
use slop_sumcheck::PartialSumcheckProof;

/// The output of the log-up GKR circuit.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound(serialize = "Mle<EF, B>: Serialize", deserialize = "Mle<EF, B>: Deserialize<'de>"))]
pub struct LogUpGkrOutput<EF, B: Backend = CpuBackend> {
    pub numerator: Mle<EF, B>,
    pub denominator: Mle<EF, B>,
}

/// The proof for a single round of the log-up GKR circuit.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogupGkrRoundProof<EF> {
    pub numerator_0: EF,
    pub numerator_1: EF,
    pub denominator_0: EF,
    pub denominator_1: EF,
    pub sumcheck_proof: PartialSumcheckProof<EF>,
}

/// The proof for the log-up GKR circuit.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogupGkrProof<F, EF> {
    pub circuit_output: LogUpGkrOutput<EF>,
    pub round_proofs: Vec<LogupGkrRoundProof<EF>>,
    pub logup_evaluations: LogUpEvaluations<EF>,
    pub witness: F,
}

/// The evaluations for a chip.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChipEvaluation<EF> {
    pub main_trace_evaluations: MleEval<EF>,
    pub preprocessed_trace_evaluations: Option<MleEval<EF>>,
}

/// The data passed from the GKR prover to the zerocheck prover.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LogUpEvaluations<EF> {
    pub point: Point<EF>,
    pub chip_openings: BTreeMap<String, ChipEvaluation<EF>>,
}
