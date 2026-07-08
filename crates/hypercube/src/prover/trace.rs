use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use serde::{Deserialize, Serialize};
use slop_alloc::Backend;
use slop_multilinear::PaddedMle;
use slop_tensor::Tensor;

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
