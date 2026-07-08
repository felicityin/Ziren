use serde::{Deserialize, Serialize};

use crate::lookup::LookupKind;

/// A Lookup is a cross-table lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirLookup<E> {
    pub values: Vec<E>,
    pub multiplicity: E,
    pub kind: LookupKind,
}

impl<E> AirLookup<E> {
    pub const fn new(values: Vec<E>, multiplicity: E, kind: LookupKind) -> Self {
        Self { values, multiplicity, kind }
    }
}
