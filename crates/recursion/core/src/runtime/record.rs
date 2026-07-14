use std::{array, sync::Arc};

use hashbrown::HashMap;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_hypercube::{
    air::{MachineAir, ZKMAirBuilder},
    record::MachineRecord,
    PROOF_MAX_NUM_PVS,
};

use super::{
    BaseAluEvent, CommitPublicValuesEvent, ExtAluEvent, MemEvent, Poseidon2Event,
    Poseidon2LinearLayerEvent, Poseidon2SBoxEvent, PrefixSumChecksEvent, RecursionProgram,
    RecursionPublicValues, SelectEvent,
};

#[derive(Clone, Default, Debug)]
pub struct ExecutionRecord<F> {
    pub program: Arc<RecursionProgram<F>>,
    /// The index of the shard.
    pub index: u32,

    pub base_alu_events: Vec<BaseAluEvent<F>>,
    pub ext_alu_events: Vec<ExtAluEvent<F>>,
    pub mem_const_count: usize,
    pub mem_var_events: Vec<MemEvent<F>>,
    /// The public values.
    pub public_values: RecursionPublicValues<F>,

    pub poseidon2_events: Vec<Poseidon2Event<F>>,
    pub poseidon2_linear_layer_events: Vec<Poseidon2LinearLayerEvent<F>>,
    pub poseidon2_sbox_events: Vec<Poseidon2SBoxEvent<F>>,
    pub select_events: Vec<SelectEvent<F>>,
    pub prefix_sum_checks_events: Vec<PrefixSumChecksEvent<F>>,
    pub commit_pv_hash_events: Vec<CommitPublicValuesEvent<F>>,
}

impl<F: PrimeField32> MachineRecord for ExecutionRecord<F> {
    fn stats(&self) -> hashbrown::HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("base_alu_events".to_string(), self.base_alu_events.len());
        stats.insert("ext_alu_events".to_string(), self.ext_alu_events.len());
        stats.insert("mem_var_events".to_string(), self.mem_var_events.len());

        stats.insert("poseidon2_events".to_string(), self.poseidon2_events.len());
        stats.insert(
            "poseidon2_linear_layer_events".to_string(),
            self.poseidon2_linear_layer_events.len(),
        );
        stats.insert("poseidon2_sbox_events".to_string(), self.poseidon2_sbox_events.len());

        stats
    }

    fn append(&mut self, other: &mut Self) {
        // Exhaustive destructuring for refactoring purposes.
        let Self {
            program: _,
            index: _,
            base_alu_events,
            ext_alu_events,
            mem_const_count,
            mem_var_events,
            public_values: _,
            poseidon2_events,
            poseidon2_linear_layer_events,
            poseidon2_sbox_events,
            select_events,
            prefix_sum_checks_events,
            commit_pv_hash_events,
        } = self;
        base_alu_events.append(&mut other.base_alu_events);
        ext_alu_events.append(&mut other.ext_alu_events);
        *mem_const_count += other.mem_const_count;
        mem_var_events.append(&mut other.mem_var_events);
        poseidon2_events.append(&mut other.poseidon2_events);
        poseidon2_linear_layer_events.append(&mut other.poseidon2_linear_layer_events);
        poseidon2_sbox_events.append(&mut other.poseidon2_sbox_events);
        select_events.append(&mut other.select_events);
        prefix_sum_checks_events.append(&mut other.prefix_sum_checks_events);
        commit_pv_hash_events.append(&mut other.commit_pv_hash_events);
    }

    fn public_values<T: FieldAlgebra>(&self) -> Vec<T> {
        let pv_elms = self.public_values.as_array();

        let ret: [T; PROOF_MAX_NUM_PVS] = array::from_fn(|i| {
            if i < pv_elms.len() {
                T::from_canonical_u32(pv_elms[i].as_canonical_u32())
            } else {
                T::ZERO
            }
        });

        ret.to_vec()
    }

    // Recursion programs execute as a single unsharded record, so there is no cross-shard
    // boundary state to chain here (unlike `zkm_core_executor::ExecutionRecord`, which anchors
    // memory/global accumulation lookups across shards).
    fn eval_public_values<AB: ZKMAirBuilder>(_builder: &mut AB) {}

    fn max_public_values_interaction_arity() -> usize {
        0
    }
}

impl<F: Field> ExecutionRecord<F> {
    #[inline]
    pub fn fixed_log2_rows<A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        self.program.fixed_log2_rows(air)
    }
}
