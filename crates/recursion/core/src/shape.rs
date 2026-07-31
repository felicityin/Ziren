use std::marker::PhantomData;

use hashbrown::HashMap;

use itertools::Itertools;
use p3_field::{extension::BinomiallyExtendable, PrimeField32};
use serde::{Deserialize, Serialize};
use zkm_hypercube::{air::MachineAir, shape::OrderedShape};

use crate::{
    chips::{
        alu_base::BaseAluChip,
        alu_ext::ExtAluChip,
        mem::{MemoryConstChip, MemoryVarChip},
        poseidon2_wide::Poseidon2WideChip,
        prefix_sum_checks::PrefixSumChecksChip,
        public_values::{PublicValuesChip, PUB_VALUES_LOG_HEIGHT},
        select::SelectChip,
    },
    machine::RecursionAir,
    RecursionProgram, D,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecursionShape {
    pub(crate) inner: HashMap<String, usize>,
}

impl RecursionShape {
    pub fn clone_into_hash_map(&self) -> HashMap<String, usize> {
        self.inner.clone()
    }
}

impl From<HashMap<String, usize>> for RecursionShape {
    fn from(value: HashMap<String, usize>) -> Self {
        Self { inner: value }
    }
}

pub struct RecursionShapeConfig<F, A> {
    allowed_shapes: Vec<HashMap<String, usize>>,
    _marker: PhantomData<(F, A)>,
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize>
    RecursionShapeConfig<F, RecursionAir<F, DEGREE>>
{
    pub fn fix_shape(&self, program: &mut RecursionProgram<F>) {
        let heights = RecursionAir::<F, DEGREE>::heights(program);

        let mut closest_shape = None;

        for shape in self.allowed_shapes.iter() {
            // If any of the heights is greater than the shape, continue.
            let mut valid = true;
            for (name, height) in heights.iter() {
                if *height > (1 << shape.get(name).unwrap()) {
                    valid = false;
                }
            }

            if !valid {
                continue;
            }

            closest_shape = Some(shape.clone());
            break;
        }

        if let Some(shape) = closest_shape {
            let shape = RecursionShape { inner: shape };
            *program.shape_mut() = Some(shape);
        } else {
            panic!("no shape found for heights: {heights:?}");
        }
    }

    pub fn get_all_shape_combinations(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Vec<OrderedShape>> + '_ {
        (0..batch_size)
            .map(|_| {
                self.allowed_shapes
                    .iter()
                    .cloned()
                    .map(|map| map.into_iter().collect::<OrderedShape>())
            })
            .multi_cartesian_product()
    }

    pub fn union_config_with_extra_room(&self) -> Self {
        let mut map = HashMap::new();
        for shape in self.allowed_shapes.clone() {
            for key in shape.keys() {
                let current = map.get(key).unwrap_or(&0);
                map.insert(key.clone(), *current.max(shape.get(key).unwrap()));
            }
        }
        map.values_mut().for_each(|x| *x += 2);
        map.insert("PublicValues".to_string(), 4);
        Self { allowed_shapes: vec![map], _marker: PhantomData }
    }

    pub fn from_hash_map(hash_map: &HashMap<String, usize>) -> Self {
        Self { allowed_shapes: vec![hash_map.clone()], _marker: PhantomData }
    }

    pub fn first(&self) -> Option<&HashMap<String, usize>> {
        self.allowed_shapes.first()
    }
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> Default
    for RecursionShapeConfig<F, RecursionAir<F, DEGREE>>
{
    fn default() -> Self {
        // Get the names of all the recursion airs to make the shape specification more readable.
        let mem_const = RecursionAir::<F, DEGREE>::MemoryConst(MemoryConstChip::default()).name();
        let mem_var = RecursionAir::<F, DEGREE>::MemoryVar(MemoryVarChip::default()).name();
        let base_alu = RecursionAir::<F, DEGREE>::BaseAlu(BaseAluChip).name();
        let ext_alu = RecursionAir::<F, DEGREE>::ExtAlu(ExtAluChip).name();
        let poseidon2_wide =
            RecursionAir::<F, DEGREE>::Poseidon2Wide(Poseidon2WideChip::<DEGREE>).name();
        let select = RecursionAir::<F, DEGREE>::Select(SelectChip).name();
        let prefix_sum_checks =
            RecursionAir::<F, DEGREE>::PrefixSumChecks(PrefixSumChecksChip).name();
        let public_values = RecursionAir::<F, DEGREE>::PublicValues(PublicValuesChip).name();

        // Specify allowed shapes.
        //
        // Each chip's real per-program row count is measured by `heights()` above and compared
        // against these entries by `fix_shape`; a chip's actual padded row count at runtime is
        // driven by the *shape* entry selected here (`fixed_log2_rows`, `chips/alu_ext.rs` and
        // siblings), not by the real measured height directly -- so an oversized entry wastes
        // real rows, and every entry must stay `<= RECURSION_MAX_LOG_ROW_COUNT`
        // (`crates/stark/src/opts.rs`): the jagged PCS's `PaddedMle` storage is fixed to
        // `2^RECURSION_MAX_LOG_ROW_COUNT` regardless of what a shape entry claims, so an entry
        // exceeding it pads to more real rows than the PCS config can hold, panicking in
        // `PaddedMle::padded`/`padded_with_zeros`.
        //
        // "Fastest" covers small/simple shards (few active core chips, e.g. `fibonacci`), with
        // headroom over real measured heights. "Fallback" covers larger, more chip-diverse
        // shards (e.g. `tendermint`), every dimension capped at the hard `RECURSION_MAX_LOG_ROW_COUNT`
        // ceiling -- the most headroom obtainable under that limit. If a real shard's measured
        // height still exceeds even this tier, `fix_shape` panics with "no shape found". These
        // have not yet been validated against deferred proofs, which could plausibly need a
        // larger `RECURSION_MAX_LOG_ROW_COUNT` outright.
        let allowed_shapes = [
            // Fastest shape.
            [
                (mem_var.clone(), 20),
                (select.clone(), 20),
                (mem_const.clone(), 19),
                (base_alu.clone(), 18),
                (ext_alu.clone(), 20),
                (poseidon2_wide.clone(), 18),
                (prefix_sum_checks.clone(), 19),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Fallback shape, with more headroom on the dimensions with the most real
            // per-program variance (`BaseAlu`, `Select`, `Poseidon2WideDeg3`). Every dimension
            // capped at 21 (`RECURSION_MAX_LOG_ROW_COUNT`) -- the maximum this tier can offer.
            [
                (mem_var.clone(), 21),
                (select.clone(), 21),
                (mem_const.clone(), 21),
                (base_alu.clone(), 18),
                (ext_alu.clone(), 21),
                (poseidon2_wide.clone(), 20),
                (prefix_sum_checks.clone(), 19),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
        ]
        .map(HashMap::from)
        .to_vec();
        Self { allowed_shapes, _marker: PhantomData }
    }
}
