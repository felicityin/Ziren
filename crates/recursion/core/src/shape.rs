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
        // The recursion verifier circuit's size is dominated by protocol-fixed jagged/basefold
        // verification overhead tied to `core_max_log_row_count()` (currently 22 -- see
        // `crates/hypercube/src/config.rs`) plus the number of distinct chips active in the core
        // shard being verified, not by the guest program's cycle count directly. A trivial,
        // single-chip guest program and a large, many-chip guest program stress different chips
        // in this table: the former needs `ExtAlu` at its ceiling with little else, the latter
        // needs far more `BaseAlu`/`Select` than the former ever exercises. Every entry here is
        // calibrated to comfortably cover both extremes, with headroom for real per-shard
        // variance on top of the larger of the two.
        //
        // Every entry here must also stay `<= recursion_max_log_row_count()` (currently 22,
        // `crates/stark/src/opts.rs`'s `RECURSION_MAX_SHARD_SIZE`): a chip's actual row count at
        // runtime is driven by the *shape* assigned here (the recursion runtime pads each chip's
        // event count up to it), while the jagged PCS's `PaddedMle` storage is fixed to
        // `2^recursion_max_log_row_count()` regardless of what the shape claims -- a shape entry
        // exceeding that ceiling pads to more real rows than the PCS config could ever hold.
        // `ExtAlu`'s natural requirement already equals that ceiling, so it has zero slack in
        // every tier. These have not yet been validated against deferred proofs, which could
        // plausibly need a larger `recursion_max_log_row_count()` outright (see
        // [[stage4-1-verifier-migration]] for the broader pattern of still-provisional
        // recursion-layer constants in this migration).
        let allowed_shapes = [
            // Fastest shape.
            [
                (mem_var.clone(), 22),
                (select.clone(), 20),
                (mem_const.clone(), 21),
                (base_alu.clone(), 17),
                (ext_alu.clone(), 22),
                (poseidon2_wide.clone(), 19),
                (prefix_sum_checks.clone(), 18),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Fallback shape, with more headroom on the dimensions with the most real
            // per-program variance (`BaseAlu`, `Select`, `Poseidon2WideDeg3`).
            [
                (mem_var.clone(), 22),
                (select.clone(), 21),
                (mem_const.clone(), 22),
                (base_alu.clone(), 18),
                (ext_alu.clone(), 22),
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
