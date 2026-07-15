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
        // These bounds were re-derived 2026-07-15 (task #17, 阶段5.3) against the first real
        // compress-stage recursion program this migration has actually compiled and measured
        // (verifying one core shard for `test_artifacts::HELLO_WORLD_ELF`, the smallest possible
        // guest program). The recursion verifier circuit's size is dominated by protocol-fixed
        // jagged/basefold verification overhead tied to `core_max_log_row_count()` (currently 22
        // -- see `crates/hypercube/src/config.rs`), not by the guest program's actual cycle
        // count, so even this trivial program produced event counts the old hardcoded shapes
        // (topping out around 2^20) were nowhere near large enough for: MemoryConst=1008930,
        // MemoryVar=1221969, BaseAlu=5327, ExtAlu=2556571, Poseidon2WideDeg3=251099, Select=160,
        // PrefixSumChecks=64464.
        //
        // Critically, every entry here must also stay `<= recursion_max_log_row_count()`
        // (currently 22, `crates/stark/src/opts.rs`'s `RECURSION_MAX_SHARD_SIZE`): a chip's
        // actual row count at runtime is driven by the *shape* assigned here (the recursion
        // runtime pads each chip's event count up to it), while the jagged PCS's `PaddedMle`
        // storage is fixed to `2^recursion_max_log_row_count()` regardless of what the shape
        // claims -- a shape entry exceeding that ceiling padded to more real rows than the PCS
        // config could ever hold, which is exactly what the first version of this fix got wrong
        // (it treated the shape table as independent headroom on top of the natural measurements
        // without checking it against the machine-wide row cap). `ExtAlu`'s natural requirement
        // (22 bits) already equals that ceiling, so it has zero slack across all three tiers.
        // These have not yet been validated against a program with multiple shards or deferred
        // proofs, which could plausibly need a larger `recursion_max_log_row_count()` outright
        // (see [[stage4-1-verifier-migration]] for the broader pattern of still-provisional
        // recursion-layer constants in this migration).
        let allowed_shapes = [
            // Fastest shape.
            [
                (mem_var.clone(), 21),
                (select.clone(), 9),
                (mem_const.clone(), 20),
                (base_alu.clone(), 13),
                (ext_alu.clone(), 22),
                (poseidon2_wide.clone(), 18),
                (prefix_sum_checks.clone(), 17),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            // Second fastest shape.
            [
                (mem_var.clone(), 22),
                (select.clone(), 10),
                (mem_const.clone(), 21),
                (base_alu.clone(), 14),
                (ext_alu.clone(), 22),
                (poseidon2_wide.clone(), 19),
                (prefix_sum_checks.clone(), 18),
                (public_values.clone(), PUB_VALUES_LOG_HEIGHT),
            ],
            [
                (mem_var.clone(), 22),
                (select.clone(), 11),
                (mem_const.clone(), 22),
                (base_alu.clone(), 15),
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
