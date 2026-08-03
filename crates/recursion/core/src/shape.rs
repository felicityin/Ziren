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
        public_values::PublicValuesChip,
        select::SelectChip,
    },
    machine::RecursionAir,
    RecursionProgram, D,
};

/// The exact row count `PublicValues` gets padded to when a shape is configured. Real content is
/// tiny and essentially constant across programs (a handful of rows), so no tier variance is
/// needed here.
const PUB_VALUES_NUM_ROWS: usize = 16;

/// Pads `rows` in place to a target row count, mirroring
/// `zkm_core_machine::utils::pad_rows_fixed` -- except `fixed_num_rows` (when set) is an exact
/// row count rather than a log2 exponent, matching `RecursionProgram::fixed_num_rows`. Recursion
/// chips use this instead of the core-machine version specifically to avoid that log2
/// reinterpretation; core-machine's own shape-quantization mechanism was removed separately
/// (task #24) and every core-machine call site now always passes `None`.
pub fn pad_rows_fixed<R: Clone>(
    rows: &mut Vec<R>,
    row_fn: impl Fn() -> R,
    fixed_num_rows: Option<usize>,
    chip: &str,
) {
    let nb_rows = rows.len();
    let dummy_row = row_fn();
    let target = match fixed_num_rows {
        Some(target) => {
            assert!(
                nb_rows <= target,
                "{chip}: fixed rows is too small: got {nb_rows}, expected {target}"
            );
            target
        }
        None => nb_rows.next_power_of_two().max(16),
    };
    rows.resize(target, dummy_row);
}

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
                if *height > *shape.get(name).unwrap() {
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
        // 4x headroom over the union of every tier (mirrors the old log2 representation's
        // `+= 2`, i.e. two extra power-of-two doublings, now expressed directly since entries
        // are exact row counts rather than exponents).
        map.values_mut().for_each(|x| *x *= 4);
        map.insert("PublicValues".to_string(), PUB_VALUES_NUM_ROWS);
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
        // driven by the *shape* entry selected here (`fixed_num_rows`, `chips/alu_ext.rs` and
        // siblings), not by the real measured height directly -- so an oversized entry wastes
        // real rows. Entries are exact row counts, not log2 exponents: the underlying `Mle`/
        // `Tensor` machinery accepts any row count, and the jagged/stacked PCS's own alignment
        // requirement applies to the aggregate committed area across all chips, not any
        // individual chip's row count (see `RecursionProgram::fixed_num_rows`'s doc comment).
        // Every entry must still stay `<= 1 << RECURSION_MAX_LOG_ROW_COUNT`
        // (`crates/stark/src/opts.rs`): the jagged PCS's `PaddedMle` storage is fixed to that
        // size regardless of what a shape entry claims, so an entry exceeding it pads to more
        // real rows than the PCS config can hold, panicking in
        // `PaddedMle::padded`/`padded_with_zeros`.
        //
        // `fix_shape` tries each entry in order and only errors if every one is too small for
        // the real program, so an undersized "fastest" entry costs a fallback to a bigger tier
        // for that one program, not a crash -- "fastest" can stay calibrated tight to real
        // measured data rather than needing its own independent safety margin. `select` is
        // already at its real measured ceiling (no slack observed across sampled programs).
        // `prefix_sum_checks` is kept above its measured ceiling deliberately: real usage swings
        // by over an order of magnitude across sampled programs depending on precompile/syscall
        // mix, the widest per-program variance of any dimension here, so it's the one entry
        // where an unsampled program is most likely to exceed what's been measured so far.
        // "Fastest x2" sits between "fastest" and "fallback": every dimension is exactly double
        // "fastest"'s. `heights()` scales almost exactly linearly with `REDUCE_BATCH_SIZE` (each
        // additional child in a compress node is an independent verify-shard sub-circuit, so this
        // is a structural property of the circuit, not data-dependent) -- measured directly via a
        // real compress-node compile at arity 2 vs 4, every non-PublicValues dimension came out
        // within 0.02% of exactly 2x. "Fastest" itself stays calibrated to `REDUCE_BATCH_SIZE`'s
        // arity for `recursion_program`'s always-arity-1 case, which this tier leaves untouched
        // (it's still the first, tightest entry `fix_shape` tries).
        //
        // "Fallback" covers shards whose real heights don't fit "fastest" or "fastest x2", every
        // dimension capped at the hard `1 << RECURSION_MAX_LOG_ROW_COUNT` ceiling -- the most
        // headroom obtainable under that limit. If a real shard's measured height still exceeds
        // even this tier, `fix_shape` panics with "no shape found". These have not yet been
        // validated against deferred proofs, which could plausibly need a larger
        // `RECURSION_MAX_LOG_ROW_COUNT` outright.
        let allowed_shapes = [
            // Fastest shape.
            [
                (mem_var.clone(), 524_288),
                (select.clone(), 1_048_576),
                (mem_const.clone(), 131_072),
                (base_alu.clone(), 131_072),
                (ext_alu.clone(), 262_144),
                (poseidon2_wide.clone(), 131_072),
                (prefix_sum_checks.clone(), 524_288),
                (public_values.clone(), PUB_VALUES_NUM_ROWS),
            ],
            // Fastest x2 shape.
            [
                (mem_var.clone(), 1_048_576),
                (select.clone(), 2_097_152),
                (mem_const.clone(), 262_144),
                (base_alu.clone(), 262_144),
                (ext_alu.clone(), 524_288),
                (poseidon2_wide.clone(), 262_144),
                (prefix_sum_checks.clone(), 1_048_576),
                (public_values.clone(), PUB_VALUES_NUM_ROWS),
            ],
            // Fallback shape, with more headroom on the dimensions with the most real
            // per-program variance (`BaseAlu`, `Select`, `Poseidon2WideDeg3`). Every dimension
            // capped at `1 << 21` (`RECURSION_MAX_LOG_ROW_COUNT`) -- the maximum this tier can
            // offer.
            [
                (mem_var.clone(), 2_097_152),
                (select.clone(), 2_097_152),
                (mem_const.clone(), 2_097_152),
                (base_alu.clone(), 262_144),
                (ext_alu.clone(), 2_097_152),
                (poseidon2_wide.clone(), 1_048_576),
                (prefix_sum_checks.clone(), 524_288),
                (public_values.clone(), PUB_VALUES_NUM_ROWS),
            ],
        ]
        .map(HashMap::from)
        .to_vec();
        Self { allowed_shapes, _marker: PhantomData }
    }
}
