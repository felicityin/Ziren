//! Proof-shape bookkeeping.
//!
//! Most of what used to live here (`ZKMCompressProgramShape`, `check_shapes`, `build_vk_map`,
//! `build_vk_map_to_file`, and the `CoreShapeConfig`-driven `ZKMProofShape::{generate,
//! generate_maximal_shapes, dummy_vk_map}`) was infrastructure for the old FRI-era
//! shape-quantization/VK-determinism system: precompiling every possible recursion-program shape
//! ahead of time (via dummy witnesses) so a given proof shape always maps to the same
//! verification key, then hashing that catalog into `vk_map.bin`. `CoreShapeConfig` itself was
//! deleted from `zkm_core_machine` entirely (task #24 -- SP1 moved shape-fixing concerns to the
//! recursion layer), and the current `zkm_recursion_circuit::machine` witness-value types have no
//! `Shape`/`::dummy()` counterparts to replace it with. Redesigning this is out of scope for
//! 阶段5.1 (the user explicitly deferred regenerating VK artifacts) -- dropping the precomputed
//! shape cache just means every recursion/compress/shrink program gets (re)compiled from its real
//! witness on first use instead of being served from a warm, `vk_map.bin`-backed cache; slower,
//! not less correct.

use serde::{Deserialize, Serialize};
use zkm_hypercube::shape::OrderedShape;

/// The proof shapes that occur across a full core-to-wrap proof, used only for descriptive/debug
/// purposes now (e.g. `ZKMProver::run_e2e_prover_with_options`'s `COLLECT_SHAPES` diagnostic dump).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ZKMProofShape {
    Recursion(OrderedShape),
    Compress(Vec<OrderedShape>),
    Deferred(OrderedShape),
    Shrink(OrderedShape),
}
