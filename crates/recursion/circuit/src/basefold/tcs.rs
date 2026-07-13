use itertools::Itertools;
use p3_field::FieldAlgebra;
use slop_tensor::Tensor;
use std::marker::PhantomData;
use zkm_recursion_compiler::ir::{Builder, Felt};

use crate::{hash::FieldHasherVariable, CircuitConfig};

/// An opening of a tensor commitment scheme.
pub struct RecursiveTensorCsOpening<F, CommitmentVariable> {
    /// The claimed values of the opening.
    pub values: Tensor<Felt<F>>,
    /// The proof of the opening.
    pub proof: Tensor<CommitmentVariable>,

    pub merkle_root: CommitmentVariable,

    pub log_height: usize,
    pub width: usize,
}

#[derive(Debug, Copy, PartialEq, Eq)]
pub struct RecursiveMerkleTreeTcs<C, HV>(pub PhantomData<(C, HV)>);

impl<C, HV> Clone for RecursiveMerkleTreeTcs<C, HV> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<C, HV> RecursiveMerkleTreeTcs<C, HV>
where
    C: CircuitConfig,
    HV: FieldHasherVariable<C>,
{
    pub fn verify_tensor_openings(
        builder: &mut Builder<C>,
        commit: &HV::DigestVariable,
        indices: &[Vec<C::Bit>],
        opening: &RecursiveTensorCsOpening<C::F, HV::DigestVariable>,
    ) {
        let log_height = builder.constant(C::F::from_canonical_usize(opening.log_height));
        let width = builder.constant(C::F::from_canonical_usize(opening.width));
        let hash = HV::hash(builder, &[log_height, width]);
        let expected_commit = HV::compress(builder, [opening.merkle_root, hash]);
        HV::assert_digest_eq(builder, expected_commit, *commit);

        for (i, (index, path)) in indices.iter().zip_eq(opening.proof.split()).enumerate() {
            let claimed_values_slices = opening.values.get(i).unwrap().as_slice().to_vec();
            let path = path.as_slice().to_vec();
            let digest = HV::hash(builder, &claimed_values_slices);

            verify_tensor_opening_path::<C, HV>(
                builder,
                &path,
                index,
                digest,
                opening.merkle_root,
            );
        }
    }
}

/// Verifies a single Merkle authentication path for a tensor commitment scheme opening.
///
/// This intentionally does *not* reuse `crate::merkle_tree::verify`: that gadget matches the
/// bit-reversed leaf ordering used by `zkm_recursion_circuit::merkle_tree::MerkleTree` (the
/// "allowed VK" aggregation tree), whereas the tensor commitment trees built by
/// `slop_merkle_tree::FieldMerkleTreeProver` (used for basefold/stacked-PCS commitments) store
/// leaves in natural (non-bit-reversed) order and build authentication paths by peeling off
/// index bits least-significant-bit first (see `FieldMerkleTreeProver::prove_openings_at_indices`
/// and `slop_merkle_tree::MerkleTreeTcs::verify_tensor_openings`). The `index` here comes from
/// `FieldChallengerVariable::sample_bits`, which is also least-significant-bit first, so it must
/// be consumed in that same order, without reversal.
fn verify_tensor_opening_path<C: CircuitConfig, HV: FieldHasherVariable<C>>(
    builder: &mut Builder<C>,
    path: &[HV::DigestVariable],
    index: &[C::Bit],
    value: HV::DigestVariable,
    merkle_root: HV::DigestVariable,
) {
    let mut value = value;
    for (sibling, bit) in path.iter().zip(index.iter()) {
        let sibling = *sibling;
        // If the index is odd, swap the order of [value, sibling].
        let new_pair = HV::select_chain_digest(builder, *bit, [value, sibling]);
        value = HV::compress(builder, new_pair);
    }
    HV::assert_digest_eq(builder, value, merkle_root);
}
