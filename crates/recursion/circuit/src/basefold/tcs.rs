use itertools::Itertools;
use p3_field::FieldAlgebra;
use slop_tensor::Tensor;
use std::marker::PhantomData;
use zkm_recursion_compiler::ir::{Builder, Felt};

use crate::{
    hash::FieldHasherVariable, merkle_tree::verify, stark::MerkleProofVariable, CircuitConfig,
};

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

            verify::<C, HV>(
                builder,
                MerkleProofVariable { index: index.clone(), path },
                digest,
                opening.merkle_root,
            );
        }
    }
}
