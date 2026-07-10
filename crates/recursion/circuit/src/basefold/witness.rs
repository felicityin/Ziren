use slop_basefold::BasefoldProof;
use slop_challenger::{GrindingChallenger, IopCtx};
use slop_merkle_tree::{MerkleTreeOpeningAndProof, MerkleTreeTcsProof};
use slop_multilinear::Evaluations;
use slop_stacked::StackedBasefoldProof;
use slop_tensor::Tensor;
use zkm_recursion_compiler::ir::{Builder, Felt};
use zkm_stark::{InnerChallenge, InnerVal};

use crate::{
    basefold::{
        stacked::RecursiveStackedPcsProof, tcs::RecursiveTensorCsOpening, RecursiveBasefoldProof,
    },
    hash::FieldHasherVariable,
    witness::{WitnessWriter, Witnessable},
    CircuitConfig,
};

impl<C: CircuitConfig, T: Witnessable<C>> Witnessable<C> for Evaluations<T> {
    type WitnessVariable = Evaluations<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let round_evaluations = self.round_evaluations.read(builder);
        Evaluations { round_evaluations }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.round_evaluations.write(witness);
    }
}

impl<GC: IopCtx<F = C::F>, C: CircuitConfig<F = InnerVal>> Witnessable<C>
    for MerkleTreeOpeningAndProof<GC>
where
    GC::Digest: Witnessable<C>,
{
    type WitnessVariable =
        RecursiveTensorCsOpening<C::F, <GC::Digest as Witnessable<C>>::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let values: Tensor<Felt<C::F>> = self.values.read(builder);
        let proof = self.proof.read(builder);
        RecursiveTensorCsOpening::<C::F, <GC::Digest as Witnessable<C>>::WitnessVariable> {
            values,
            proof: proof.paths,
            merkle_root: proof.merkle_root,
            log_height: proof.log_tensor_height,
            width: proof.width,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.values.write(witness);
        self.proof.write(witness);
    }
}

impl<C, T> Witnessable<C> for MerkleTreeTcsProof<T>
where
    C: CircuitConfig,
    T: Witnessable<C>,
{
    type WitnessVariable = MerkleTreeTcsProof<T::WitnessVariable>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let paths = self.paths.read(builder);
        let merkle_root = self.merkle_root.read(builder);
        MerkleTreeTcsProof {
            paths,
            merkle_root,
            log_tensor_height: self.log_tensor_height,
            width: self.width,
        }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.paths.write(witness);
        self.merkle_root.write(witness);
    }
}

impl<C, GC> Witnessable<C> for BasefoldProof<GC>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge>,
    GC: IopCtx<F = C::F, EF = C::EF> + FieldHasherVariable<C>,
    <GC::Challenger as GrindingChallenger>::Witness: Witnessable<C, WitnessVariable = Felt<C::F>>,
    <GC as IopCtx>::Digest:
        Witnessable<C, WitnessVariable = <GC as FieldHasherVariable<C>>::DigestVariable>,
    MerkleTreeOpeningAndProof<GC>: Witnessable<
        C,
        WitnessVariable = RecursiveTensorCsOpening<
            C::F,
            <GC as FieldHasherVariable<C>>::DigestVariable,
        >,
    >,
{
    type WitnessVariable = RecursiveBasefoldProof<C, GC>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let univariate_messages = self.univariate_messages.read(builder);
        let fri_commitments = self.fri_commitments.read(builder);
        let component_polynomials_query_openings =
            self.component_polynomials_query_openings_and_proofs.read(builder);
        let query_phase_openings = self.query_phase_openings_and_proofs.read(builder);
        let final_poly = self.final_poly.read(builder);
        let pow_witness = self.pow_witness.read(builder);
        let batch_grinding_witness = self.batch_grinding_witness.read(builder);
        RecursiveBasefoldProof::<C, GC> {
            univariate_messages,
            fri_commitments,
            component_polynomials_query_openings_and_proofs: component_polynomials_query_openings,
            query_phase_openings_and_proofs: query_phase_openings,
            final_poly,
            pow_witness,
            batch_grinding_witness,
        }
    }
    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.univariate_messages.write(witness);
        self.fri_commitments.write(witness);
        self.component_polynomials_query_openings_and_proofs.write(witness);
        self.query_phase_openings_and_proofs.write(witness);
        self.final_poly.write(witness);
        self.pow_witness.write(witness);
        self.batch_grinding_witness.write(witness);
    }
}

impl<GC: IopCtx<F = C::F, EF = C::EF>, C, RecursivePcsProof> Witnessable<C>
    for StackedBasefoldProof<GC>
where
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge>,
    BasefoldProof<GC>: Witnessable<C, WitnessVariable = RecursivePcsProof>,
{
    type WitnessVariable = RecursiveStackedPcsProof<RecursivePcsProof, C::F, C::EF>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let batch_evaluations = self.batch_evaluations.read(builder);
        let pcs_proof = self.basefold_proof.read(builder);
        RecursiveStackedPcsProof::<RecursivePcsProof, C::F, C::EF> { pcs_proof, batch_evaluations }
    }

    fn write(&self, witness: &mut impl WitnessWriter<C>) {
        self.batch_evaluations.write(witness);
        self.basefold_proof.write(witness);
    }
}
