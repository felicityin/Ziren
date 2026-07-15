#![allow(unused_variables)]
use std::collections::BTreeSet;

use zkm_core_executor::ZKMContext;
use zkm_core_machine::io::ZKMStdin;
use zkm_hypercube::{config::default_fri_config, verifier::MerkleProof};

use crate::{
    Prover, ZKMProof, ZKMProofKind, ZKMProofWithPublicValues, ZKMProvingKey, ZKMVerificationError,
    ZKMVerifyingKey,
};
use anyhow::Result;
use p3_field::PrimeField;
use p3_koala_bear::KoalaBear;
use zkm_prover::{
    components::DefaultProverComponents,
    verify::{verify_groth16_bn254_public_inputs, verify_plonk_bn254_public_inputs},
    CompressAir, DvSnarkBn254Proof, Groth16Bn254Proof, HashableKey, PlonkBn254Proof, ZKMProver,
    ZKMReduceProofWrapper,
};
use zkm_recursion_circuit::dummy::{dummy_shard_proof, dummy_vk};

use super::{ProofOpts, ProverType};

/// An implementation of [crate::ProverClient] that can generate mock proofs.
pub struct MockProver {
    pub(crate) prover: ZKMProver,
}

impl MockProver {
    /// Creates a new [MockProver].
    pub fn new() -> Self {
        let prover = ZKMProver::new();
        Self { prover }
    }
}

impl Prover<DefaultProverComponents> for MockProver {
    fn id(&self) -> ProverType {
        ProverType::Mock
    }

    fn setup(&self, elf: &[u8]) -> (ZKMProvingKey, ZKMVerifyingKey) {
        let (pk, _, vk) = self.prover.setup(elf);
        (pk, vk)
    }

    fn zkm_prover(&self) -> &ZKMProver {
        &self.prover
    }

    fn prove_impl<'a>(
        &'a self,
        pk: &ZKMProvingKey,
        stdin: ZKMStdin,
        opts: ProofOpts,
        context: ZKMContext<'a>,
        kind: ZKMProofKind,
        _elf_id: Option<String>,
    ) -> Result<(ZKMProofWithPublicValues, u64)> {
        match kind {
            ZKMProofKind::Core => {
                let (public_values, _) = self.prover.execute(&pk.elf, &stdin, context)?;
                Ok((
                    ZKMProofWithPublicValues {
                        proof: ZKMProof::Core(vec![]),
                        public_values,
                        zkm_version: self.version().to_string(),
                    },
                    0,
                ))
            }
            ZKMProofKind::Compressed => {
                let (public_values, _) = self.prover.execute(&pk.elf, &stdin, context)?;

                // A syntactically-valid but cryptographically-meaningless shard proof: the mock
                // prover never actually verifies proof content (see `verify()` below), so an
                // empty-chip-set dummy proof is enough to satisfy the type.
                let shard_proof = dummy_shard_proof::<CompressAir<KoalaBear>>(
                    BTreeSet::new(),
                    1,
                    default_fri_config(),
                    4,
                    &[],
                    &[],
                );
                let reduce_vk = dummy_vk();
                let vk_merkle_proof = MerkleProof { index: 0, path: vec![] };

                let proof = ZKMProof::Compressed(Box::new(ZKMReduceProofWrapper {
                    vk: reduce_vk,
                    proof: shard_proof,
                    vk_merkle_proof,
                }));

                Ok((
                    ZKMProofWithPublicValues {
                        proof,
                        public_values,
                        zkm_version: self.version().to_string(),
                    },
                    0,
                ))
            }
            ZKMProofKind::Plonk => {
                let (public_values, _) = self.prover.execute(&pk.elf, &stdin, context)?;
                Ok((
                    ZKMProofWithPublicValues {
                        proof: ZKMProof::Plonk(PlonkBn254Proof {
                            public_inputs: [
                                pk.vk.hash_bn254().as_canonical_biguint().to_string(),
                                public_values.hash_bn254().to_string(),
                            ],
                            encoded_proof: "".to_string(),
                            raw_proof: "".to_string(),
                            plonk_vkey_hash: [0; 32],
                        }),
                        public_values,
                        zkm_version: self.version().to_string(),
                    },
                    0,
                ))
            }
            ZKMProofKind::Groth16 => {
                let (public_values, _) = self.prover.execute(&pk.elf, &stdin, context)?;
                Ok((
                    ZKMProofWithPublicValues {
                        proof: ZKMProof::Groth16(Groth16Bn254Proof {
                            public_inputs: [
                                pk.vk.hash_bn254().as_canonical_biguint().to_string(),
                                public_values.hash_bn254().to_string(),
                            ],
                            encoded_proof: "".to_string(),
                            raw_proof: "".to_string(),
                            groth16_vkey_hash: [0; 32],
                        }),
                        public_values,
                        zkm_version: self.version().to_string(),
                    },
                    0,
                ))
            }
            ZKMProofKind::DvSnark => {
                let (public_values, _) = self.prover.execute(&pk.elf, &stdin, context)?;
                Ok((
                    ZKMProofWithPublicValues {
                        proof: ZKMProof::DvSnark(DvSnarkBn254Proof {}),
                        public_values,
                        zkm_version: self.version().to_string(),
                    },
                    0,
                ))
            }
            ZKMProofKind::CompressToGroth16 => unreachable!(),
        }
    }

    fn verify(
        &self,
        bundle: &ZKMProofWithPublicValues,
        vkey: &ZKMVerifyingKey,
    ) -> Result<(), ZKMVerificationError> {
        match &bundle.proof {
            ZKMProof::Plonk(PlonkBn254Proof { public_inputs, .. }) => {
                verify_plonk_bn254_public_inputs(vkey, &bundle.public_values, public_inputs)
                    .map_err(ZKMVerificationError::Plonk)
            }
            ZKMProof::Groth16(Groth16Bn254Proof { public_inputs, .. }) => {
                verify_groth16_bn254_public_inputs(vkey, &bundle.public_values, public_inputs)
                    .map_err(ZKMVerificationError::Groth16)
            }
            _ => Ok(()),
        }
    }
}

impl Default for MockProver {
    fn default() -> Self {
        Self::new()
    }
}
