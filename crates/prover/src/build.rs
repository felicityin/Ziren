//! Outer/Bn254 wrap-circuit artifact building (Plonk/Groth16/DvSnark).
//!
//! Every non-trivial function below builds the gnark wrap circuit (`build_outer_circuit`,
//! removed) or produces a template proof to build it from (`dummy_proof`), both of which need
//! `OuterSC` to implement `slop_challenger::IopCtx` so it can flow through
//! `zkm_recursion_circuit::machine`'s now-hypercube-native `ZKMCompressWitnessValues`/
//! `ZKMWrapVerifier`/`OuterWitness` machinery. `OuterSC` still aliases the old FRI-era
//! `KoalaBearPoseidon2Outer`, which has no such impl -- a real replacement (`ZkmOuterGlobalContext`,
//! wired through `slop_bn254::Poseidon2Bn254GlobalConfig`) is task #57 and is out of scope for
//! 阶段5.1 (the user explicitly deferred regenerating VK/wrap artifacts). The public signatures
//! here are kept intact (crates/sdk's cpu/cuda provers call them directly), but the bodies that
//! actually need the missing `IopCtx` impl are stubbed out until task #57 lands.
use std::{
    fs::{metadata, File},
    io::Write,
    path::PathBuf,
};
use zkm_recursion_compiler::{config::OuterConfig, constraints::Constraint};

pub use zkm_recursion_core::stark::{outer_perm, zkm_dev_mode, zkm_imm_wrap_vk_mode};

pub use zkm_recursion_circuit::witness::{OuterWitness, Witnessable};

use zkm_recursion_gnark_ffi::{DvSnarkBn254Prover, Groth16Bn254Prover, PlonkBn254Prover};
use zkm_stark::{ShardProof, StarkVerifyingKey};

use crate::OuterSC;

pub const PART_STARK_VK_PATH: &str = "part_stark_vk.bin";

/// Tries to build the PLONK artifacts inside the development directory.
pub fn try_build_plonk_bn254_artifacts_dev(
    template_vk: &StarkVerifyingKey<OuterSC>,
    template_proof: &ShardProof<OuterSC>,
) -> PathBuf {
    let build_dir = plonk_bn254_artifacts_dev_dir();
    println!("[zkm] building plonk bn254 artifacts in development mode");
    build_plonk_bn254_artifacts(template_vk, template_proof, &build_dir);
    build_dir
}

/// Tries to build the groth16 bn254 artifacts in the current environment.
pub fn try_build_groth16_bn254_artifacts_dev(
    template_vk: &StarkVerifyingKey<OuterSC>,
    template_proof: &ShardProof<OuterSC>,
) -> PathBuf {
    let build_dir = groth16_bn254_artifacts_dev_dir();
    println!("[zkm] building groth16 bn254 artifacts in development mode");
    build_groth16_bn254_artifacts(template_vk, template_proof, &build_dir);
    build_dir
}

/// Tries to build the dv-snark bn254 artifacts in the current environment.
pub fn try_build_dvsnark_bn254_artifacts_dev(
    template_vk: &StarkVerifyingKey<OuterSC>,
    template_proof: &ShardProof<OuterSC>,
    store_dir: &PathBuf,
) -> PathBuf {
    tracing::info!("build dvsnark artifacts dev");
    let build_dir = dvsnark_bn254_artifacts_dev_dir();

    let r1cs_to_dvsnark_path = store_dir.join("r1cs_to_dvsnark");
    let r1cs_cached_path = store_dir.join("r1cs_cached");

    let mut r1cs_to_dvsnark_content_exist = false;
    if let Ok(md) = metadata(&r1cs_to_dvsnark_path) {
        if md.len() > 1024 {
            r1cs_to_dvsnark_content_exist = true;
        }
    }

    let mut r1cs_cached_content_exist = false;
    if let Ok(md) = metadata(&r1cs_cached_path) {
        if md.len() > 1024 {
            r1cs_cached_content_exist = true;
        }
    }

    if r1cs_cached_content_exist && r1cs_to_dvsnark_content_exist {
        println!("[zkm] build dir contains cached r1cs");
        return build_dir; // early return if content already exist
    }

    println!("[zkm] building dv-snark bn254 artifacts in development mode");
    build_dvsnark_bn254_artifacts(template_vk, template_proof, &build_dir, store_dir);
    build_dir
}

/// Gets the directory where the PLONK artifacts are installed in development mode.
pub fn plonk_bn254_artifacts_dev_dir() -> PathBuf {
    dirs::home_dir().unwrap().join(".zkm").join("circuits").join("dev")
}

/// Gets the directory where the groth16 artifacts are installed in development mode.
pub fn groth16_bn254_artifacts_dev_dir() -> PathBuf {
    dirs::home_dir().unwrap().join(".zkm").join("circuits").join("dev")
}

/// Gets the directory where the dv-snark artifacts are installed in development mode.
pub fn dvsnark_bn254_artifacts_dev_dir() -> PathBuf {
    dirs::home_dir().unwrap().join(".zkm").join("circuits").join("dev")
}

/// Build the plonk bn254 artifacts to the given directory for the given verification key and
/// template proof.
pub fn build_plonk_bn254_artifacts(
    template_vk: &StarkVerifyingKey<OuterSC>,
    template_proof: &ShardProof<OuterSC>,
    build_dir: impl Into<PathBuf>,
) {
    let build_dir = build_dir.into();
    std::fs::create_dir_all(&build_dir).expect("failed to create build directory");
    let (constraints, witness) = build_constraints_and_witness(template_vk, template_proof);
    PlonkBn254Prover::build(constraints, witness, build_dir);
}

/// Build the groth16 bn254 artifacts to the given directory for the given verification key and
/// template proof.
pub fn build_groth16_bn254_artifacts(
    template_vk: &StarkVerifyingKey<OuterSC>,
    template_proof: &ShardProof<OuterSC>,
    build_dir: impl Into<PathBuf>,
) {
    let build_dir = build_dir.into();
    std::fs::create_dir_all(&build_dir).expect("failed to create build directory");
    let (constraints, witness) = build_constraints_and_witness(template_vk, template_proof);
    Groth16Bn254Prover::build(constraints, witness, build_dir.clone());

    // Serialize the part vk to a file
    let serialized = bincode::serialize(&template_vk.part_vk()).unwrap();
    let path = build_dir.join(PART_STARK_VK_PATH);
    let mut file = File::create(path).unwrap();
    file.write_all(&serialized).unwrap();
}

/// Build the dv-snark bn254 artifacts to the given directory for the given verification key and
/// template proof.
pub fn build_dvsnark_bn254_artifacts(
    template_vk: &StarkVerifyingKey<OuterSC>,
    template_proof: &ShardProof<OuterSC>,
    build_dir: impl Into<PathBuf>,
    store_dir: impl Into<PathBuf>,
) {
    let build_dir = build_dir.into();
    let store_dir = store_dir.into();
    std::fs::create_dir_all(&build_dir).expect("failed to create build directory");
    std::fs::create_dir_all(&store_dir).expect("failed to create store directory");
    let (constraints, witness) = build_constraints_and_witness(template_vk, template_proof);
    DvSnarkBn254Prover::build(constraints, witness, build_dir, store_dir);
}

/// Builds the plonk bn254 artifacts to the given directory.
///
/// This may take a while as it needs to first generate a dummy proof and then it needs to compile
/// the circuit.
pub fn build_plonk_bn254_artifacts_with_dummy(build_dir: impl Into<PathBuf>) {
    let (wrap_vk, wrapped_proof) = dummy_proof();
    crate::build::build_plonk_bn254_artifacts(&wrap_vk, &wrapped_proof, build_dir.into());
}

/// Builds the groth16 bn254 artifacts to the given directory.
///
/// This may take a while as it needs to first generate a dummy proof and then it needs to compile
/// the circuit.
pub fn build_groth16_bn254_artifacts_with_dummy(build_dir: impl Into<PathBuf>) {
    let (wrap_vk, wrapped_proof) = dummy_proof();
    crate::build::build_groth16_bn254_artifacts(&wrap_vk, &wrapped_proof, build_dir.into());
}

/// Build the verifier constraints and template witness for the circuit.
///
/// Blocked on task #57 (`ZkmOuterGlobalContext`) -- see the module doc comment.
pub fn build_constraints_and_witness(
    _template_vk: &StarkVerifyingKey<OuterSC>,
    _template_proof: &ShardProof<OuterSC>,
) -> (Vec<Constraint>, OuterWitness<OuterConfig>) {
    unimplemented!(
        "outer/Bn254 wrap circuit construction is blocked on task #57 (ZkmOuterGlobalContext)"
    )
}

/// Generate a dummy proof that we can use to build the circuit. We need this to know the shape of
/// the proof.
///
/// Blocked on task #57 (`ZkmOuterGlobalContext`) -- see the module doc comment.
pub fn dummy_proof() -> (StarkVerifyingKey<OuterSC>, ShardProof<OuterSC>) {
    unimplemented!(
        "outer/Bn254 wrap circuit construction is blocked on task #57 (ZkmOuterGlobalContext)"
    )
}
