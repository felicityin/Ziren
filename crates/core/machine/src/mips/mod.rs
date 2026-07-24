use crate::{
    adapter::StateBumpChip,
    global::GlobalChip,
    memory::{MemoryBumpChip, MemoryChipType, MemoryLocalChip},
    syscall::precompiles::{
        fptower::{Fp2AddSubAssignChip, Fp2MulAssignChip, FpOpChip},
        poseidon2::Poseidon2PermuteChip,
    },
};
use core::fmt;
use hashbrown::HashMap;
pub use mips_chips::*;
use p3_field::PrimeField32;
use strum_macros::{EnumDiscriminants, EnumIter};
use zkm_curves::weierstrass::{bls12_381::Bls12381BaseField, bn254::Bn254BaseField};
use zkm_hypercube::{
    air::{LookupScope, MachineAir, PicusInfo},
    Chip,
};
// TODO(zkm-hypercube): `StarkGenericConfig`/`StarkMachine`/`ZKM_PROOF_NUM_PV_ELTS` have no
// equivalent yet (no shard-proving driver exists in zkm-hypercube). `MipsAir::machine` below
// is commented out until that lands; these old-backend imports it needed are commented out too.
// use zkm_stark::{air::ZKM_PROOF_NUM_PV_ELTS, StarkGenericConfig, StarkMachine};

/// A module for importing all the different MIPS chips.
pub(crate) mod mips_chips {
    pub use crate::{
        alu::{
            AddChip, AddNoopChip, AddiChip, AluX0Chip, BitwiseChip, CloClzChip, DivRemChip,
            LtChip, MulChip, ShiftLeft, ShiftRightChip, SltiChip, SubChip,
        },
        bytes::ByteChip,
        control_flow::{BranchChip, JumpChip},
        memory::{LoadWordChip, MemoryGlobalChip, MemoryInstructionsChip, StoreWordChip},
        misc::{MiscInstrsChip, MovCondChip},
        program::ProgramChip,
        syscall::{
            chip::SyscallChip,
            instructions::SyscallInstrsChip,
            precompiles::{
                edwards::{EdAddAssignChip, EdDecompressChip},
                keccak_sponge::KeccakSpongeChip,
                sha256::{
                    ShaCompressChip, ShaCompressControlChip, ShaExtendChip, ShaExtendControlChip,
                },
                sys_linux::SysLinuxChip,
                u256x2048_mul::U256x2048MulChip,
                uint256::Uint256MulChip,
                weierstrass::{
                    WeierstrassAddAssignChip, WeierstrassDecompressChip,
                    WeierstrassDoubleAssignChip,
                },
            },
        },
    };
    pub use zkm_curves::{
        edwards::{ed25519::Ed25519Parameters, EdwardsCurve},
        weierstrass::{
            bls12_381::Bls12381Parameters, bn254::Bn254Parameters, secp256k1::Secp256k1Parameters,
            secp256r1::Secp256r1Parameters, SwCurve,
        },
    };
}

/// The maximum log number of shards in core.
pub const MAX_LOG_NUMBER_OF_SHARDS: usize = 16;

/// The maximum number of shards in core.
pub const MAX_NUMBER_OF_SHARDS: usize = 1 << MAX_LOG_NUMBER_OF_SHARDS;

/// An AIR for encoding MIPS execution.
///
/// This enum contains all the different AIRs that are used in the Ziren IOP. Each variant is
/// a different AIR that is used to encode a different part of the Ziren execution, and the
/// different AIR variants have a joint lookup argument.
#[derive(zkm_derive::MachineAir, EnumDiscriminants)]
#[strum_discriminants(derive(Hash, EnumIter))]
pub enum MipsAir<F: PrimeField32> {
    /// An AIR that contains a preprocessed program table and a lookup for the instructions.
    Program(ProgramChip),
    /// An AIR for the register-form MIPS ADD instruction.
    Add(AddChip),
    /// An AIR for the immediate-form MIPS ADDI/ADDIU instruction.
    Addi(AddiChip),
    /// An AIR for the fully-immediate MIPS ADD shape (SYNC/Pref).
    AddNoop(AddNoopChip),
    /// An AIR for the MIPS SUB instruction.
    Sub(SubChip),
    /// An AIR for the shared add/sub-to-register-0 shape (see `AluX0Chip`'s doc comment).
    AluX0(AluX0Chip),
    /// An AIR for MIPS Bitwise instructions.
    Bitwise(BitwiseChip),
    /// An AIR for MIPS Mul instruction.
    Mul(MulChip),
    /// An AIR for MIPS Div and Rem instructions.
    DivRem(DivRemChip),
    /// An AIR for MIPS Lt instruction.
    Lt(LtChip),
    /// An AIR for the immediate-form MIPS SLTI/SLTIU instruction.
    Slti(SltiChip),
    /// An AIR for MIPS CLO and CLZ instruction.
    CloClz(CloClzChip),
    /// An AIR for MIPS SLL instruction.
    ShiftLeft(ShiftLeft),
    /// An AIR for MIPS SRL and SRA instruction.
    ShiftRight(ShiftRightChip),
    /// A lookup table for byte operations.
    ByteLookup(ByteChip<F>),
    /// An AIR for MIPS Branch instructions.
    Branch(BranchChip),
    /// An AIR for MIPS Jump instructions.
    Jump(JumpChip),
    /// An AIR for the rare MIPS memory instructions (everything except LW/SW).
    MemoryInstrs(MemoryInstructionsChip),
    /// An AIR for the word-aligned MIPS load instruction (LW).
    LoadWord(LoadWordChip),
    /// An AIR for the word-aligned MIPS store instruction (SW).
    StoreWord(StoreWordChip),
    /// An AIR for MIPS mov condition instructions.
    MovCond(MovCondChip),
    /// An AIR for MIPS misc instructions.
    MiscInstrs(MiscInstrsChip),
    /// An AIR proving `clk_high` transitions (see [`StateBumpChip`]'s doc comment).
    StateBump(StateBumpChip),
    /// An AIR proving per-register `clk_high` realignment (see [`MemoryBumpChip`]'s doc comment).
    MemoryBump(MemoryBumpChip),
    /// An AIR for MIPS syscall instructions.
    SyscallInstrs(SyscallInstrsChip),
    /// A table for initializing the global memory state.
    MemoryGlobalInit(MemoryGlobalChip),
    /// A table for finalizing the global memory state.
    MemoryGlobalFinal(MemoryGlobalChip),
    /// A table for the local memory state.
    MemoryLocal(MemoryLocalChip),
    /// A table for all the syscall invocations.
    SyscallCore(SyscallChip),
    /// A table for all the precompile invocations.
    SyscallPrecompile(SyscallChip),
    /// A table for all the global lookups.
    Global(GlobalChip),
    /// Brackets a `SHA_EXTEND` syscall's worker chain (see [`Sha256Extend`]).
    Sha256ExtendControl(ShaExtendControlChip),
    /// A precompile for sha256 extend.
    Sha256Extend(ShaExtendChip),
    /// Brackets a `SHA_COMPRESS` syscall's worker chain (see [`Sha256Compress`]).
    Sha256CompressControl(ShaCompressControlChip),
    /// A precompile for sha256 compress.
    Sha256Compress(ShaCompressChip),
    /// A precompile for addition on the Elliptic curve ed25519.
    Ed25519Add(EdAddAssignChip<EdwardsCurve<Ed25519Parameters>>),
    /// A precompile for decompressing a point on the Edwards curve ed25519.
    Ed25519Decompress(EdDecompressChip<Ed25519Parameters>),
    /// A precompile for decompressing a point on the K256 curve.
    K256Decompress(WeierstrassDecompressChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for decompressing a point on the P256 curve.
    P256Decompress(WeierstrassDecompressChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256k1.
    Secp256k1Add(WeierstrassAddAssignChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256k1.
    Secp256k1Double(WeierstrassDoubleAssignChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256r1.
    Secp256r1Add(WeierstrassAddAssignChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256r1.
    Secp256r1Double(WeierstrassDoubleAssignChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for the Poseidon2 permutation
    Poseidon2Permute(Poseidon2PermuteChip),
    /// A precompile for the Keccak Sponge
    KeccakSponge(KeccakSpongeChip),
    /// A precompile for addition on the Elliptic curve bn254.
    Bn254Add(WeierstrassAddAssignChip<SwCurve<Bn254Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bn254.
    Bn254Double(WeierstrassDoubleAssignChip<SwCurve<Bn254Parameters>>),
    /// A precompile for addition on the Elliptic curve bls12_381.
    Bls12381Add(WeierstrassAddAssignChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bls12_381.
    Bls12381Double(WeierstrassDoubleAssignChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for uint256 mul.
    Uint256Mul(Uint256MulChip),
    /// A precompile for u256x2048 mul.
    U256x2048Mul(U256x2048MulChip),
    /// A precompile for decompressing a point on the BLS12-381 curve.
    Bls12381Decompress(WeierstrassDecompressChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for BLS12-381 fp operation.
    Bls12381Fp(FpOpChip<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 multiplication.
    Bls12381Fp2Mul(Fp2MulAssignChip<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 addition/subtraction.
    Bls12381Fp2AddSub(Fp2AddSubAssignChip<Bls12381BaseField>),
    /// A precompile for BN-254 fp operation.
    Bn254Fp(FpOpChip<Bn254BaseField>),
    /// A precompile for BN-254 fp2 multiplication.
    Bn254Fp2Mul(Fp2MulAssignChip<Bn254BaseField>),
    /// A precompile for BN-254 fp2 addition/subtraction.
    Bn254Fp2AddSub(Fp2AddSubAssignChip<Bn254BaseField>),
    /// A precompile for Linux Syscall.
    SysLinux(SysLinuxChip),
}

impl<F: PrimeField32> MipsAir<F> {
    // TODO(zkm-hypercube): no shard-proving driver / StarkMachine equivalent exists yet.
    // pub fn machine<SC: StarkGenericConfig<Val = F>>(config: SC) -> StarkMachine<SC, Self> {
    //     let chips = Self::chips();
    //     StarkMachine::new(config, chips, ZKM_PROOF_NUM_PV_ELTS)
    // }

    /// Builds the zkm-hypercube [`zkm_hypercube::Machine`] over all MIPS chips.
    ///
    /// Registers a curated, finite set of chip clusters rather than adding a catch-all covering
    /// every chip: `Machine::smallest_cluster` (called with `.unwrap()` in
    /// `generate_main_traces`) panics if a shard's chip set isn't a subset of any cluster below.
    /// This is intentional -- it relies on (and cross-checks) the checkpoint/deferred-event
    /// splitting invariant that a shard only ever combines ordinary CPU execution with the
    /// deferred global-memory-init/finalize events and/or the small set of precompiles below, or
    /// executes exactly one precompile family on its own, never an arbitrary combination. A base
    /// `core_cluster` (ordinary CPU execution: ALU, control flow, memory access, syscall
    /// dispatch) is combinatorially extended with a small, curated set of precompile/
    /// memory-boundary chip groups (`core_cluster_exts`) so that a shard combining ordinary CPU
    /// execution with the deferred global-memory-init/finalize events, or with one of a few
    /// common precompiles, gets a cluster sized for just that combination. This matters beyond
    /// native proving time -- for a shard verified inside a recursion circuit, the DSL/IR-compiled
    /// verifier does real per-chip constraint/interaction-evaluation work for every chip in the
    /// chosen cluster (`RecursiveShardVerifier::verify_shard`), even for chips with an all-zero
    /// padded trace, so an unnecessarily large cluster can blow the compiled circuit's own
    /// row*column area past the jagged-PCS protocol's bound (task #48).
    /// - `core_cluster`: `Program`/`Byte`/`Global` plus every chip that's part of ordinary CPU
    ///   execution. Covers the overwhelming majority of shards in a long-running program, which
    ///   just keep executing user code.
    /// - `core_clusters`: given E extension groups (`core_cluster_exts`), `core_cluster` with no
    ///   extension (E choose 0), with exactly one extension group added (E choose 1), and with
    ///   every extension group added (E choose E).
    /// - `core_cluster_special`: one specific extra combination beyond the combinatorics above --
    ///   `core_cluster` plus both the memory-boundary pair and the SHA-256/Uint256 precompiles
    ///   together, for a shard that both finalizes deferred global memory and executes those
    ///   precompiles inline.
    /// - `memory_boundary_cluster`: `Program`/`Byte`/`Global` plus `MemoryGlobalInit`/
    ///   `MemoryGlobalFinal`, for the (typically one) shard that commits deferred global memory
    ///   init/finalize events with no CPU activity of its own.
    /// - `precompile_clusters`: `base_precompile_cluster` (`Program`/`Byte`/`Global`/
    ///   `SyscallPrecompile`/`MemoryLocal`, deliberately without any of `core_cluster`'s CPU/ALU
    ///   chips) extended with exactly one precompile family, for a deferred shard that executes
    ///   only that precompile's worker chips and no ordinary CPU code of its own -- the checkpoint
    ///   splitting that produces deferred shards keeps different precompiles' events in separate
    ///   shards, so (unlike `core_cluster_exts`) these stay one precompile family per cluster
    ///   rather than combinatorial.
    pub fn hypercube_machine() -> zkm_hypercube::Machine<F, Self>
    where
        F: slop_algebra::Field,
    {
        use itertools::Itertools;
        use std::collections::BTreeSet;
        use strum::IntoEnumIterator;
        use MipsAirDiscriminants::*;

        let chips = Self::chips();
        let by_variant: HashMap<MipsAirDiscriminants, Chip<F, Self>> =
            chips.iter().map(|c| (c.air.as_ref().into(), c.clone())).collect();
        assert_eq!(
            by_variant.len(),
            MipsAirDiscriminants::iter().len(),
            "every MipsAir variant must have exactly one chip in Self::chips()"
        );
        let cluster = |variants: &[MipsAirDiscriminants]| -> BTreeSet<Chip<F, Self>> {
            variants.iter().map(|v| by_variant[v].clone()).collect()
        };
        let extend = |base: &BTreeSet<Chip<F, Self>>,
                      variants: &[MipsAirDiscriminants]|
         -> BTreeSet<Chip<F, Self>> {
            let mut set = base.clone();
            set.extend(variants.iter().map(|v| by_variant[v].clone()));
            set
        };

        let core_cluster = cluster(&[
            Program,
            ByteLookup,
            Global,
            Add,
            Addi,
            AddNoop,
            Sub,
            AluX0,
            Bitwise,
            Mul,
            ShiftRight,
            ShiftLeft,
            Lt,
            Slti,
            DivRem,
            CloClz,
            Branch,
            Jump,
            MiscInstrs,
            MovCond,
            StateBump,
            MemoryBump,
            MemoryInstrs,
            LoadWord,
            StoreWord,
            SyscallCore,
            SyscallInstrs,
            MemoryLocal,
        ]);
        let memory_boundary_cluster =
            cluster(&[Program, ByteLookup, Global, MemoryGlobalInit, MemoryGlobalFinal]);

        // Chip groups that may extend `core_cluster`. Deliberately small (one entry per
        // precompile would work too, but combinatorial growth isn't worth it for precompiles
        // that rarely co-occur with CPU execution in the same shard).
        let core_cluster_exts: Vec<&[MipsAirDiscriminants]> = vec![
            &[MemoryGlobalInit, MemoryGlobalFinal],
            &[Bls12381Fp],
            &[Bn254Fp],
            &[Sha256ExtendControl, Sha256Extend, Sha256CompressControl, Sha256Compress],
            &[Uint256Mul],
            &[Poseidon2Permute],
        ];
        let core_clusters = [0usize, 1, core_cluster_exts.len()]
            .into_iter()
            .flat_map(|k| core_cluster_exts.iter().copied().combinations(k).collect::<Vec<_>>())
            .map(|ext_set| {
                ext_set
                    .into_iter()
                    .fold(core_cluster.clone(), |set, variants| extend(&set, variants))
            });

        // A specific extra combination beyond the `core_cluster_exts` combinatorics above: a
        // shard that both commits the deferred global-memory-init/finalize events and executes
        // SHA-256/Uint256 precompiles inline.
        let core_cluster_special = extend(
            &core_cluster,
            &[
                MemoryGlobalInit,
                MemoryGlobalFinal,
                Sha256ExtendControl,
                Sha256Extend,
                Sha256CompressControl,
                Sha256Compress,
                Uint256Mul,
            ],
        );

        let base_precompile_cluster =
            cluster(&[Program, ByteLookup, Global, SyscallPrecompile, MemoryLocal]);

        // One entry per precompile family (a syscall's worker chip plus its control chip, if
        // any), each its own cluster -- unlike `core_cluster_exts`, these aren't combined,
        // since a deferred precompile-only shard only ever contains events for one precompile.
        let precompile_clusters: Vec<&[MipsAirDiscriminants]> = vec![
            &[Sha256ExtendControl, Sha256Extend],
            &[Sha256CompressControl, Sha256Compress],
            &[Ed25519Add],
            &[Ed25519Decompress],
            &[K256Decompress],
            &[Secp256k1Add],
            &[Secp256k1Double],
            &[P256Decompress],
            &[Secp256r1Add],
            &[Secp256r1Double],
            &[Poseidon2Permute],
            &[KeccakSponge],
            &[Bn254Add],
            &[Bn254Double],
            &[Bls12381Add],
            &[Bls12381Double],
            &[Uint256Mul],
            &[U256x2048Mul],
            &[Bls12381Decompress],
            &[Bls12381Fp],
            &[Bls12381Fp2Mul],
            &[Bls12381Fp2AddSub],
            &[Bn254Fp],
            &[Bn254Fp2Mul],
            &[Bn254Fp2AddSub],
            &[SysLinux],
        ];
        let precompile_clusters = precompile_clusters
            .into_iter()
            .map(|variants| extend(&base_precompile_cluster, variants));

        let clusters = core_clusters
            .chain(std::iter::once(core_cluster_special))
            .chain(std::iter::once(memory_boundary_cluster))
            .chain(precompile_clusters)
            .collect::<Vec<_>>();

        let shape = zkm_hypercube::MachineShape::new(clusters);
        zkm_hypercube::Machine::new(chips, zkm_hypercube::air::ZKM_PROOF_NUM_PV_ELTS, shape)
    }

    /// Get all the different MIPS AIRs.
    pub fn chips() -> Vec<Chip<F, Self>> {
        let (chips, _) = Self::get_chips_and_costs();
        chips
    }

    /// Get all the costs of the different MIPS AIRs.
    pub fn costs() -> HashMap<String, u64> {
        let (_, costs) = Self::get_chips_and_costs();
        costs
    }

    /// Get all the different MIPS AIRs and their costs.
    pub fn get_airs_and_costs() -> (Vec<Self>, HashMap<String, u64>) {
        let (chips, costs) = Self::get_chips_and_costs();
        (chips.into_iter().map(|chip| chip.into_inner().unwrap()).collect(), costs)
    }

    /// Get all the different MIPS chips and their costs.
    pub fn get_chips_and_costs() -> (Vec<Chip<F, Self>>, HashMap<String, u64>) {
        let mut costs: HashMap<String, u64> = HashMap::new();

        // The order of the chips is used to determine the order of trace generation.
        let mut chips = vec![];
        let program = Chip::new(MipsAir::Program(ProgramChip::default()));
        costs.insert(program.name(), program.cost());
        chips.push(program);

        let sha_extend_control =
            Chip::new(MipsAir::Sha256ExtendControl(ShaExtendControlChip::default()));
        costs.insert(sha_extend_control.name(), sha_extend_control.cost());
        chips.push(sha_extend_control);

        let sha_extend = Chip::new(MipsAir::Sha256Extend(ShaExtendChip::default()));
        costs.insert(sha_extend.name(), 48 * sha_extend.cost());
        chips.push(sha_extend);

        let sha_compress_control =
            Chip::new(MipsAir::Sha256CompressControl(ShaCompressControlChip::default()));
        costs.insert(sha_compress_control.name(), sha_compress_control.cost());
        chips.push(sha_compress_control);

        let sha_compress = Chip::new(MipsAir::Sha256Compress(ShaCompressChip::default()));
        costs.insert(sha_compress.name(), 80 * sha_compress.cost());
        chips.push(sha_compress);

        let ed_add_assign = Chip::new(MipsAir::Ed25519Add(EdAddAssignChip::<
            EdwardsCurve<Ed25519Parameters>,
        >::new()));
        costs.insert(ed_add_assign.name(), ed_add_assign.cost());
        chips.push(ed_add_assign);

        let ed_decompress =
            Chip::new(MipsAir::Ed25519Decompress(EdDecompressChip::<Ed25519Parameters>::default()));
        costs.insert(ed_decompress.name(), ed_decompress.cost());
        chips.push(ed_decompress);

        let k256_decompress = Chip::new(MipsAir::K256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256k1Parameters>,
        >::with_lsb_rule()));
        costs.insert(k256_decompress.name(), k256_decompress.cost());
        chips.push(k256_decompress);

        let secp256k1_add_assign = Chip::new(MipsAir::Secp256k1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256k1Parameters>,
        >::new()));
        costs.insert(secp256k1_add_assign.name(), secp256k1_add_assign.cost());
        chips.push(secp256k1_add_assign);

        let secp256k1_double_assign =
            Chip::new(MipsAir::Secp256k1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256k1Parameters>,
            >::new()));
        costs.insert(secp256k1_double_assign.name(), secp256k1_double_assign.cost());
        chips.push(secp256k1_double_assign);

        let p256_decompress = Chip::new(MipsAir::P256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256r1Parameters>,
        >::with_lsb_rule()));
        costs.insert(p256_decompress.name(), p256_decompress.cost());
        chips.push(p256_decompress);

        let secp256r1_add_assign = Chip::new(MipsAir::Secp256r1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256r1Parameters>,
        >::new()));
        costs.insert(secp256r1_add_assign.name(), secp256r1_add_assign.cost());
        chips.push(secp256r1_add_assign);

        let secp256r1_double_assign =
            Chip::new(MipsAir::Secp256r1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256r1Parameters>,
            >::new()));
        costs.insert(secp256r1_double_assign.name(), secp256r1_double_assign.cost());
        chips.push(secp256r1_double_assign);

        let poseidon2_permute = Chip::new(MipsAir::Poseidon2Permute(Poseidon2PermuteChip::new()));
        costs.insert(poseidon2_permute.name(), poseidon2_permute.cost());
        chips.push(poseidon2_permute);

        let keccak_sponge = Chip::new(MipsAir::KeccakSponge(KeccakSpongeChip::new()));
        costs.insert(keccak_sponge.name(), 24 * keccak_sponge.cost());
        chips.push(keccak_sponge);

        let bn254_add_assign = Chip::new(MipsAir::Bn254Add(WeierstrassAddAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        costs.insert(bn254_add_assign.name(), bn254_add_assign.cost());
        chips.push(bn254_add_assign);

        let bn254_double_assign = Chip::new(MipsAir::Bn254Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        costs.insert(bn254_double_assign.name(), bn254_double_assign.cost());
        chips.push(bn254_double_assign);

        let bls12381_add = Chip::new(MipsAir::Bls12381Add(WeierstrassAddAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        costs.insert(bls12381_add.name(), bls12381_add.cost());
        chips.push(bls12381_add);

        let bls12381_double = Chip::new(MipsAir::Bls12381Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        costs.insert(bls12381_double.name(), bls12381_double.cost());
        chips.push(bls12381_double);

        let uint256_mul = Chip::new(MipsAir::Uint256Mul(Uint256MulChip::default()));
        costs.insert(uint256_mul.name(), uint256_mul.cost());
        chips.push(uint256_mul);

        let u256x2048_mul = Chip::new(MipsAir::U256x2048Mul(U256x2048MulChip::default()));
        costs.insert(u256x2048_mul.name(), u256x2048_mul.cost());
        chips.push(u256x2048_mul);

        let bls12381_fp = Chip::new(MipsAir::Bls12381Fp(FpOpChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp.name(), bls12381_fp.cost());
        chips.push(bls12381_fp);

        let bls12381_fp2_addsub =
            Chip::new(MipsAir::Bls12381Fp2AddSub(Fp2AddSubAssignChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp2_addsub.name(), bls12381_fp2_addsub.cost());
        chips.push(bls12381_fp2_addsub);

        let bls12381_fp2_mul =
            Chip::new(MipsAir::Bls12381Fp2Mul(Fp2MulAssignChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp2_mul.name(), bls12381_fp2_mul.cost());
        chips.push(bls12381_fp2_mul);

        let bn254_fp = Chip::new(MipsAir::Bn254Fp(FpOpChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp.name(), bn254_fp.cost());
        chips.push(bn254_fp);

        let bn254_fp2_addsub =
            Chip::new(MipsAir::Bn254Fp2AddSub(Fp2AddSubAssignChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp2_addsub.name(), bn254_fp2_addsub.cost());
        chips.push(bn254_fp2_addsub);

        let bn254_fp2_mul =
            Chip::new(MipsAir::Bn254Fp2Mul(Fp2MulAssignChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp2_mul.name(), bn254_fp2_mul.cost());
        chips.push(bn254_fp2_mul);

        let bls12381_decompress =
            Chip::new(MipsAir::Bls12381Decompress(WeierstrassDecompressChip::<
                SwCurve<Bls12381Parameters>,
            >::with_lexicographic_rule()));
        costs.insert(bls12381_decompress.name(), bls12381_decompress.cost());
        chips.push(bls12381_decompress);

        let syscall_core = Chip::new(MipsAir::SyscallCore(SyscallChip::core()));
        costs.insert(syscall_core.name(), syscall_core.cost());
        chips.push(syscall_core);

        let syscall_precompile = Chip::new(MipsAir::SyscallPrecompile(SyscallChip::precompile()));
        costs.insert(syscall_precompile.name(), syscall_precompile.cost());
        chips.push(syscall_precompile);

        let div_rem = Chip::new(MipsAir::DivRem(DivRemChip::default()));
        costs.insert(div_rem.name(), div_rem.cost());
        chips.push(div_rem);

        let add = Chip::new(MipsAir::Add(AddChip::default()));
        costs.insert(add.name(), add.cost());
        chips.push(add);

        let addi = Chip::new(MipsAir::Addi(AddiChip));
        costs.insert(addi.name(), addi.cost());
        chips.push(addi);

        let add_noop = Chip::new(MipsAir::AddNoop(AddNoopChip::default()));
        costs.insert(add_noop.name(), add_noop.cost());
        chips.push(add_noop);

        let sub = Chip::new(MipsAir::Sub(SubChip::default()));
        costs.insert(sub.name(), sub.cost());
        chips.push(sub);

        let alu_x0 = Chip::new(MipsAir::AluX0(AluX0Chip));
        costs.insert(alu_x0.name(), alu_x0.cost());
        chips.push(alu_x0);

        let bitwise = Chip::new(MipsAir::Bitwise(BitwiseChip::default()));
        costs.insert(bitwise.name(), bitwise.cost());
        chips.push(bitwise);

        let mul = Chip::new(MipsAir::Mul(MulChip::default()));
        costs.insert(mul.name(), mul.cost());
        chips.push(mul);

        let shift_right = Chip::new(MipsAir::ShiftRight(ShiftRightChip::default()));
        costs.insert(shift_right.name(), shift_right.cost());
        chips.push(shift_right);

        let shift_left = Chip::new(MipsAir::ShiftLeft(ShiftLeft::default()));
        costs.insert(shift_left.name(), shift_left.cost());
        chips.push(shift_left);

        let lt = Chip::new(MipsAir::Lt(LtChip::default()));
        costs.insert(lt.name(), lt.cost());
        chips.push(lt);

        let slti = Chip::new(MipsAir::Slti(SltiChip));
        costs.insert(slti.name(), slti.cost());
        chips.push(slti);

        let clo_clz = Chip::new(MipsAir::CloClz(CloClzChip::default()));
        costs.insert(clo_clz.name(), clo_clz.cost());
        chips.push(clo_clz);

        let branch = Chip::new(MipsAir::Branch(BranchChip::default()));
        costs.insert(branch.name(), branch.cost());
        chips.push(branch);

        let jump = Chip::new(MipsAir::Jump(JumpChip::default()));
        costs.insert(jump.name(), jump.cost());
        chips.push(jump);

        let syscall_instrs = Chip::new(MipsAir::SyscallInstrs(SyscallInstrsChip::default()));
        costs.insert(syscall_instrs.name(), syscall_instrs.cost());
        chips.push(syscall_instrs);

        let memory_instructions =
            Chip::new(MipsAir::MemoryInstrs(MemoryInstructionsChip::default()));
        costs.insert(memory_instructions.name(), memory_instructions.cost());
        chips.push(memory_instructions);

        let load_word = Chip::new(MipsAir::LoadWord(LoadWordChip::default()));
        costs.insert(load_word.name(), load_word.cost());
        chips.push(load_word);

        let store_word = Chip::new(MipsAir::StoreWord(StoreWordChip::default()));
        costs.insert(store_word.name(), store_word.cost());
        chips.push(store_word);

        let misc_instrs = Chip::new(MipsAir::MiscInstrs(MiscInstrsChip::default()));
        costs.insert(misc_instrs.name(), misc_instrs.cost());
        chips.push(misc_instrs);

        let memory_global_init =
            Chip::new(MipsAir::MemoryGlobalInit(MemoryGlobalChip::new(MemoryChipType::Initialize)));
        costs.insert(memory_global_init.name(), memory_global_init.cost());
        chips.push(memory_global_init);

        let memory_global_finalize =
            Chip::new(MipsAir::MemoryGlobalFinal(MemoryGlobalChip::new(MemoryChipType::Finalize)));
        costs.insert(memory_global_finalize.name(), memory_global_finalize.cost());
        chips.push(memory_global_finalize);

        let memory_local = Chip::new(MipsAir::MemoryLocal(MemoryLocalChip::new()));
        costs.insert(memory_local.name(), memory_local.cost());
        chips.push(memory_local);

        let global = Chip::new(MipsAir::Global(GlobalChip));
        costs.insert(global.name(), global.cost());
        chips.push(global);

        let byte = Chip::new(MipsAir::ByteLookup(ByteChip::default()));
        costs.insert(byte.name(), byte.cost());
        chips.push(byte);

        let sys_linux = Chip::new(MipsAir::SysLinux(SysLinuxChip::default()));
        costs.insert(sys_linux.name(), sys_linux.cost());
        chips.push(sys_linux);

        let movcond_instrs = Chip::new(MipsAir::MovCond(MovCondChip::default()));
        costs.insert(movcond_instrs.name(), movcond_instrs.cost());
        chips.push(movcond_instrs);

        let state_bump = Chip::new(MipsAir::StateBump(StateBumpChip::new()));
        costs.insert(state_bump.name(), state_bump.cost());
        chips.push(state_bump);

        let memory_bump = Chip::new(MipsAir::MemoryBump(MemoryBumpChip::new()));
        costs.insert(memory_bump.name(), memory_bump.cost());
        chips.push(memory_bump);

        (chips, costs)
    }
}

impl<F: PrimeField32> fmt::Debug for MipsAir<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<F: PrimeField32> PartialEq for MipsAir<F> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl<F: PrimeField32> Eq for MipsAir<F> {}

impl<F: PrimeField32> core::hash::Hash for MipsAir<F> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
pub mod tests {
    use crate::programs::tests::other_memory_program;
    use crate::programs::tests::add_sub_x0_program;
    use crate::programs::tests::simple_program;
    use crate::programs::tests::slt_x0_program;
    use crate::programs::tests::{
        fibonacci_program, hello_world_program, max_memory_program, sha3_chain_program,
        simple_memory_program, ssz_withdrawals_program, unconstrained_program,
    };
    use crate::{
        // io::ZKMStdin,
        mips::MipsAir,
        // utils,
        utils::{run_test, setup_logger},
    };

    use hashbrown::HashMap;
    use itertools::Itertools;
    use p3_koala_bear::KoalaBear;
    use strum::IntoEnumIterator;

    use zkm_core_executor::{Instruction, MipsAirId, Opcode, Program};
    use zkm_hypercube::air::MachineAir;
    // use zkm_stark::{
    //     koala_bear_poseidon2::KoalaBearPoseidon2, CpuProver, StarkProvingKey, StarkVerifyingKey,
    //     ZKMCoreOpts,
    // };

    #[test]
    fn test_primitives_and_machine_air_names_match() {
        let chips = MipsAir::<KoalaBear>::chips();
        for (a, b) in chips.iter().zip_eq(MipsAirId::iter()) {
            assert_eq!(a.name(), b.to_string());
        }
    }

    #[test]
    fn core_air_cost_consistency() {
        let file = std::fs::File::open("../executor/src/artifacts/mips_costs.json").unwrap();
        let costs: HashMap<String, u64> = serde_json::from_reader(file).unwrap();
        // Compare with costs computed by machine
        let machine_costs = MipsAir::<KoalaBear>::costs();
        log::info!("{machine_costs:?}");
        assert_eq!(costs, machine_costs);
    }

    #[test]
    fn write_core_air_costs() {
        let costs = MipsAir::<KoalaBear>::costs();
        println!("{costs:?}");
        // write to file
        // Create directory if it doesn't exist
        let dir = std::path::Path::new("../executor/src/artifacts");
        if !dir.exists() {
            std::fs::create_dir_all(dir).unwrap();
        }
        let file = std::fs::File::create(dir.join("mips_costs.json")).unwrap();
        serde_json::to_writer_pretty(file, &costs).unwrap();
    }

    /// Loads a real guest ELF from disk, runs it with real stdin data, then checks that every
    /// chip's local-scope send/receive interactions balance across every record the run produces
    /// (the main execution record plus any deferred-memory-event records it splits into --
    /// mirrors `prove_with_context`'s public-values threading exactly, crates/core/machine/src/
    /// utils/prove.rs, since a real guest's global memory init/finalize events can split off into
    /// their own non-execution record(s) that need their own public values). Much cheaper than a
    /// full proving run: only execution, trace generation, and a direct interaction-count check,
    /// no FRI/PCS commitment. Returns `true` iff everything balanced, printing which record index
    /// broke first otherwise. Panics if `elf_path` doesn't exist -- build it first (real zkVM
    /// toolchain required; these ELFs aren't part of the repo's default `ZKM_SKIP_PROGRAM_BUILD`
    /// test path).
    fn debug_local_interactions_balance_for_elf(elf_path: &str, stdin_bufs: Vec<Vec<u8>>) -> bool {
        use p3_air::BaseAir;
        use slop_multilinear::{Mle, PaddedMle};
        use std::sync::Arc;
        use zkm_core_executor::{Executor, ExecutionRecord};
        use zkm_hypercube::{
            air::{LookupScope, PublicValues},
            lookup::{debug_interactions_with_all_chips, LookupKind},
            prover::Traces,
            record::MachineRecord,
        };
        use zkm_stark::ZKMCoreOpts;

        setup_logger();

        let elf_bytes = std::fs::read(elf_path)
            .unwrap_or_else(|e| panic!("failed to read {elf_path}: {e} -- build it first"));
        let program = Arc::new(Program::from(&elf_bytes).unwrap());
        let mut runtime = Executor::new(Program::clone(&program), ZKMCoreOpts::default());
        runtime.write_vecs(&stdin_bufs);
        runtime.run().unwrap();
        println!("{} execution shard(s)", runtime.records.len());

        // Mirrors `prove_with_context`'s public-values threading and shared `deferred`
        // accumulator exactly (crates/core/machine/src/utils/prove.rs), across every execution
        // shard `Executor::run()` produced -- not just one, since a real guest can genuinely
        // span many shards.
        let mut state = PublicValues::<u32, u32>::default().reset();
        let mut deferred = ExecutionRecord::new(program.clone());
        let mut all_records = Vec::new();
        let num_execution_shards = runtime.records.len();
        for (execution_shard, mut record) in runtime.records.into_iter().enumerate() {
            let execution_shard = execution_shard as u32 + 1;
            let done = execution_shard as usize == num_execution_shards;

            state.shard += 1;
            state.execution_shard = execution_shard;
            state.is_execution_shard = record.contains_cpu() as u32;
            if let Some(first_pc) = record.first_instruction_pc {
                state.start_pc = first_pc;
                state.next_pc = record.last_next_pc;
                let first_clk = record.first_instruction_clk.unwrap();
                state.initial_clk_high = (first_clk >> 24) as u32;
                state.initial_clk_low = (first_clk & 0xFFFFFF) as u32;
                // See `prove_with_context`'s identical computation (and
                // `ExecutionRecord::last_instruction_clk`'s doc comment) for why this must use
                // `last_instruction_clk`'s own high limb rather than `last_timestamp`'s.
                let last_clk_high = record.last_instruction_clk >> 24;
                state.last_clk_high = last_clk_high as u32;
                state.last_clk_low = (record.last_timestamp - (last_clk_high << 24)) as u32;
            }
            state.committed_value_digest = record.public_values.committed_value_digest;
            state.deferred_proofs_digest = record.public_values.deferred_proofs_digest;
            record.public_values = state;

            deferred.append(&mut record.defer());
            let mut records = vec![record];
            let mut split_records =
                deferred.split(done, records.last_mut(), ZKMCoreOpts::default().split_opts);

            if !done {
                state.execution_shard += 1;
            }
            for split_record in &mut split_records {
                state.shard += 1;
                state.is_execution_shard = 0;
                state.previous_init_addr_bits = split_record.public_values.previous_init_addr_bits;
                state.last_init_addr_bits = split_record.public_values.last_init_addr_bits;
                state.previous_finalize_addr_bits =
                    split_record.public_values.previous_finalize_addr_bits;
                state.last_finalize_addr_bits = split_record.public_values.last_finalize_addr_bits;
                state.start_pc = state.next_pc;
                state.initial_clk_high = state.last_clk_high;
                state.initial_clk_low = state.last_clk_low;
                split_record.public_values = state;
            }
            records.append(&mut split_records);
            all_records.append(&mut records);
        }
        println!("produced {} record(s)", all_records.len());

        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        machine.generate_dependencies(all_records.iter_mut(), None).unwrap();
        let chips = machine.chips().to_vec();
        let max_log_row_count = 22u32;

        let mut all_balanced = true;
        for (i, record) in all_records.into_iter().enumerate() {
            let mut preprocessed_named = std::collections::BTreeMap::new();
            let mut main_named = std::collections::BTreeMap::new();
            for chip in &chips {
                let name = MachineAir::<KoalaBear>::name(chip);
                let pre_mle = match chip.generate_preprocessed_trace(&record.program) {
                    Some(t) => {
                        PaddedMle::padded_with_zeros(Arc::new(Mle::from(t)), max_log_row_count)
                    }
                    None => PaddedMle::zeros(0, max_log_row_count),
                };
                preprocessed_named.insert(name.clone(), pre_mle);

                let main_mle = if chip.included(&record) {
                    let trace = chip.generate_trace(&record, &mut Default::default()).unwrap();
                    PaddedMle::padded_with_zeros(Arc::new(Mle::from(trace)), max_log_row_count)
                } else {
                    PaddedMle::zeros(chip.width(), max_log_row_count)
                };
                main_named.insert(name, main_mle);
            }
            let preprocessed_traces = Traces { named_traces: preprocessed_named };
            let traces = Traces { named_traces: main_named };
            let public_values = record.public_values::<KoalaBear>();

            let balanced = debug_interactions_with_all_chips(
                &chips,
                &preprocessed_traces,
                &traces,
                public_values,
                LookupKind::all_kinds(),
                LookupScope::Local,
            );
            all_balanced &= balanced;
            if !balanced {
                println!("record {i} is imbalanced -- stopping early instead of checking the rest");
                break;
            }
        }
        all_balanced
    }

    /// Loads the real `examples/fibonacci/guest` ELF and feeds it a real `n = 1000` via stdin,
    /// exactly like `examples/fibonacci/host` -- exercising the real `HINT_LEN`/`HINT_READ`/
    /// `COMMIT` syscall path with real data. Requires the guest to already be built (`cargo run
    /// --release` under `examples/fibonacci/host` with the zkVM toolchain sourced); `#[ignore]`d
    /// since this repo's default `ZKM_SKIP_PROGRAM_BUILD` test path never builds it.
    #[test]
    #[ignore = "needs examples/fibonacci/guest built via the real zkVM toolchain"]
    fn debug_real_fibonacci_guest_with_stdin_interactions_balance() {
        let mut stdin_buf = Vec::new();
        bincode::serialize_into(&mut stdin_buf, &1000u32).unwrap();
        assert!(
            debug_local_interactions_balance_for_elf(
                "../../../examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/fibonacci",
                vec![stdin_buf],
            ),
            "local-scope send/receive interactions don't balance"
        );
    }

    /// Loads the real `examples/tendermint/guest` ELF and feeds it the same two real, CBOR-
    /// encoded light-block fixtures `examples/tendermint/host` uses (dumped to disk via
    /// `examples/tendermint/host/bin/dump_stdin.rs`), exercising ed25519/secp256k1 signature
    /// verification and sha256 hashing with real data -- a much heavier, more varied precompile
    /// mix than the fibonacci guest above, and (at ~56M cycles) large enough to actually cross a
    /// `clk_high` boundary. Requires both the guest ELF and the two `.bin` stdin dumps to already
    /// exist; `#[ignore]`d for the same reason as the fibonacci variant.
    #[test]
    #[ignore = "needs examples/tendermint/guest built via the real zkVM toolchain, plus dumped stdin fixtures"]
    fn debug_real_tendermint_guest_with_stdin_interactions_balance() {
        let stdin_1 = std::fs::read("../../../examples/target/tendermint_stdin_1.bin")
            .expect("run `cargo run --release --bin dump_stdin` under examples/tendermint/host first");
        let stdin_2 = std::fs::read("../../../examples/target/tendermint_stdin_2.bin")
            .expect("run `cargo run --release --bin dump_stdin` under examples/tendermint/host first");
        assert!(
            debug_local_interactions_balance_for_elf(
                "../../../examples/target/elf-compilation/mipsel-zkm-zkvm-elf/release/tendermint",
                vec![stdin_1, stdin_2],
            ),
            "local-scope send/receive interactions don't balance"
        );
    }

    #[test]
    fn test_simple_prove() {
        setup_logger();
        let program = simple_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_add_sub_x0_prove() {
        setup_logger();
        let program = add_sub_x0_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_slt_x0_prove() {
        setup_logger();
        let program = slt_x0_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_beq_branching_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 1, false, true),
            Instruction::new(Opcode::BEQ, 29, 30, 100, false, true),
            Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_beq_not_branching_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 2, false, true),
            Instruction::new(Opcode::BEQ, 29, 30, 100, false, true),
            Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_bne_branching_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 2, false, true),
            Instruction::new(Opcode::BNE, 29, 30, 100, false, true),
            Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_bne_not_branching_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 0, false, true),
            Instruction::new(Opcode::BNE, 29, 30, 100, false, true),
            Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_rest_branch_prove() {
        setup_logger();
        let branch_ops = [Opcode::BLTZ, Opcode::BGEZ, Opcode::BLEZ, Opcode::BGTZ];
        let operands = [0, 1, 0xFFFF_FFFF];
        for branch_op in branch_ops.iter() {
            for operand in operands.iter() {
                let instructions = vec![
                    Instruction::new(Opcode::ADD, 29, 0, *operand, false, true),
                    Instruction::new(*branch_op, 29, 0, 100, true, true),
                    Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
                ];
                let program = Program::new(instructions, 0, 0);
                run_test(program).unwrap();
            }
        }
    }

    #[test]
    fn test_shift_prove() {
        setup_logger();
        let shift_ops = [Opcode::SRL, Opcode::ROR, Opcode::SRA, Opcode::SLL];
        let operands =
            [(1, 1), (1234, 5678), (0xffff, 0xffff - 1), (u32::MAX - 1, u32::MAX), (u32::MAX, 0)];
        for shift_op in shift_ops.iter() {
            for op in operands.iter() {
                let instructions = vec![
                    Instruction::new(Opcode::ADD, 29, 0, op.0, false, true),
                    Instruction::new(Opcode::ADD, 30, 0, op.1, false, true),
                    Instruction::new(*shift_op, 31, 29, 3, false, false),
                ];
                let program = Program::new(instructions, 0, 0);
                run_test(program).unwrap();
            }
        }
    }

    #[test]
    fn test_sub_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 8, false, true),
            Instruction::new(Opcode::SUB, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_add_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 8, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_add_overflow_prove() {
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xEFFF_FFFF, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 2, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_mul_mod_prove() {
        setup_logger();
        let mul_ops = [Opcode::MUL, Opcode::MOD, Opcode::MODU];
        let operands =
            [(1, 1), (1234, 5678), (8765, 4321), (0xffff, 0xffff - 1), (u32::MAX - 1, u32::MAX)];
        for mul_op in mul_ops.iter() {
            for operand in operands.iter() {
                let instructions = vec![
                    Instruction::new(Opcode::ADD, 29, 0, operand.0, false, true),
                    Instruction::new(Opcode::ADD, 30, 0, operand.1, false, true),
                    Instruction::new(*mul_op, 31, 30, 29, false, false),
                ];
                let program = Program::new(instructions, 0, 0);
                run_test(program).unwrap();
            }
        }
    }

    #[test]
    fn test_mult_div_prove() {
        setup_logger();
        let mul_ops = [Opcode::MULT, Opcode::MULTU, Opcode::DIV, Opcode::DIVU];
        let operands =
            [(1, 1), (1234, 5678), (8765, 4321), (0xffff, 0xffff - 1), (u32::MAX - 1, u32::MAX)];
        for mul_op in mul_ops.iter() {
            for operand in operands.iter() {
                let instructions = vec![
                    Instruction::new(Opcode::ADD, 29, 0, operand.0, false, true),
                    Instruction::new(Opcode::ADD, 30, 0, operand.1, false, true),
                    Instruction::new(*mul_op, 32, 30, 29, false, false),
                ];
                let program = Program::new(instructions, 0, 0);
                run_test(program).unwrap();
            }
        }
    }

    #[test]
    fn test_lt_prove() {
        setup_logger();
        let less_than = [Opcode::SLT, Opcode::SLTU];
        for lt_op in less_than.iter() {
            let instructions = vec![
                Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
                Instruction::new(Opcode::ADD, 30, 0, 8, false, true),
                Instruction::new(*lt_op, 31, 30, 29, false, false),
            ];
            let program = Program::new(instructions, 0, 0);
            run_test(program).unwrap();
        }
    }

    #[test]
    fn test_bitwise_prove() {
        setup_logger();
        let bitwise_opcodes = [Opcode::XOR, Opcode::OR, Opcode::AND];

        for bitwise_op in bitwise_opcodes.iter() {
            let instructions = vec![
                Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
                Instruction::new(Opcode::ADD, 30, 0, 8, false, true),
                Instruction::new(*bitwise_op, 31, 30, 29, false, false),
            ];
            let program = Program::new(instructions, 0, 0);
            run_test(program).unwrap();
        }
    }

    #[test]
    fn test_divrem_prove() {
        setup_logger();
        let div_rem_ops = [Opcode::DIV, Opcode::DIVU];
        let operands = [
            (1, 1),
            (123, 456 * 789),
            (123 * 456, 789),
            (0xffff * (0xffff - 1), 0xffff),
            (u32::MAX - 5, u32::MAX - 7),
            (5, i32::MIN.unsigned_abs()),
        ];
        for div_rem_op in div_rem_ops.iter() {
            for op in operands.iter() {
                let instructions = vec![
                    Instruction::new(Opcode::ADD, 29, 0, op.0, false, true),
                    Instruction::new(Opcode::ADD, 30, 0, op.1, false, true),
                    Instruction::new(*div_rem_op, 32, 29, 30, false, false),
                ];
                let program = Program::new(instructions, 0, 0);
                run_test(program).unwrap();
            }
        }
    }

    #[test]
    fn test_cloclz_prove() {
        setup_logger();
        let clz_clo_ops = [Opcode::CLZ, Opcode::CLO];
        let operands = [0u32, 0x0a0b0c0d, 0x1000, 0xff7fffff, 0x7fffffff, 0x80000000, 0xffffffff];

        for clo_clz_op in clz_clo_ops.iter() {
            for op in operands.iter() {
                let instructions = vec![
                    Instruction::new(Opcode::ADD, 29, 0, *op, false, true),
                    Instruction::new(*clo_clz_op, 30, 29, 0, false, true),
                ];
                let program = Program::new(instructions, 0, 0);
                run_test(program).unwrap();
            }
        }
    }

    #[test]
    fn test_j_prove() {
        //   j 100
        //
        // The j instruction performs an unconditional jump to a specified address.
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 11, 0, 100, false, true),
            Instruction::new(Opcode::Jumpi, 0, 100, 0, true, true),
            Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_jr_prove() {
        //   addi x11, x11, 100
        //   jr x11
        //
        // The jr instruction jumps to an address stored in a register.
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 11, 0, 100, false, true),
            Instruction::new(Opcode::Jump, 0, 11, 0, false, true),
            Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_jalr_prove() {
        //   addi x11, x11, 100
        //   jalr x11
        //
        // Similar to jal, but jumps to an address stored in a register.
        setup_logger();
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, 0, false, true),
            Instruction::new(Opcode::ADD, 11, 11, 100, false, true),
            Instruction::new(Opcode::Jump, 5, 11, 0, false, true),
            Instruction::new(Opcode::ADD, 0, 0, 0, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_sc_prove() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0x12348765, false, true),
            Instruction::new(Opcode::SW, 29, 0, 0x27654320, false, true),
            // LL and SC
            Instruction::new(Opcode::LL, 28, 0, 0x27654320, false, true),
            Instruction::new(Opcode::ADD, 28, 28, 1, false, true),
            Instruction::new(Opcode::SC, 28, 0, 0x27654320, false, true),
            Instruction::new(Opcode::LW, 29, 0, 0x27654320, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        run_test(program).unwrap();
    }

    #[test]
    fn test_hello_world_prove_simple() {
        setup_logger();
        let program = hello_world_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_fibonacci_prove_simple() {
        setup_logger();
        let program = fibonacci_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_max_memory_prove_simple() {
        setup_logger();
        let program = max_memory_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_sha3_chain_prove_simple() {
        setup_logger();
        let program = sha3_chain_program();
        run_test(program).unwrap();
    }

    #[test]
    #[ignore = "no zkm-hypercube shard prove/verify driver yet (old FRI-backed run_test/CpuProver removed)"]
    fn test_fibonacci_prove_checkpoints() {
        // setup_logger();
        //
        // let program = fibonacci_program();
        // let stdin = ZKMStdin::new();
        // let mut opts = ZKMCoreOpts::default();
        // opts.shard_size = 1024;
        // opts.shard_batch_size = 2;
        // prove::<_, CpuProver<_, _>>(program, &stdin, KoalaBearPoseidon2::new(), opts, None)
        //     .unwrap();
    }

    #[test]
    #[ignore = "no zkm-hypercube shard prove/verify driver yet (old FRI-backed run_test/CpuProver removed)"]
    fn test_fibonacci_prove_batch() {
        // setup_logger();
        // let program = fibonacci_program();
        // let stdin = ZKMStdin::new();
        // prove::<_, CpuProver<_, _>>(
        //     program,
        //     &stdin,
        //     KoalaBearPoseidon2::new(),
        //     ZKMCoreOpts::default(),
        //     None,
        // )
        // .unwrap();
    }

    #[test]
    fn test_simple_memory_program_prove() {
        setup_logger();
        let program = simple_memory_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_simple_memory_program_2_prove() {
        setup_logger();
        let program = other_memory_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_ssz_withdrawal() {
        setup_logger();
        let program = ssz_withdrawals_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_unconstrained() {
        setup_logger();
        let program = unconstrained_program();
        run_test(program).unwrap();
    }

    #[test]
    fn test_key_serde() {
        use std::sync::Arc;

        use serde::{Deserialize, Serialize};
        use zkm_hypercube::{
            config::{default_fri_config, ZkmGlobalContext},
            prover::{AirProver, ProverSemaphore, ZkmShardProver},
            ShardVerifier,
        };
        use zkm_stark::{CORE_LOG_STACKING_HEIGHT, CORE_MAX_LOG_ROW_COUNT};

        fn roundtrip<T: Serialize + for<'de> Deserialize<'de>>(value: &T) -> T {
            let bytes = bincode::serialize(value).unwrap();
            bincode::deserialize(&bytes).unwrap()
        }

        let program = ssz_withdrawals_program();
        let machine = MipsAir::<KoalaBear>::hypercube_machine();
        let max_log_row_count = CORE_MAX_LOG_ROW_COUNT;
        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            CORE_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        let shard_prover = ZkmShardProver::<MipsAir<KoalaBear>>::new(shard_verifier);

        let setup_rt = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (preprocessed, vk) =
            setup_rt.block_on(shard_prover.setup(Arc::new(program), ProverSemaphore::new(1)));
        let pk = preprocessed.pk;

        let deserialized_pk = roundtrip(pk.as_ref());
        assert_eq!(pk.vk, deserialized_pk.vk);
        assert_eq!(
            bincode::serialize(&pk.preprocessed_data).unwrap(),
            bincode::serialize(&deserialized_pk.preprocessed_data).unwrap(),
        );

        let deserialized_vk: zkm_hypercube::MachineVerifyingKey<ZkmGlobalContext> = roundtrip(&vk);
        assert_eq!(vk, deserialized_vk);
    }

    // -----------------------------------------------------------------------
    // Syscall soundness regression tests
    //
    // These tests exercise Go ELF programs that trigger linux syscalls
    // (mmap, clone, brk, fcntl, read, write, exit_group, nop) through the
    // Go runtime initialization. They validate that the bidirectional flag
    // constraints, bytewise heap updates, and other soundness fixes do not
    // reject honest traces.
    //
    // Covered issues:
    //   #1  is_sys_linux bidirectional
    //   #2  SysLinux flag routing bidirectional
    //   #3  is_mmap_a0_0 bidirectional
    //   #4  Reduced arg1/arg2 bound to packed half-words
    //   #5  mmap A3 output zeroed unconditionally
    //   #6  page_offset decomposition + range check + alignment
    //   #7  exit_group result zeroed
    //   #8  fnctl(a1==1) result constrained
    //   #9  is_a0_0/1/2 bidirectional
    //   #10 write: read value = prev_value
    //   #11 mmap: bytewise heap update via AddOperation
    //   #12 is_a1_1/3 bidirectional
    // -----------------------------------------------------------------------

    /// Exercises SYS_WRITE, exit_group, mmap, clone, brk, fcntl, and nop
    /// syscall paths through the Go hello_world runtime.
    #[test]
    fn test_syscall_soundness_hello_world() {
        setup_logger();
        let program = hello_world_program();
        run_test(program).unwrap();
    }

    /// Exercises the full Go runtime init: mmap2 with a0=0 (heap allocation),
    /// fcntl with a1=1 and a1=3, clone, brk, read, and exit_group.
    #[test]
    fn test_syscall_soundness_fibonacci() {
        setup_logger();
        let program = fibonacci_program();
        run_test(program).unwrap();
    }

    /// Every chip name referenced by `MipsAir::hypercube_machine`'s cluster construction must
    /// resolve against `Self::chips()` (`by_name[name]` panics on a typo or a renamed chip).
    #[test]
    fn hypercube_machine_builds_without_panicking() {
        let machine = MipsAir::<p3_koala_bear::KoalaBear>::hypercube_machine();
        assert!(!machine.chips().is_empty());
    }
}
