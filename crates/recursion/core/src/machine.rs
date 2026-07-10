use std::fmt;
use std::ops::{Add, AddAssign};

use hashbrown::HashMap;
use p3_field::{extension::BinomiallyExtendable, PrimeField32};
use zkm_hypercube::{
    air::{LookupScope, MachineAir, PicusInfo},
    shape::OrderedShape,
    Chip, Machine, MachineShape,
};

use crate::{
    chips::{
        alu_base::{BaseAluChip, NUM_BASE_ALU_ENTRIES_PER_ROW},
        alu_ext::{ExtAluChip, NUM_EXT_ALU_ENTRIES_PER_ROW},
        batch_fri::BatchFRIChip,
        exp_reverse_bits::ExpReverseBitsLenChip,
        fri_fold::FriFoldChip,
        mem::{
            constant::NUM_CONST_MEM_ENTRIES_PER_ROW, variable::NUM_VAR_MEM_ENTRIES_PER_ROW,
            MemoryConstChip, MemoryVarChip,
        },
        poseidon2_skinny::Poseidon2SkinnyChip,
        poseidon2_wide::Poseidon2WideChip,
        public_values::{PublicValuesChip, PUB_VALUES_LOG_HEIGHT},
        select::SelectChip,
    },
    instruction::{HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr},
    shape::RecursionShape,
    ExpReverseBitsInstr, Instruction, RecursionProgram, D,
};

#[derive(zkm_derive::MachineAir)]
#[zkm_core_path = "zkm_core_machine"]
#[execution_record_path = "crate::ExecutionRecord<F>"]
#[program_path = "crate::RecursionProgram<F>"]
#[builder_path = "crate::builder::ZKMRecursionAirBuilder<F = F>"]
#[error_path = "crate::RecursionChipError"]
#[eval_trait_bound = "AB::Var: 'static"]
pub enum RecursionAir<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> {
    MemoryConst(MemoryConstChip<F>),
    MemoryVar(MemoryVarChip<F>),
    BaseAlu(BaseAluChip),
    ExtAlu(ExtAluChip),
    Poseidon2Skinny(Poseidon2SkinnyChip<DEGREE>),
    Poseidon2Wide(Poseidon2WideChip<DEGREE>),
    Select(SelectChip),
    FriFold(FriFoldChip<DEGREE>),
    BatchFRI(BatchFRIChip<DEGREE>),
    ExpReverseBitsLen(ExpReverseBitsLenChip<DEGREE>),
    PublicValues(PublicValuesChip),
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> fmt::Debug
    for RecursionAir<F, DEGREE>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RecursionAirEventCount {
    pub mem_const_events: usize,
    pub mem_var_events: usize,
    pub base_alu_events: usize,
    pub ext_alu_events: usize,
    pub poseidon2_wide_events: usize,
    pub fri_fold_events: usize,
    pub batch_fri_events: usize,
    pub select_events: usize,
    pub exp_reverse_bits_len_events: usize,
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> RecursionAir<F, DEGREE> {
    /// Get a machine with all chips, except the dummy chip.
    pub fn machine_wide_with_all_chips() -> Machine<F, Self>
    where
        F: slop_algebra::Field,
    {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Wide(Poseidon2WideChip::<DEGREE>),
            RecursionAir::FriFold(FriFoldChip::<DEGREE>::default()),
            RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        let shape = MachineShape::all(&chips);
        Machine::new(chips, crate::air::RECURSIVE_PROOF_NUM_PV_ELTS, shape)
    }

    /// Get a machine with all chips, except the dummy chip.
    pub fn machine_skinny_with_all_chips() -> Machine<F, Self>
    where
        F: slop_algebra::Field,
    {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Skinny(Poseidon2SkinnyChip::<DEGREE>::default()),
            RecursionAir::FriFold(FriFoldChip::<DEGREE>::default()),
            RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        let shape = MachineShape::all(&chips);
        Machine::new(chips, crate::air::RECURSIVE_PROOF_NUM_PV_ELTS, shape)
    }

    /// A machine with dyunamic chip sizes that includes the wide variant of the Poseidon2 chip.
    pub fn compress_machine() -> Machine<F, Self>
    where
        F: slop_algebra::Field,
    {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Wide(Poseidon2WideChip::<DEGREE>),
            RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        let shape = MachineShape::all(&chips);
        Machine::new(chips, crate::air::RECURSIVE_PROOF_NUM_PV_ELTS, shape)
    }

    pub fn shrink_machine() -> Machine<F, Self>
    where
        F: slop_algebra::Field,
    {
        Self::compress_machine()
    }

    /// A machine with dynamic chip sizes that includes the skinny variant of the Poseidon2 chip.
    ///
    /// This machine assumes that the `shrink` stage has a fixed shape, so there is no need to
    /// fix the trace sizes.
    pub fn wrap_machine() -> Machine<F, Self>
    where
        F: slop_algebra::Field,
    {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Skinny(Poseidon2SkinnyChip::<DEGREE>::default()),
            RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        let shape = MachineShape::all(&chips);
        Machine::new(chips, crate::air::RECURSIVE_PROOF_NUM_PV_ELTS, shape)
    }

    pub fn shrink_shape() -> RecursionShape {
        let shape = HashMap::from(
            [
                (Self::MemoryVar(MemoryVarChip::default()), 18),
                (Self::Select(SelectChip), 18),
                (Self::MemoryConst(MemoryConstChip::default()), 17),
                (Self::BatchFRI(BatchFRIChip::<DEGREE>), 17),
                (Self::BaseAlu(BaseAluChip), 17),
                (Self::ExtAlu(ExtAluChip), 15),
                (Self::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>), 17),
                (Self::Poseidon2Wide(Poseidon2WideChip::<DEGREE>), 16),
                (Self::PublicValues(PublicValuesChip), PUB_VALUES_LOG_HEIGHT),
            ]
            .map(|(chip, log_height)| (chip.name(), log_height)),
        );
        RecursionShape { inner: shape }
    }

    pub fn heights(program: &RecursionProgram<F>) -> Vec<(String, usize)> {
        let heights = program
            .instructions
            .iter()
            .fold(RecursionAirEventCount::default(), |heights, instruction| heights + instruction);

        [
            (
                Self::MemoryConst(MemoryConstChip::default()),
                heights.mem_const_events.div_ceil(NUM_CONST_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::MemoryVar(MemoryVarChip::default()),
                heights.mem_var_events.div_ceil(NUM_VAR_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::BaseAlu(BaseAluChip),
                heights.base_alu_events.div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW),
            ),
            (
                Self::ExtAlu(ExtAluChip),
                heights.ext_alu_events.div_ceil(NUM_EXT_ALU_ENTRIES_PER_ROW),
            ),
            (Self::Poseidon2Wide(Poseidon2WideChip::<DEGREE>), heights.poseidon2_wide_events),
            (Self::BatchFRI(BatchFRIChip::<DEGREE>), heights.batch_fri_events),
            (Self::Select(SelectChip), heights.select_events),
            (
                Self::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
                heights.exp_reverse_bits_len_events,
            ),
            (Self::PublicValues(PublicValuesChip), PUB_VALUES_LOG_HEIGHT),
        ]
        .map(|(chip, log_height)| (chip.name(), log_height))
        .to_vec()
    }
}

impl<F> AddAssign<&Instruction<F>> for RecursionAirEventCount {
    #[inline]
    fn add_assign(&mut self, rhs: &Instruction<F>) {
        match rhs {
            Instruction::BaseAlu(_) => self.base_alu_events += 1,
            Instruction::ExtAlu(_) => self.ext_alu_events += 1,
            Instruction::Mem(_) => self.mem_const_events += 1,
            Instruction::Poseidon2(_) => self.poseidon2_wide_events += 1,
            Instruction::Select(_) => self.select_events += 1,
            Instruction::ExpReverseBitsLen(ExpReverseBitsInstr { addrs, .. }) => {
                self.exp_reverse_bits_len_events += addrs.exp.len()
            }
            Instruction::Hint(HintInstr { output_addrs_mults })
            | Instruction::HintBits(HintBitsInstr {
                output_addrs_mults,
                input_addr: _, // No receive lookup for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::HintExt2Felts(HintExt2FeltsInstr {
                output_addrs_mults,
                input_addr: _, // No receive lookup for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::FriFold(_) => self.fri_fold_events += 1,
            Instruction::BatchFRI(instr) => {
                self.batch_fri_events += instr.base_vec_addrs.p_at_x.len()
            }
            Instruction::HintAddCurve(HintAddCurveInstr {
                output_x_addrs_mults,
                output_y_addrs_mults,
                ..
            }) => {
                self.mem_var_events += output_x_addrs_mults.len();
                self.mem_var_events += output_y_addrs_mults.len();
            }
            Instruction::CommitPublicValues(_) => {}
            Instruction::Print(_) => {}
        }
    }
}

impl<F> Add<&Instruction<F>> for RecursionAirEventCount {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Instruction<F>) -> Self::Output {
        self += rhs;
        self
    }
}

impl From<RecursionShape> for OrderedShape {
    fn from(value: RecursionShape) -> Self {
        value.inner.into_iter().collect()
    }
}

#[cfg(test)]
pub mod tests {

    use std::{iter::once, sync::Arc};

    use crate::machine::RecursionAir;
    use p3_field::{
        extension::{BinomialExtensionField, HasFrobenius},
        Field, FieldAlgebra, FieldExtensionAlgebra,
    };
    use p3_koala_bear::Poseidon2InternalLayerKoalaBear;
    use rand::prelude::*;
    use slop_challenger::IopCtx;
    use zkm_hypercube::{
        config::{default_fri_config, ZkmGlobalContext},
        prover::{AirProver, ProverSemaphore, ZkmShardProver},
        Machine, ShardVerifier,
    };
    use zkm_stark::{koala_bear_poseidon2::KoalaBearPoseidon2, StarkGenericConfig};

    use crate::{
        runtime::{
            instruction as instr, BaseAluOpcode, ExtAluOpcode, Instruction, RecursionProgram,
            Runtime,
        },
        MemAccessKind, D,
    };

    type SC = KoalaBearPoseidon2;
    type F = <SC as StarkGenericConfig>::Val;
    type EF = <SC as StarkGenericConfig>::Challenge;
    type A = RecursionAir<F, 3>;

    /// The log2 of the number of rows each stacked-PCS column is grouped into. Mirrors
    /// `zkm_core_machine::utils::prove::ZKM_LOG_STACKING_HEIGHT` (which is crate-private to
    /// `zkm-core-machine`), kept in sync by convention.
    const RECURSION_LOG_STACKING_HEIGHT: u32 = 4;

    /// Sets up, proves, and verifies a single recursion shard for `program`/`record` against
    /// `machine`. Mirrors `zkm_core_machine::utils::prove::run_test_core`, simplified since
    /// recursion programs run as a single unsharded `ExecutionRecord` (no checkpointing).
    pub(crate) fn run_recursion_test_machine<const DEGREE: usize>(
        machine: Machine<F, RecursionAir<F, DEGREE>>,
        program: RecursionProgram<F>,
        record: crate::ExecutionRecord<F>,
    ) {
        let max_log_row_count = zkm_stark::ZKMCoreOpts::recursion().shard_size.ilog2() as usize;
        let program = Arc::new(program);

        let shard_prover = ZkmShardProver::<RecursionAir<F, DEGREE>>::new(
            ShardVerifier::from_basefold_parameters(
                default_fri_config(),
                RECURSION_LOG_STACKING_HEIGHT,
                max_log_row_count,
                machine.clone(),
            ),
        );

        let setup_rt =
            tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
        let (vk, proof, _permit) = setup_rt.block_on(shard_prover.setup_and_prove_shard(
            program,
            record,
            None,
            ProverSemaphore::new(1),
        ));

        let shard_verifier = ShardVerifier::from_basefold_parameters(
            default_fri_config(),
            RECURSION_LOG_STACKING_HEIGHT,
            max_log_row_count,
            machine,
        );
        let mut challenger = ZkmGlobalContext::default_challenger();
        vk.observe_into(&mut challenger);
        if let Err(e) = shard_verifier.verify_shard(&vk, &proof, &mut challenger) {
            panic!("Verification failed: {e:?}");
        }
    }

    /// Runs the given program on the machine that uses the wide Poseidon2 chip.
    ///
    /// TODO(zkm-hypercube): also run `B::machine_skinny_with_all_chips()` (DEGREE=9). Two
    /// distinct, stacked problems block this, both constraint-degree (not row-adjacency, which
    /// is already fixed below) issues:
    /// 1. Every DEGREE-parameterized recursion chip's `Air::eval` (FriFold, BatchFRI,
    ///    ExpReverseBitsLen, Poseidon2Wide, Poseidon2Skinny) has a "dummy constraints to
    ///    normalize to DEGREE" step (`(0..DEGREE).map(...).product()`) that intentionally forces
    ///    the AIR's polynomial degree up to DEGREE. At DEGREE=9 this alone exceeds
    ///    `zkm_hypercube::chip::MAX_CONSTRAINT_DEGREE` (3), independent of any chip's own logic
    ///    -- confirmed live via `chips::poseidon2_wide::tests::test_poseidon2`'s DEGREE=9 half
    ///    (also dropped, for the same reason).
    /// 2. `Poseidon2SkinnyChip` additionally has its own, chip-specific degree issue: its
    ///    13-round internal sbox chain, once gated by `is_internal_row`, exceeds the same cap on
    ///    its own merits.
    /// SP1's own recursion machine (architecturally ahead of this port) doesn't patch either: it
    /// deletes the row-per-round "skinny" design entirely and replaces it with row-local
    /// `Poseidon2SBoxChip`/`Poseidon2LinearLayerChip`/`ConvertChip` (one sbox/linear-layer op per
    /// row, degree <=3 by construction with no DEGREE-normalization trick needed at all, chained
    /// across rounds via ordinary virtual-memory addresses emitted by the compiler -- see
    /// `sp1/crates/recursion/machine/src/chips/poseidon2_helper/`). Porting that requires new
    /// instruction types and `recursion/compiler` changes (阶段3.3 scope), not just this crate.
    pub fn run_recursion_test_machines(program: RecursionProgram<F>) {
        let program = Arc::new(program);
        let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
            program.clone(),
            SC::new().perm,
        );
        runtime.run().unwrap();

        // Run with the poseidon2 wide chip.
        run_recursion_test_machine::<3>(
            A::machine_wide_with_all_chips(),
            (*program).clone(),
            runtime.record,
        );
    }

    fn test_instructions(instructions: Vec<Instruction<F>>) {
        let program = RecursionProgram { instructions, ..Default::default() };
        run_recursion_test_machines(program);
    }

    #[test]
    pub fn fibonacci() {
        let n = 10;

        let instructions = once(instr::mem(MemAccessKind::Write, 1, 0, 0))
            .chain(once(instr::mem(MemAccessKind::Write, 2, 1, 1)))
            .chain((2..=n).map(|i| instr::base_alu(BaseAluOpcode::AddF, 2, i, i - 2, i - 1)))
            .chain(once(instr::mem(MemAccessKind::Read, 1, n - 1, 34)))
            .chain(once(instr::mem(MemAccessKind::Read, 2, n, 55)))
            .collect::<Vec<_>>();

        test_instructions(instructions);
    }

    #[test]
    #[should_panic]
    pub fn div_nonzero_by_zero() {
        let instructions = vec![
            instr::mem(MemAccessKind::Write, 1, 0, 0),
            instr::mem(MemAccessKind::Write, 1, 1, 1),
            instr::base_alu(BaseAluOpcode::DivF, 1, 2, 1, 0),
            instr::mem(MemAccessKind::Read, 1, 2, 1),
        ];

        test_instructions(instructions);
    }

    #[test]
    pub fn div_zero_by_zero() {
        let instructions = vec![
            instr::mem(MemAccessKind::Write, 1, 0, 0),
            instr::mem(MemAccessKind::Write, 1, 1, 0),
            instr::base_alu(BaseAluOpcode::DivF, 1, 2, 1, 0),
            instr::mem(MemAccessKind::Read, 1, 2, 1),
        ];

        test_instructions(instructions);
    }

    #[test]
    pub fn field_norm() {
        let mut instructions = Vec::new();

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut addr = 0;
        for _ in 0..100 {
            let inner: [F; 4] = std::iter::repeat_with(|| {
                core::array::from_fn(|_| rng.sample(rand::distributions::Standard))
            })
            .find(|xs| !xs.iter().all(F::is_zero))
            .unwrap();
            let x = BinomialExtensionField::<F, D>::from_base_slice(&inner);
            let gal = x.galois_group();

            let mut acc = BinomialExtensionField::ONE;

            instructions.push(instr::mem_ext(MemAccessKind::Write, 1, addr, acc));
            for conj in gal {
                instructions.push(instr::mem_ext(MemAccessKind::Write, 1, addr + 1, conj));
                instructions.push(instr::ext_alu(ExtAluOpcode::MulE, 1, addr + 2, addr, addr + 1));

                addr += 2;
                acc *= conj;
            }
            let base_cmp: F = acc.as_base_slice()[0];
            instructions.push(instr::mem_single(MemAccessKind::Read, 1, addr, base_cmp));
            addr += 1;
        }

        test_instructions(instructions);
    }
}
