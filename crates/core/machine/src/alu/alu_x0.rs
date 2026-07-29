use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord, MemoryAccessPosition},
    ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState, InstructionCols},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::RegisterAccessCols,
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `AluX0Chip`.
pub const NUM_ALU_X0_COLS: usize = size_of::<AluX0Cols<u8>>();

/// A chip that handles every real, retired `RTypeReader`/`AluTypeReader`-family instruction whose
/// destination register (`op_a`) is register 0 (`$zero`) -- today, register-form `add $zero,
/// ...`/`sub $zero, ...`, XOR/OR/AND/NOR (register- or immediate-form), SRL/SRA/ROR
/// (register- or immediate-shift-amount-form), MEQ/MNE/WSBH, MOD/MODU, CLZ/CLO, and INS with
/// `op_a==0`. Since `$zero` is hardwired to always read as 0, the computed result is never
/// observable by anything downstream (any later read of `$zero` yields 0 regardless), so this
/// chip doesn't compute it at all: it only verifies the program lookup (opcode/operands match the
/// ROM) and the register-consistency accesses (`op_b`/`op_c` reads, `op_a`'s write modeled as a
/// no-op since `$zero`'s value never changes). MOD/MODU never write a second register (unlike
/// DIV/DIVU, which also always write HI and never reach this chip in the first place -- see
/// `DivRemChip`'s doc comment), so their `op_a==0` case fits this "discard everything" model
/// cleanly; same for CLZ/CLO and INS/EXT/SEXT (no second register at all, and INS/EXT/SEXT's
/// computed result is entirely discarded/unobservable when `op_a==0`, so their shift
/// chain/sign-extension need not be verified for that row).
///
/// This is a growing catch-all: as more chips migrate to `RTypeReader`/`AluTypeReader`/
/// `ITypeReaderNonZero`, their own `op_a==0` case gets added here too (extend the one-hot opcode
/// selectors below and add a routing arm in `emit_alu_event`/`emit_misc_event`), rather than
/// spinning up a new per-opcode chip each time. Currently supports: `ADD`, `SUB`, `SLT`, `SLTU`,
/// `XOR`, `OR`, `AND`, `NOR`, `SRL`, `SRA`, `ROR`, `SLL`, `MUL`, `MEQ`, `MNE`, `WSBH`, `MOD`,
/// `MODU`, `CLZ`, `CLO`, `INS`, `EXT`, `SEXT`.
///
/// The opcode is encoded as one witnessed boolean selector per supported opcode (`is_add`,
/// `is_sub`, ...) rather than a single witnessed value plus a root-set validity check
/// (`(opcode-ADD)*(opcode-SUB)*... == 0`) -- the latter was this chip's original design, but its
/// constraint degree grows by one with every opcode added, and quickly exceeds
/// `zkm_hypercube::chip::MAX_CONSTRAINT_DEGREE` (3): already at the cap with just two opcodes
/// (`is_real * (opcode-ADD) * (opcode-SUB)`, degree 3). One-hot selectors cost one extra *column*
/// per opcode instead, with the reconstructed `opcode` a degree-1 linear combination
/// (`Σ is_X * OPCODE_X`) -- constant degree regardless of how many opcodes this chip grows to
/// support, which is the whole point of designing it as a growing catch-all.
///
/// Every row here is a real, retired instruction -- no `is_retired` split needed, since no chip
/// ever sends a synthetic dependency row shaped like this (an `op_a==0` row has no meaningful
/// "result" for another chip to depend on).
#[derive(Default)]
pub struct AluX0Chip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct AluX0Cols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// One-hot opcode selectors -- exactly one is set on a real row, all zero on padding (see
    /// `AluX0Chip`'s doc comment for why this replaces a single witnessed `opcode` value).
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_add: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_sub: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_slt: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_sltu: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_xor: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_or: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_and: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_nor: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_srl: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_sra: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_ror: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_sll: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_mul: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_meq: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_mne: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_wsbh: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_mod: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_modu: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_clz: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_clo: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_ins: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_ext: T,
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_sext: T,

    /// Register 0's write access. Modeled as a no-op (the value never changes, since `$zero` is
    /// hardwired to always read as 0) -- see `RTypeReader`'s doc comment for why this chip exists.
    pub op_a_access: RegisterAccessCols<T>,

    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,

    /// Either the register index of `op_c` (byte 0, when `imm_c` is unset) or its immediate
    /// value (when `imm_c` is set) -- only the bitwise/shift opcodes, WSBH/CLZ/CLO (whose `op_c`
    /// is always the immediate 0), INS/EXT (whose `op_c` is always the immediate `msb(d) << 5 |
    /// lsb`), and SEXT (whose `op_c` is always the immediate 0 or 1, selecting SEB/SEH) can have
    /// `imm_c` set (MIPS has no ADDI/SUBI/SLTI-with-op_a==0 shape reaching this chip; see
    /// `emit_alu_event`/`emit_misc_event`'s routing).
    pub op_c: Word<T>,
    pub op_c_access: RegisterAccessCols<T>,
    pub imm_c: T,
}

impl<F: PrimeField32> MachineAir<F> for AluX0Chip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "AluX0".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.alu_x0_events.len(),
            None,
            <AluX0Chip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        AluX0Cols::<u8>::picus_info()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.alu_x0_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <AluX0Chip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ALU_X0_COLS);

        values.chunks_mut(chunk_size * NUM_ALU_X0_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ALU_X0_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AluX0Cols<F> = row.borrow_mut();

                    if idx < input.alu_x0_events.len() {
                        let event = &input.alu_x0_events[idx];
                        self.event_to_row(event, cols, &mut Vec::new(), &input.program);
                    }
                });
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_ALU_X0_COLS))
    }

    fn generate_dependencies(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        let chunk_size = std::cmp::max(input.alu_x0_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .alu_x0_events
            .chunks(chunk_size)
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::ZERO; NUM_ALU_X0_COLS];
                    let cols: &mut AluX0Cols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu, &input.program);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
        Ok(())
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.alu_x0_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AluX0Chip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &AluEvent,
        cols: &mut AluX0Cols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);
        cols.is_add = F::from_bool(event.opcode == Opcode::ADD);
        cols.is_sub = F::from_bool(event.opcode == Opcode::SUB);
        cols.is_slt = F::from_bool(event.opcode == Opcode::SLT);
        cols.is_sltu = F::from_bool(event.opcode == Opcode::SLTU);
        cols.is_xor = F::from_bool(event.opcode == Opcode::XOR);
        cols.is_or = F::from_bool(event.opcode == Opcode::OR);
        cols.is_and = F::from_bool(event.opcode == Opcode::AND);
        cols.is_nor = F::from_bool(event.opcode == Opcode::NOR);
        cols.is_srl = F::from_bool(event.opcode == Opcode::SRL);
        cols.is_sra = F::from_bool(event.opcode == Opcode::SRA);
        cols.is_ror = F::from_bool(event.opcode == Opcode::ROR);
        cols.is_sll = F::from_bool(event.opcode == Opcode::SLL);
        cols.is_mul = F::from_bool(event.opcode == Opcode::MUL);
        cols.is_meq = F::from_bool(event.opcode == Opcode::MEQ);
        cols.is_mne = F::from_bool(event.opcode == Opcode::MNE);
        cols.is_wsbh = F::from_bool(event.opcode == Opcode::WSBH);
        cols.is_mod = F::from_bool(event.opcode == Opcode::MOD);
        cols.is_modu = F::from_bool(event.opcode == Opcode::MODU);
        cols.is_clz = F::from_bool(event.opcode == Opcode::CLZ);
        cols.is_clo = F::from_bool(event.opcode == Opcode::CLO);
        cols.is_ins = F::from_bool(event.opcode == Opcode::INS);
        cols.is_ext = F::from_bool(event.opcode == Opcode::EXT);
        cols.is_sext = F::from_bool(event.opcode == Opcode::SEXT);

        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);

        if let Some(record) = event.a_record {
            cols.op_a_access.populate(record, blu);
            // Unlike a plain read (where the current value is the same as `prev_value`, already
            // range-checked by `populate` above), a write's current value is a *different* word
            // that `eval_register_access_write_value`'s defense-in-depth check also range-checks
            // -- so it needs its own byte-lookup event here too (always the zero word here, but
            // still needs to exist as an event to balance the interaction).
            blu.add_u8_range_checks(&record.current_record().value.to_le_bytes());
        }

        cols.op_b = F::from_canonical_u32(instruction.op_b);
        if let Some(record) = event.b_record {
            cols.op_b_access.populate(record, blu);
        }

        cols.imm_c = F::from_bool(instruction.imm_c);
        cols.op_c = Word::from(instruction.op_c);
        if instruction.imm_c {
            cols.op_c_access.prev_value = cols.op_c;
        } else if let Some(record) = event.c_record {
            cols.op_c_access.populate(record, blu);
        }
    }
}

impl<F> BaseAir<F> for AluX0Chip {
    fn width(&self) -> usize {
        NUM_ALU_X0_COLS
    }
}

impl<AB> Air<AB> for AluX0Chip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AluX0Cols<AB::Var> = (*local).borrow();

        // Each opcode selector is its own boolean; their sum derives `is_real` (real vs padding)
        // -- extend with one more selector + term per opcode as more chips migrate to
        // `RTypeReader`, rather than a single opcode-value + root-set check (whose degree would
        // grow without bound -- see this chip's doc comment).
        builder.assert_bool(local.is_add);
        builder.assert_bool(local.is_sub);
        builder.assert_bool(local.is_slt);
        builder.assert_bool(local.is_sltu);
        builder.assert_bool(local.is_xor);
        builder.assert_bool(local.is_or);
        builder.assert_bool(local.is_and);
        builder.assert_bool(local.is_nor);
        builder.assert_bool(local.is_srl);
        builder.assert_bool(local.is_sra);
        builder.assert_bool(local.is_ror);
        builder.assert_bool(local.is_sll);
        builder.assert_bool(local.is_mul);
        builder.assert_bool(local.is_meq);
        builder.assert_bool(local.is_mne);
        builder.assert_bool(local.is_wsbh);
        builder.assert_bool(local.is_mod);
        builder.assert_bool(local.is_modu);
        builder.assert_bool(local.is_clz);
        builder.assert_bool(local.is_clo);
        builder.assert_bool(local.is_ins);
        builder.assert_bool(local.is_ext);
        builder.assert_bool(local.is_sext);
        let is_real = local.is_add
            + local.is_sub
            + local.is_slt
            + local.is_sltu
            + local.is_xor
            + local.is_or
            + local.is_and
            + local.is_nor
            + local.is_srl
            + local.is_sra
            + local.is_ror
            + local.is_sll
            + local.is_mul
            + local.is_meq
            + local.is_mne
            + local.is_wsbh
            + local.is_mod
            + local.is_modu
            + local.is_clz
            + local.is_clo
            + local.is_ins
            + local.is_ext
            + local.is_sext;
        builder.assert_bool(is_real.clone());

        // Only the bitwise and shift opcodes can ever have `imm_c` set (see `AluX0Cols::op_c`'s
        // doc comment); for the others, the program lookup below fails to find a matching ROM
        // entry if a malicious prover set it anyway (ADD/SUB/SLT/SLTU never decode with `imm_c`
        // true in a shape that reaches this chip), so no separate constraint is needed to rule it
        // out.
        builder.when(is_real.clone()).assert_bool(local.imm_c);
        // Defense-in-depth: force `imm_c` to 0 outside real rows too, so `is_real - imm_c` below
        // (an affine combination, required since lookup multiplicities must have degree <= 1)
        // can only ever land on 0 or 1.
        builder.when_not(is_real.clone()).assert_zero(local.imm_c);

        // If `op_c` is an immediate, assert its value is copied into `op_c_access.prev_value` (so
        // the interaction skip below is sound).
        builder
            .when(is_real.clone() * Into::<AB::Expr>::into(local.imm_c))
            .assert_word_eq(local.op_c_access.prev_value, local.op_c);

        // The written value is always the constant zero -- `$zero` never actually changes, so
        // writing to it is representationally a no-op; see `RTypeReader`'s doc comment.
        let zero_word =
            Word([AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]);

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here: `op_a=0`/`op_a_0=1`/`imm_b=0` are
        // compile-time constants (this shape always has a zero destination and never an
        // immediate `op_b`); `opcode` is a degree-1 linear combination of the one-hot selectors
        // above; `op_b` is zero-extended from its register-index column, `op_c` is already a
        // full word (register-index-in-byte0 or immediate, per `imm_c`).
        let opcode = local.is_add * Opcode::ADD.as_field::<AB::F>()
            + local.is_sub * Opcode::SUB.as_field::<AB::F>()
            + local.is_slt * Opcode::SLT.as_field::<AB::F>()
            + local.is_sltu * Opcode::SLTU.as_field::<AB::F>()
            + local.is_xor * Opcode::XOR.as_field::<AB::F>()
            + local.is_or * Opcode::OR.as_field::<AB::F>()
            + local.is_and * Opcode::AND.as_field::<AB::F>()
            + local.is_nor * Opcode::NOR.as_field::<AB::F>()
            + local.is_srl * Opcode::SRL.as_field::<AB::F>()
            + local.is_sra * Opcode::SRA.as_field::<AB::F>()
            + local.is_ror * Opcode::ROR.as_field::<AB::F>()
            + local.is_sll * Opcode::SLL.as_field::<AB::F>()
            + local.is_mul * Opcode::MUL.as_field::<AB::F>()
            + local.is_meq * Opcode::MEQ.as_field::<AB::F>()
            + local.is_mne * Opcode::MNE.as_field::<AB::F>()
            + local.is_wsbh * Opcode::WSBH.as_field::<AB::F>()
            + local.is_mod * Opcode::MOD.as_field::<AB::F>()
            + local.is_modu * Opcode::MODU.as_field::<AB::F>()
            + local.is_clz * Opcode::CLZ.as_field::<AB::F>()
            + local.is_clo * Opcode::CLO.as_field::<AB::F>()
            + local.is_ins * Opcode::INS.as_field::<AB::F>()
            + local.is_ext * Opcode::EXT.as_field::<AB::F>()
            + local.is_sext * Opcode::SEXT.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: AB::Expr::zero(),
            op_b: Word::extend_var::<AB>(local.op_b),
            op_c: local.op_c.map(Into::into),
            op_a_0: AB::Expr::one(),
            imm_b: AB::Expr::zero(),
            imm_c: local.imm_c.into(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        // Register positions must be read/written in the order C, B, A (see
        // `MemoryAccessPosition`'s doc comment). `op_c`'s access is skipped (zero multiplicity)
        // when it's an immediate -- `is_real - imm_c` (not `is_real * (1 - imm_c)`) to keep this
        // an affine lookup multiplicity (see the `imm_c`-boolean defense-in-depth check above).
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
            local.op_c[0].into(),
            &local.op_c_access,
            is_real.clone() - Into::<AB::Expr>::into(local.imm_c),
        );
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            local.op_b.into(),
            &local.op_b_access,
            is_real.clone(),
        );
        builder.eval_register_access_write_value(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            AB::Expr::zero(),
            zero_word,
            &local.op_a_access,
            is_real.clone(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.clone());

        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk_high,
            clk_low,
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::AluEvent, ExecutionRecord, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::AluX0Chip;

    #[test]
    fn generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.program = Program::new(
            vec![zkm_core_executor::Instruction::new(Opcode::ADD, 0, 29, 30, false, false)],
            0,
            0,
        )
        .into();
        shard.alu_x0_events = vec![AluEvent::new(0, Opcode::ADD, 0, 5, 6)];
        let chip = AluX0Chip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
