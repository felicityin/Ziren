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
    events::{ByteLookupEvent, ByteRecord, MemoryAccessPosition, MiscEvent},
    ByteOpcode, ExecutionRecord, Opcode, Program,
};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState, InstructionCols},
    air::ZKMCoreAirBuilder,
    memory::{RegisterAccessCols, RegisterWriteAccessCols},
    utils::{next_power_of_two, zeroed_f_vec},
    CoreChipError,
};

/// The number of main trace columns for `SextChip`.
pub const NUM_SEXT_COLS: usize = size_of::<SextCols<u8>>();

/// A chip that implements the MIPS sign-extend instructions SEB/SEH (both decoded as
/// `Opcode::SEXT`, distinguished by the encoded `c` immediate: 0 for SEB, 1 for SEH -- which is
/// therefore not stored as its own column here, but derived directly as `is_seh`, see
/// `SextCols::is_seh`'s doc comment).
///
/// Every row is a real, retired instruction: nothing sends a synthetic dependency row into
/// `sext_events`. `op_a` may be any register (including register 0 -- routed to `AluX0Chip`
/// instead, see its doc comment, since the sign-extended result is then unobservable); `op_b` is
/// always a register. SEXT is a fresh write of `op_a` (not read-modify-write), but its written
/// value is a per-byte mux between `op_b`'s bytes and a sign-extension byte (depending on
/// `is_seb`/`is_seh`) -- degree 2, which can't be sent directly as a lookup value the way an
/// `RTypeReader`-style chip does (see `RegisterWriteAccessCols`'s doc comment), so `op_a` uses
/// that instead, with each byte separately asserted against its witnessed `value`.
#[derive(Default)]
pub struct SextChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct SextCols<T: Copy> {
    /// The current shard and clk.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The register index of `op_a` (written; may be any register, including register 0 -- see
    /// this chip's doc comment).
    pub op_a: T,
    /// `op_a`'s access. A fresh write whose witnessed `value` is masked/muxed depending on
    /// `is_seb`/`is_seh`, so it needs `RegisterWriteAccessCols` rather than a directly-fed lookup
    /// value.
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,

    /// The most significant bit of the most significant byte.
    pub most_sig_bit: T,

    /// The most significant byte.
    pub sig_byte: T,

    /// Flag indicating whether the opcode is SEB. `op_c` (the instruction's own encoded
    /// immediate, always exactly 0 or 1) is therefore never stored as its own column: it equals
    /// `is_seh` directly.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_seb: T,
    /// Flag indicating whether the opcode is SEH.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_seh: T,
}

impl<F: PrimeField32> MachineAir<F> for SextChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "Sext".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        SextCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.sext_events.len(),
            None,
            <SextChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let chunk_size = std::cmp::max(input.sext_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <SextChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SEXT_COLS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_SEXT_COLS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_SEXT_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SextCols<F> = row.borrow_mut();

                    if idx < input.sext_events.len() {
                        let event = &input.sext_events[idx];
                        self.event_to_row(event, cols, &mut blu, &input.program);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        Ok(RowMajorMatrix::new(values, NUM_SEXT_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.sext_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl SextChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MiscEvent,
        cols: &mut SextCols<F>,
        blu: &mut impl ByteRecord,
        program: &Program,
    ) {
        cols.pc = F::from_canonical_u32(event.pc);
        cols.next_pc = F::from_canonical_u32(event.next_pc);

        // Every `sext_events` row is a real, retired instruction -- nothing ever produces a
        // synthetic dependency row here.
        cols.state.populate(blu, event.clk);

        let instruction = program.fetch(event.pc);

        cols.op_a = F::from_canonical_u8(instruction.op_a);
        if let Some(record) = event.a_record {
            cols.op_a_access.populate(record, blu);
        }

        cols.op_b = F::from_canonical_u32(instruction.op_b);
        if let Some(record) = event.b_record {
            cols.op_b_access.populate(record, blu);
        }

        let (sig_bit, sig_byte) = if event.c > 0 {
            cols.is_seh = F::ONE;
            ((event.b as u16) >> 15, (event.b >> 8 & 0xff) as u8)
        } else {
            cols.is_seb = F::ONE;
            (((event.b as u8) >> 7) as u16, event.b as u8)
        };
        cols.most_sig_bit = F::from_canonical_u16(sig_bit);
        cols.sig_byte = F::from_canonical_u8(sig_byte);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::MSB,
            a1: sig_bit,
            a2: 0,
            b: sig_byte,
            c: 0,
        });
    }
}

impl<F> BaseAir<F> for SextChip {
    fn width(&self) -> usize {
        NUM_SEXT_COLS
    }
}

impl<AB> Air<AB> for SextChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SextCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_seb);
        builder.assert_bool(local.is_seh);
        let is_real = local.is_seb + local.is_seh;
        builder.assert_bool(is_real.clone());

        let op_b_val = local.op_b_access.prev_value;
        let written_value: Word<AB::Expr> = local.op_a_access.value.map(Into::into);

        // ---- Program lookup, state chain, register access. ----
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // `op_c`'s value equals `is_seh` directly (see `SextCols::is_seh`'s doc comment): SEB
        // always decodes with `op_c=0`, SEH always with `op_c=1`. `imm_c` (whether `op_c` is an
        // immediate at all, as opposed to its value) is a separate, always-true compile-time
        // constant -- SEXT always decodes with `imm_c=true` for both SEB and SEH.
        let op_c_lsb: AB::Expr = local.is_seh.into();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SEXT.as_field::<AB::F>().into(),
            op_a: local.op_a.into(),
            op_b: Word::extend_var::<AB>(local.op_b),
            op_c: Word([op_c_lsb, AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()]),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::one(),
        };

        // most_sig_bit is bit 7 of sig_byte.
        builder.send_byte(
            ByteOpcode::MSB.as_field::<AB::F>(),
            local.most_sig_bit,
            local.sig_byte,
            AB::Expr::zero(),
            is_real.clone(),
        );

        // For seb, sig_byte is byte 0 of op_b. For seh, sig_byte is byte 1 of op_b.
        builder.when(local.is_seb).assert_eq(op_b_val[0], local.sig_byte);
        builder.when(local.is_seh).assert_eq(op_b_val[1], local.sig_byte);

        // Constraints for result value: for both seb and seh, bytes lower than sig_byte (contain)
        // equal op_b, bytes upper than sig_byte equal sign byte (0xff when sig_bit is 1,
        // otherwise 0).
        let sign_byte = AB::Expr::from_canonical_u8(0xFF) * local.most_sig_bit;

        builder.when(is_real.clone()).assert_eq(written_value[0].clone(), op_b_val[0].into());
        builder.when(local.is_seb).assert_eq(written_value[1].clone(), sign_byte.clone());
        builder.when(local.is_seh).assert_eq(written_value[1].clone(), op_b_val[1].into());
        builder.when(is_real.clone()).assert_eq(written_value[2].clone(), sign_byte.clone());
        builder.when(is_real.clone()).assert_eq(written_value[3].clone(), sign_byte);

        builder.send_program(local.pc, instruction, is_real.clone());

        // Register positions must be read/written in the order B, A (see
        // `MemoryAccessPosition`'s doc comment).
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            local.op_b.into(),
            &local.op_b_access,
            is_real.clone(),
        );
        builder.eval_register_access_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            local.op_a.into(),
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
    use zkm_core_executor::{events::MiscEvent, ExecutionRecord, Instruction, Opcode, Program};
    use zkm_hypercube::air::MachineAir;

    use super::SextChip;

    #[test]
    fn generate_trace() {
        let program = Program {
            instructions: vec![Instruction {
                opcode: Opcode::SEXT,
                op_a: 5,
                op_b: 8,
                op_c: 0,
                imm_b: false,
                imm_c: true,
                raw: None,
            }],
            pc_start: 0,
            pc_base: 0,
            next_pc: 4,
            image: Default::default(),
        };
        let mut shard = ExecutionRecord { program: program.into(), ..Default::default() };
        shard.sext_events = vec![MiscEvent::new(
            0,
            0,
            4,
            Opcode::SEXT,
            0xFFFFFF80,
            0x80,
            0,
            0,
            Default::default(),
        )];
        let chip = SextChip;
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
