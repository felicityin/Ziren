use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Program,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::{MachineAir, ZKMAirBuilder};

use crate::{utils::next_power_of_two, CoreChipError};

pub(crate) const NUM_STATE_BUMP_COLS: usize = size_of::<StateBumpCols<u8>>();

/// Proves a `clk_high` transition whenever an instruction's clk advance is about to cross a
/// `1 << 24` boundary (see [`BumpClkHighEvent`]'s doc comment). Receives the exact
/// `(clk_high, clk_low, pc, next_pc)` state some other row sent -- `clk_low` here can be as
/// large as `1 << 24` (not itself range-checked to 24 bits, unlike an ordinary opcode chip's
/// *incoming* `clk_low`), since it's only used to source this row's outgoing `clk_high + 1`
/// transition, not reassembled into a checked value here -- and sends on the corrected
/// `(clk_high + 1, 0, pc, next_pc)` pair, with `clk_high + 1` freshly decomposed into a 16+8-bit
/// limb pair and range-checked. This is what bounds the global clk to 48 bits total and is the
/// only place `clk_high` is ever range-checked: an ordinary opcode chip's row never checks its
/// own `clk_high`, trusting it inductively via the `LookupKind::State` chain back to either a
/// shard-boundary public value or one of this chip's own outputs.
#[derive(AlignedBorrow, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct StateBumpCols<T: Copy> {
    /// The incoming `clk_high`.
    pub clk_high: T,
    /// The incoming `clk_low` (may be as large as `1 << 24`, see the chip's doc comment).
    pub clk_low: T,
    /// The 16-bit limb of `clk_high + 1`.
    pub next_clk_high_16bit_limb: T,
    /// The 8-bit limb of `clk_high + 1`.
    pub next_clk_high_8bit_limb: T,
    /// This row's own pc (matches the predecessor's outgoing `next_pc`).
    pub pc: T,
    /// This row's own incoming `next_pc` (matches the predecessor's outgoing `next_next_pc`).
    pub next_pc: T,
    /// Whether this row is a real bump event.
    pub is_real: T,
}

#[derive(Default)]
pub struct StateBumpChip;

impl StateBumpChip {
    pub const fn new() -> Self {
        Self
    }
}

impl<F> BaseAir<F> for StateBumpChip {
    fn width(&self) -> usize {
        NUM_STATE_BUMP_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for StateBumpChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "StateBump".to_string()
    }

    fn generate_dependencies(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<(), Self::Error> {
        let mut blu: Vec<ByteLookupEvent> = Vec::new();
        for event in &input.bump_clk_high_events {
            let next_clk = event.prev_clk + event.increment;
            let next_clk_high = next_clk >> 24;
            let next_clk_high_16bit_limb = (next_clk_high & 0xffff) as u16;
            let next_clk_high_8bit_limb = ((next_clk_high >> 16) & 0xff) as u8;
            blu.push(ByteLookupEvent::new(ByteOpcode::U16Range, next_clk_high_16bit_limb, 0, 0, 0));
            blu.push(ByteLookupEvent::new(ByteOpcode::U8Range, 0, 0, 0, next_clk_high_8bit_limb));
        }
        output.add_byte_lookup_events(blu);
        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.bump_clk_high_events.len();
        Some(next_power_of_two(nb_rows, None, <Self as MachineAir<F>>::name(self).as_str()))
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = &input.bump_clk_high_events;
        let padded_nb_rows =
            <StateBumpChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_STATE_BUMP_COLS];

        values[..events.len() * NUM_STATE_BUMP_COLS]
            .chunks_mut(NUM_STATE_BUMP_COLS)
            .zip(events.iter())
            .for_each(|(row, event)| {
                let cols: &mut StateBumpCols<F> = row.borrow_mut();

                let clk_high = event.prev_clk >> 24;
                let clk_low = event.prev_clk & 0xffffff;
                let next_clk = event.prev_clk + event.increment;
                let next_clk_high = next_clk >> 24;
                let next_clk_high_16bit_limb = (next_clk_high & 0xffff) as u16;
                let next_clk_high_8bit_limb = ((next_clk_high >> 16) & 0xff) as u8;

                cols.clk_high = F::from_canonical_u64(clk_high);
                cols.clk_low = F::from_canonical_u64(clk_low);
                cols.next_clk_high_16bit_limb = F::from_canonical_u16(next_clk_high_16bit_limb);
                cols.next_clk_high_8bit_limb = F::from_canonical_u8(next_clk_high_8bit_limb);
                cols.pc = F::from_canonical_u32(event.pc);
                cols.next_pc = F::from_canonical_u32(event.next_pc);
                cols.is_real = F::ONE;
            });

        Ok(RowMajorMatrix::new(values, NUM_STATE_BUMP_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.bump_clk_high_events.is_empty()
    }
}

impl<AB> Air<AB> for StateBumpChip
where
    AB: ZKMAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &StateBumpCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        // Receive the incoming (possibly non-canonical) state.
        builder.receive_state(local.clk_high, local.clk_low, local.pc, local.next_pc, local.is_real);

        // `next_clk_high` is freshly decomposed and range-checked here -- this, not any
        // per-row check in an ordinary opcode chip, is what bounds `clk_high` to 24 bits (see
        // this chip's doc comment).
        let next_clk_high = local.next_clk_high_16bit_limb
            + local.next_clk_high_8bit_limb * AB::Expr::from_canonical_u32(1 << 16);
        builder.when(local.is_real).assert_eq(
            next_clk_high.clone(),
            local.clk_high.into() + AB::Expr::one(),
        );

        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U16Range as u8),
            local.next_clk_high_16bit_limb,
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real,
        );
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.next_clk_high_8bit_limb,
            local.is_real,
        );

        // Send the corrected, canonical state on: `clk_low` resets to 0, `pc`/`next_pc` pass
        // through unchanged (this row doesn't correspond to an executed instruction, it only
        // re-bases the clk).
        builder.send_state(
            next_clk_high,
            AB::Expr::zero(),
            local.pc,
            local.next_pc,
            local.is_real,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_koala_bear::KoalaBear;
    use slop_multilinear::{Mle, PaddedMle};
    use std::sync::Arc;
    use zkm_core_executor::events::BumpClkHighEvent;
    use zkm_hypercube::{
        air::{LookupScope, PublicValues},
        lookup::{debug_interactions_with_all_chips, LookupKind},
        prover::Traces,
        record::MachineRecord,
        Chip,
    };

    /// Directly exercises a `clk_high` crossing (the one code path a full program run can't
    /// reach without ~16M cycles): builds a synthetic shard containing nothing but a single
    /// `BumpClkHighEvent`, anchored at both ends to `PublicValues`' own `initial_clk_high`/
    /// `initial_clk_low`/`last_clk_high`/`last_clk_low`/`start_pc`/`next_pc` boundary values
    /// (mirroring how `ExecutionRecord::eval_public_values` closes the `LookupKind::State` chain
    /// for a real shard), and checks every chip's send/receive interactions balance.
    #[test]
    fn state_bump_local_interactions_balance() {
        let mut record = ExecutionRecord::default();

        let prev_clk: u64 = (1 << 24) - 4;
        let increment: u64 = 4;
        let pc = 100u32;
        let next_pc = pc + 4;

        record.bump_clk_high_events.push(BumpClkHighEvent { prev_clk, increment, pc, next_pc });

        let mut public_values = PublicValues::<u32, u32>::default();
        public_values.is_execution_shard = 1;
        public_values.start_pc = pc;
        public_values.next_pc = pc;
        public_values.initial_clk_high = (prev_clk >> 24) as u32;
        public_values.initial_clk_low = (prev_clk & 0xffffff) as u32;
        let next_clk = prev_clk + increment;
        public_values.last_clk_high = (next_clk >> 24) as u32;
        public_values.last_clk_low = (next_clk & 0xffffff) as u32;
        record.public_values = public_values;

        let program = Program::default();
        let machine = crate::mips::MipsAir::<KoalaBear>::hypercube_machine();
        machine.generate_dependencies(std::iter::once(&mut record), None).unwrap();

        let chip_names = ["Program", "Byte", "StateBump"];
        let chips: Vec<Chip<KoalaBear, crate::mips::MipsAir<KoalaBear>>> = machine
            .chips()
            .iter()
            .filter(|c| chip_names.contains(&MachineAir::<KoalaBear>::name(*c).as_str()))
            .cloned()
            .collect();
        assert_eq!(chips.len(), chip_names.len(), "missing a chip by name");

        let max_log_row_count = 20u32;
        let mut preprocessed_named = std::collections::BTreeMap::new();
        let mut main_named = std::collections::BTreeMap::new();
        for chip in &chips {
            let name = MachineAir::<KoalaBear>::name(chip);
            let pre_mle = match chip.generate_preprocessed_trace(&program) {
                Some(t) => PaddedMle::padded_with_zeros(Arc::new(Mle::from(t)), max_log_row_count),
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

        assert!(
            debug_interactions_with_all_chips(
                &chips,
                &preprocessed_traces,
                &traces,
                public_values,
                LookupKind::all_kinds(),
                LookupScope::Local,
            ),
            "local-scope send/receive interactions don't balance"
        );
    }
}
