use core::borrow::Borrow;
use std::borrow::BorrowMut;

use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use zkm_core_machine::utils::next_power_of_two;
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::{BinomialExtension, MachineAir};

use crate::{
    air::Block, builder::ZKMRecursionAirBuilder, Address, ExecutionRecord, Instruction,
    PrefixSumChecksInstr,
};

/// A chip that evaluates the multilinear extension of an equality indicator between a bit-string
/// point and a random extension-field point, one coordinate per row, while simultaneously
/// reconstructing the integer value of (half of) the bit-string as a felt.
///
/// Used by the jagged PCS's "branching program" evaluation gadget to check a sumcheck's random
/// evaluation point against a pair of column row-count prefix sums.
#[derive(Default)]
pub struct PrefixSumChecksChip;

pub const NUM_PREFIX_SUM_CHECKS_COLS: usize = core::mem::size_of::<PrefixSumChecksCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct PrefixSumChecksCols<T: Copy> {
    pub x1: T,
    pub x2: Block<T>,
    pub acc: Block<T>,
    pub new_acc: Block<T>,
    pub felt_acc: T,
    pub felt_new_acc: T,
}

pub const NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS: usize =
    core::mem::size_of::<PrefixSumChecksPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct PrefixSumChecksPreprocessedCols<T: Copy> {
    pub x1_mem: Address<T>,
    pub x2_mem: Address<T>,
    /// The address holding this row's *incoming* accumulator (`addrs.one`/`addrs.zero` for the
    /// first row of an invocation, else the previous row's `next_acc_addr`/`felt_next_acc_addr`).
    pub acc_addr: Address<T>,
    pub next_acc_addr: Address<T>,
    pub next_acc_mult: T,
    pub felt_acc_addr: Address<T>,
    pub felt_next_acc_addr: Address<T>,
    pub felt_next_acc_mult: T,
    pub is_real: T,
}

impl<F: Field> BaseAir<F> for PrefixSumChecksChip {
    fn width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for PrefixSumChecksChip {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    type Error = crate::RecursionChipError;

    fn name(&self) -> String {
        "PrefixSumChecks".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let fixed_num_rows = program.fixed_num_rows(self);
        Some(match fixed_num_rows {
            Some(num_rows) => num_rows,
            None => next_power_of_two(
                instrs_len,
                None,
                <PrefixSumChecksChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::PrefixSumChecks(x) => Some(x.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>();

        // Each instruction contributes one row per `x1` coordinate (a full accumulation chain),
        // not one row per instruction.
        let total_rows: usize = instrs.iter().map(|instr| instr.addrs.x1.len()).sum();
        let padded_nb_rows = self.preprocessed_num_rows(program, total_rows).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS];

        let mut row_offset = 0;
        for instr in &instrs {
            let PrefixSumChecksInstr { addrs, acc_mults, field_acc_mults } = instr;
            let len = addrs.x1.len();
            let start = row_offset * NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS;
            let end = (row_offset + len) * NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS;
            values[start..end]
                .chunks_mut(NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS)
                .enumerate()
                .for_each(|(i, row)| {
                    let cols: &mut PrefixSumChecksPreprocessedCols<F> = row.borrow_mut();
                    if i == 0 {
                        cols.acc_addr = addrs.one;
                        cols.felt_acc_addr = addrs.zero;
                    } else {
                        cols.acc_addr = addrs.accs[i - 1];
                        cols.felt_acc_addr = addrs.field_accs[i - 1];
                    }
                    cols.x1_mem = addrs.x1[i];
                    cols.x2_mem = addrs.x2[i];
                    cols.next_acc_addr = addrs.accs[i];
                    cols.next_acc_mult = acc_mults[i];
                    cols.felt_next_acc_addr = addrs.field_accs[i];
                    cols.felt_next_acc_mult = field_acc_mults[i];
                    cols.is_real = F::ONE;
                });
            row_offset += len;
        }

        Some(RowMajorMatrix::new(values, NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS))
    }

    fn generate_dependencies(
        &self,
        _: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<(), Self::Error> {
        // This is a no-op.
        Ok(())
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.prefix_sum_checks_events;
        Some(match input.fixed_num_rows(self) {
            Some(num_rows) => num_rows,
            None => next_power_of_two(
                events.len(),
                None,
                <PrefixSumChecksChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = &input.prefix_sum_checks_events;
        let padded_nb_rows = self.num_rows(input).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_PREFIX_SUM_CHECKS_COLS];

        let populate_len = events.len() * NUM_PREFIX_SUM_CHECKS_COLS;
        values[..populate_len].par_chunks_mut(NUM_PREFIX_SUM_CHECKS_COLS).zip_eq(events).for_each(
            |(row, event)| {
                let cols: &mut PrefixSumChecksCols<F> = row.borrow_mut();
                cols.x1 = event.x1;
                cols.x2 = event.x2;
                cols.acc = event.acc;
                cols.new_acc = event.new_acc;
                cols.felt_acc = event.field_acc;
                cols.felt_new_acc = event.new_field_acc;
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_PREFIX_SUM_CHECKS_COLS))
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for PrefixSumChecksChip
where
    AB: ZKMRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &PrefixSumChecksCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &PrefixSumChecksPreprocessedCols<AB::Var> = (*prep_local).borrow();

        let x2 = local.x2.as_extension::<AB>();
        let prod = BinomialExtension::from_base(local.x1.into()) * x2.clone();
        let one: BinomialExtension<AB::Expr> = BinomialExtension::from_base(AB::Expr::one());
        let two = AB::Expr::from_canonical_u32(2);

        let sum_x_y = BinomialExtension::from_base(local.x1.into()) + x2;

        // `is_real` and `x1` are boolean.
        builder.assert_bool(prep_local.is_real);
        builder.assert_bool(local.x1);

        // Constrain the memory reads for x1/x2.
        builder.receive_single(prep_local.x1_mem, local.x1, prep_local.is_real);
        builder.receive_block(prep_local.x2_mem, local.x2, prep_local.is_real);

        // Constrain the memory read for the incoming accumulators.
        builder.receive_block(prep_local.acc_addr, local.acc, prep_local.is_real);
        builder.receive_single(prep_local.felt_acc_addr, local.felt_acc, prep_local.is_real);

        // new_acc = acc * (1 - x1 - x2 + 2*x1*x2), i.e. acc * eq(x1, x2).
        builder.assert_ext_eq(
            local.new_acc.as_extension::<AB>(),
            local.acc.as_extension::<AB>() * (one - sum_x_y + prod.clone() + prod),
        );
        // felt_new_acc = x1 + 2*felt_acc (big-endian bit-to-integer accumulation).
        builder.assert_eq(local.felt_new_acc, local.x1 + two * local.felt_acc);

        // Constrain the memory write for the outgoing accumulators.
        builder.send_block(prep_local.next_acc_addr, local.new_acc, prep_local.next_acc_mult);
        builder.send_single(
            prep_local.felt_next_acc_addr,
            local.felt_new_acc,
            prep_local.felt_next_acc_mult,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::machine::tests::run_recursion_test_machine;
    use p3_field::{extension::BinomialExtensionField, FieldAlgebra, FieldExtensionAlgebra};
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{
        machine::RecursionAir, runtime::instruction as instr, MemAccessKind, PrefixSumChecksEvent,
        RecursionProgram,
    };

    #[test]
    fn generate_trace() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;

        let x1 = F::ONE;
        let x2: EF = EF::ONE;
        let acc: EF = EF::ONE;
        let product = EF::from_base(x1) * x2;
        let new_acc = acc * (EF::ONE - EF::from_base(x1) - x2 + product + product);

        let shard = ExecutionRecord {
            prefix_sum_checks_events: vec![PrefixSumChecksEvent {
                x1,
                x2: Block::from(x2.as_base_slice()),
                zero: F::ZERO,
                one: Block::from(EF::ONE.as_base_slice()),
                acc: Block::from(acc.as_base_slice()),
                new_acc: Block::from(new_acc.as_base_slice()),
                field_acc: F::ZERO,
                new_field_acc: F::ONE,
            }],
            ..Default::default()
        };
        let chip = PrefixSumChecksChip;
        let trace: RowMajorMatrix<F> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    fn prove_prefix_sum_checks() {
        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;
        type A = RecursionAir<F, 3>;

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut addr = 0u32;

        let len = 8usize;
        let x1_bits: Vec<F> = (0..len).map(|_| F::from_bool(rng.gen_bool(0.5))).collect();
        let x2_vals: Vec<EF> = (0..len)
            .map(|_| {
                EF::from_base_slice(&core::array::from_fn::<F, 4, _>(|_| {
                    rng.sample(rand::distributions::Standard)
                }))
            })
            .collect();

        let x1_addrs: Vec<u32> = (0..len as u32).map(|i| addr + i).collect();
        addr += len as u32;
        let x2_addrs: Vec<u32> = (0..len as u32).map(|i| addr + i).collect();
        addr += len as u32;
        let acc_addrs: Vec<u32> = (0..len as u32).map(|i| addr + i).collect();
        addr += len as u32;
        let field_acc_addrs: Vec<u32> = (0..len as u32).map(|i| addr + i).collect();
        addr += len as u32;
        let zero_addr = addr;
        addr += 1;
        let one_addr = addr;

        // Write the operands. `zero`/`one` are the seed accumulators; `accs`/`field_accs` get
        // written by the `prefix_sum_checks` instruction itself below.
        let mut instructions = vec![
            instr::mem_single(MemAccessKind::Write, 1, zero_addr, F::ZERO),
            instr::mem_ext(MemAccessKind::Write, 1, one_addr, EF::ONE),
        ];
        for i in 0..len {
            instructions.push(instr::mem_single(MemAccessKind::Write, 1, x1_addrs[i], x1_bits[i]));
            instructions.push(instr::mem_ext(MemAccessKind::Write, 1, x2_addrs[i], x2_vals[i]));
        }

        // Every intermediate accumulator (index 0..len-2) is consumed exactly once by this
        // chip's own next row; the final accumulator (index len-1) is consumed exactly once by
        // the read instructions appended below, so every index gets multiplicity 1.
        instructions.push(instr::prefix_sum_checks(
            zero_addr,
            one_addr,
            x1_addrs,
            x2_addrs,
            acc_addrs.clone(),
            field_acc_addrs.clone(),
            vec![1; len],
            vec![1; len],
        ));

        // Compute the expected final accumulators (mirrors the runtime's own arithmetic) and
        // read them back to check the chip actually produced the right values, not just
        // internally-consistent ones.
        let mut acc = EF::ONE;
        let mut field_acc = F::ZERO;
        for i in 0..len {
            let product = EF::from_base(x1_bits[i]) * x2_vals[i];
            acc *= EF::ONE - EF::from_base(x1_bits[i]) - x2_vals[i] + product + product;
            field_acc = x1_bits[i] + field_acc * F::from_canonical_u32(2);
        }
        instructions.push(instr::mem_ext(MemAccessKind::Read, 1, acc_addrs[len - 1], acc));
        instructions
            .push(instr::mem_single(MemAccessKind::Read, 1, field_acc_addrs[len - 1], field_acc));

        let program = RecursionProgram { instructions, ..Default::default() };
        let program = std::sync::Arc::new(program);
        let mut runtime = crate::runtime::Runtime::<
            F,
            EF,
            p3_koala_bear::Poseidon2InternalLayerKoalaBear<16>,
        >::new(program.clone(), zkm_stark::inner_perm());
        runtime.run().unwrap();

        run_recursion_test_machine::<3>(
            A::machine_wide_with_all_chips(),
            (*program).clone(),
            runtime.record,
        );
    }
}
