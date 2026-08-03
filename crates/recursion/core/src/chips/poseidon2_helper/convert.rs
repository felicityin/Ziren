use core::borrow::Borrow;
use std::borrow::BorrowMut;
use std::iter::zip;

use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

use zkm_core_machine::utils::next_power_of_two;
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::MachineAir;

use crate::{builder::ZKMRecursionAirBuilder, *};

pub const NUM_CONVERT_ENTRIES_PER_ROW: usize = 1;

/// A chip converting between one extension-sized block (address 0) and `D` separate base-field
/// cells (addresses 1..=D), row-local (degree <= 3). Bridges the block-addressed
/// `Poseidon2SBoxChip`/`Poseidon2LinearLayerChip` wiring with ordinary felt-addressed memory.
///
/// Handles both directions (`ext2felt`/`felt2ext`) with the same unconditional
/// `receive`-then-`send` shape: the direction is encoded purely by the sign of the
/// preprocessed multiplicities (see `generate_preprocessed_trace`), not by a per-row selector.
#[derive(Default)]
pub struct ConvertChip;

pub const NUM_CONVERT_COLS: usize = core::mem::size_of::<ConvertCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct ConvertCols<F: Copy> {
    pub values: [ConvertValueCols<F>; NUM_CONVERT_ENTRIES_PER_ROW],
}
const NUM_CONVERT_VALUE_COLS: usize = core::mem::size_of::<ConvertValueCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct ConvertValueCols<F: Copy> {
    pub input: Block<F>,
}

pub const NUM_CONVERT_PREPROCESSED_COLS: usize =
    core::mem::size_of::<ConvertPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct ConvertPreprocessedCols<F: Copy> {
    pub accesses: [ConvertAccessCols<F>; NUM_CONVERT_ENTRIES_PER_ROW],
}

pub const NUM_CONVERT_ACCESS_COLS: usize = core::mem::size_of::<ConvertAccessCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct ConvertAccessCols<F: Copy> {
    pub addrs: [Address<F>; 5],
    pub mults: [F; 5],
}

impl<F: Field> BaseAir<F> for ConvertChip {
    fn width(&self) -> usize {
        NUM_CONVERT_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for ConvertChip {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    type Error = crate::RecursionChipError;

    fn name(&self) -> String {
        "ExtFeltConvert".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_CONVERT_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let nb_rows = instrs_len.div_ceil(NUM_CONVERT_ENTRIES_PER_ROW);
        let fixed_num_rows = program.fixed_num_rows(self);
        Some(match fixed_num_rows {
            Some(num_rows) => num_rows,
            None => next_power_of_two(
                nb_rows,
                None,
                <ConvertChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::ExtFelt(x) => Some(x),
                _ => None,
            })
            .collect::<Vec<_>>();

        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_CONVERT_PREPROCESSED_COLS];

        let populate_len = instrs.len() * NUM_CONVERT_ACCESS_COLS;
        values[..populate_len].par_chunks_mut(NUM_CONVERT_ACCESS_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let ExtFeltInstr { addrs, mults, ext2felt } = instr;
                let access: &mut ConvertAccessCols<_> = row.borrow_mut();
                access.addrs = addrs.to_owned();
                if *ext2felt {
                    access.mults[0] = F::ONE;
                    access.mults[1] = mults[1];
                    access.mults[2] = mults[2];
                    access.mults[3] = mults[3];
                    access.mults[4] = mults[4];
                } else {
                    access.mults[0] = -mults[0];
                    access.mults[1] = -F::ONE;
                    access.mults[2] = -F::ONE;
                    access.mults[3] = -F::ONE;
                    access.mults[4] = -F::ONE;
                }
            },
        );

        Some(RowMajorMatrix::new(values, NUM_CONVERT_PREPROCESSED_COLS))
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
        let events = &input.ext_felt_conversion_events;
        Some(match input.fixed_num_rows(self) {
            Some(num_rows) => num_rows,
            None => next_power_of_two(
                events.len(),
                None,
                <ConvertChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = &input.ext_felt_conversion_events;
        let padded_nb_rows = self.num_rows(input).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_CONVERT_COLS];

        let populate_len = events.len() * NUM_CONVERT_VALUE_COLS;
        values[..populate_len].par_chunks_mut(NUM_CONVERT_VALUE_COLS).zip_eq(events).for_each(
            |(row, event)| {
                let cols: &mut ConvertValueCols<_> = row.borrow_mut();
                cols.input = event.input;
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_CONVERT_COLS))
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for ConvertChip
where
    AB: ZKMRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &ConvertCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &ConvertPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (ConvertValueCols { input }, ConvertAccessCols { addrs, mults }) in
            zip(local.values, prep_local.accesses)
        {
            // First handle the read/write of the extension element.
            // If it's converting extension element to `D` field elements, this is a read.
            // If it's converting `D` field elements to an extension element, this is a write
            // (represented as a `receive` with a negated multiplicity -- see
            // `generate_preprocessed_trace`).
            builder.receive_block(addrs[0], input, mults[0]);

            // Handle the read/write of the field elements.
            // If it's converting extension element to `D` field elements, this is a write.
            // If it's converting `D` field elements to an extension element, this is a read
            // (negated, same trick as above).
            for i in 0..D {
                builder.send_single(addrs[i + 1], input.0[i], mults[i + 1]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::FieldAlgebra;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    #[test]
    fn generate_trace() {
        type F = KoalaBear;

        let input = Block(core::array::from_fn(|i| F::from_canonical_u32(i as u32 + 1)));

        let shard = ExecutionRecord {
            ext_felt_conversion_events: vec![ExtFeltEvent { input }],
            ..Default::default()
        };
        let chip = ConvertChip;
        let trace: RowMajorMatrix<F> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    pub fn prove_ext_felt_convert() {
        use crate::{machine::tests::run_recursion_test_machines, runtime::instruction as instr};

        type F = KoalaBear;

        let vals = [2u32, 3, 5, 7];
        let block = Block(vals.map(F::from_canonical_u32));

        // ext2felt: read the extension element at addr 0, write the 4 felts at addrs 1..=4.
        let mut instructions = vec![
            instr::mem_block(MemAccessKind::Write, 1, 0, block),
            instr::ext_felt(true, [0, 1, 1, 1, 1], [0, 1, 2, 3, 4]),
        ];
        for (i, &v) in vals.iter().enumerate() {
            instructions.push(instr::mem_single(
                MemAccessKind::Read,
                1,
                1 + i as u32,
                F::from_canonical_u32(v),
            ));
        }

        // felt2ext: read the 4 felts at addrs 5..=8, write the extension element at addr 9.
        for (i, &v) in vals.iter().enumerate() {
            instructions.push(instr::mem_single(
                MemAccessKind::Write,
                1,
                5 + i as u32,
                F::from_canonical_u32(v),
            ));
        }
        instructions.push(instr::ext_felt(false, [1, 0, 0, 0, 0], [9, 5, 6, 7, 8]));
        instructions.push(instr::mem_block(MemAccessKind::Read, 1, 9, block));

        let program = RecursionProgram { instructions, ..Default::default() };
        run_recursion_test_machines(program);
    }
}
