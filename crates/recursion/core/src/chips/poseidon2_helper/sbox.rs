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

pub const NUM_SBOX_ENTRIES_PER_ROW: usize = 1;

/// A chip evaluating one Poseidon2 S-box application (`x -> x^3`) per row, over a single
/// extension-sized block. Row-local (degree <= 3), unlike `Poseidon2WideChip`, which evaluates a
/// full permutation (all rounds) per invocation.
#[derive(Default)]
pub struct Poseidon2SBoxChip;

pub const NUM_SBOX_COLS: usize = core::mem::size_of::<Poseidon2SBoxCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxCols<F: Copy> {
    pub values: [Poseidon2SBoxValueCols<F>; NUM_SBOX_ENTRIES_PER_ROW],
}
const NUM_SBOX_VALUE_COLS: usize = core::mem::size_of::<Poseidon2SBoxValueCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxValueCols<F: Copy> {
    pub vals: Poseidon2SBoxIo<Block<F>>,
}

pub const NUM_SBOX_PREPROCESSED_COLS: usize =
    core::mem::size_of::<Poseidon2SBoxPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxPreprocessedCols<F: Copy> {
    pub accesses: [Poseidon2SBoxAccessCols<F>; NUM_SBOX_ENTRIES_PER_ROW],
}

pub const NUM_SBOX_ACCESS_COLS: usize = core::mem::size_of::<Poseidon2SBoxAccessCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxAccessCols<F: Copy> {
    pub addrs: Poseidon2SBoxIo<Address<F>>,
    pub external: F,
    pub internal: F,
}

impl<F: Field> BaseAir<F> for Poseidon2SBoxChip {
    fn width(&self) -> usize {
        NUM_SBOX_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Poseidon2SBoxChip {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    type Error = crate::RecursionChipError;

    fn name(&self) -> String {
        "Poseidon2SBox".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_SBOX_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let nb_rows = instrs_len.div_ceil(NUM_SBOX_ENTRIES_PER_ROW);
        let fixed_num_rows = program.fixed_num_rows(self);
        Some(match fixed_num_rows {
            Some(num_rows) => num_rows,
            None => next_power_of_two(
                nb_rows,
                None,
                <Poseidon2SBoxChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Poseidon2SBox(x) => Some(x),
                _ => None,
            })
            .collect::<Vec<_>>();

        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_SBOX_PREPROCESSED_COLS];

        let populate_len = instrs.len() * NUM_SBOX_ACCESS_COLS;
        values[..populate_len].par_chunks_mut(NUM_SBOX_ACCESS_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let Poseidon2SBoxInstr { addrs, mult, external } = instr;
                let access: &mut Poseidon2SBoxAccessCols<_> = row.borrow_mut();
                access.addrs = addrs.to_owned();
                assert_eq!(*mult, F::ONE, "Poseidon2SBox output must be consumed exactly once");
                if *external {
                    access.external = mult.to_owned();
                    access.internal = F::ZERO;
                } else {
                    access.external = F::ZERO;
                    access.internal = mult.to_owned();
                }
            },
        );

        Some(RowMajorMatrix::new(values, NUM_SBOX_PREPROCESSED_COLS))
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
        let events = &input.poseidon2_sbox_events;
        Some(match input.fixed_num_rows(self) {
            Some(num_rows) => num_rows,
            None => next_power_of_two(
                events.len(),
                None,
                <Poseidon2SBoxChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = &input.poseidon2_sbox_events;
        let padded_nb_rows = self.num_rows(input).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_SBOX_COLS];

        let populate_len = events.len() * NUM_SBOX_VALUE_COLS;
        values[..populate_len].par_chunks_mut(NUM_SBOX_VALUE_COLS).zip_eq(events).for_each(
            |(row, &vals)| {
                let cols: &mut Poseidon2SBoxValueCols<_> = row.borrow_mut();
                cols.vals = vals.to_owned();
                // The AIR's cube constraint applies to every element unconditionally (the
                // external/internal distinction only changes what gets written to memory, via
                // `external`/`internal`-gated `send_block`s in `eval`), so the trace's `output`
                // column must always hold the full cube, even though the event's `output` (which
                // mirrors the actual memory write) only cubes element 0 in the internal case.
                for i in 0..D {
                    cols.vals.output.0[i] =
                        vals.input.0[i] * vals.input.0[i] * vals.input.0[i];
                }
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_SBOX_COLS))
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for Poseidon2SBoxChip
where
    AB: ZKMRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2SBoxCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &Poseidon2SBoxPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (
            Poseidon2SBoxValueCols { vals },
            Poseidon2SBoxAccessCols { addrs, external, internal },
        ) in zip(local.values, prep_local.accesses)
        {
            // Check that the `external`, `internal` flags are boolean, and at most one is on.
            let is_real = external + internal;
            builder.assert_bool(external);
            builder.assert_bool(internal);
            builder.assert_bool(is_real.clone());

            // Read the input from memory. `D` field elements are packed inside the extension.
            builder.receive_block(addrs.input, vals.input, is_real);

            // Constrain that `vals.output.0[i] == vals.input.0[i] ** 3`.
            for i in 0..D {
                builder.assert_eq(
                    vals.input.0[i] * vals.input.0[i] * vals.input.0[i],
                    vals.output.0[i],
                );
            }

            // Write the output to memory in the external SBox case (cube every element).
            builder.send_block(addrs.output, vals.output, external);

            // Write the output to memory in the internal SBox case (cube only the first
            // element, pass the rest through unchanged).
            builder.send_block(
                addrs.output,
                Block([vals.output.0[0], vals.input.0[1], vals.input.0[2], vals.input.0[3]]),
                internal,
            );
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

        let input = Block([F::from_canonical_u32(2), F::ZERO, F::ZERO, F::ZERO]);
        let output = Block([F::from_canonical_u32(8), F::ZERO, F::ZERO, F::ZERO]);

        let shard = ExecutionRecord {
            poseidon2_sbox_events: vec![Poseidon2SBoxEvent { input, output }],
            ..Default::default()
        };
        let chip = Poseidon2SBoxChip;
        let trace: RowMajorMatrix<F> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    pub fn prove_poseidon2_sbox() {
        use crate::{machine::tests::run_recursion_test_machines, runtime::instruction as instr};

        type F = KoalaBear;

        let mut addr = 0u32;
        let mut instructions = Vec::new();

        for (external, input) in
            [(true, [2u32, 3, 5, 7]), (false, [11u32, 13, 17, 19])].into_iter()
        {
            let in_addr = addr;
            addr += 1;
            let out_addr = addr;
            addr += 1;

            instructions.push(instr::mem_block(
                MemAccessKind::Write,
                1,
                in_addr,
                Block(input.map(F::from_canonical_u32)),
            ));
            instructions.push(instr::poseidon2_sbox(external, 1, out_addr, in_addr));

            let cube = |x: u32| {
                let x = F::from_canonical_u32(x);
                x * x * x
            };
            let expected = if external {
                Block([cube(input[0]), cube(input[1]), cube(input[2]), cube(input[3])])
            } else {
                Block([
                    cube(input[0]),
                    F::from_canonical_u32(input[1]),
                    F::from_canonical_u32(input[2]),
                    F::from_canonical_u32(input[3]),
                ])
            };
            instructions.push(instr::mem_block(MemAccessKind::Read, 1, out_addr, expected));
        }

        let program = RecursionProgram { instructions, ..Default::default() };
        run_recursion_test_machines(program);
    }
}
