use core::borrow::Borrow;
use std::borrow::BorrowMut;
use std::iter::zip;

use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::{Field, FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

use zkm_core_machine::utils::next_multiple_of_32;
use zkm_derive::AlignedBorrow;
use zkm_hypercube::air::MachineAir;

use crate::{
    builder::ZKMRecursionAirBuilder,
    chips::poseidon2_wide::{external_linear_layer, internal_linear_layer},
    *,
};

pub const NUM_LINEAR_ENTRIES_PER_ROW: usize = 1;

/// A chip evaluating one external or internal Poseidon2 linear-layer round per row, over the
/// full `WIDTH`-sized permutation state (packed as `WIDTH / D` extension-sized blocks). Row-local
/// (degree <= 3), unlike `Poseidon2WideChip`, which evaluates a full permutation (all rounds) per
/// invocation.
#[derive(Default)]
pub struct Poseidon2LinearLayerChip;

pub const NUM_LINEAR_COLS: usize = core::mem::size_of::<Poseidon2LinearLayerCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerCols<F: Copy> {
    pub values: [Poseidon2LinearLayerValueCols<F>; NUM_LINEAR_ENTRIES_PER_ROW],
}
const NUM_LINEAR_VALUE_COLS: usize = core::mem::size_of::<Poseidon2LinearLayerValueCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerValueCols<F: Copy> {
    pub input: [Block<F>; WIDTH / D],
}

pub const NUM_LINEAR_PREPROCESSED_COLS: usize =
    core::mem::size_of::<Poseidon2LinearLayerPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerPreprocessedCols<F: Copy> {
    pub accesses: [Poseidon2LinearLayerAccessCols<F>; NUM_LINEAR_ENTRIES_PER_ROW],
}

pub const NUM_LINEAR_ACCESS_COLS: usize =
    core::mem::size_of::<Poseidon2LinearLayerAccessCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerAccessCols<F: Copy> {
    pub addrs: Poseidon2LinearLayerIo<Address<F>>,
    pub external: F,
    pub internal: F,
}

impl<F: Field> BaseAir<F> for Poseidon2LinearLayerChip {
    fn width(&self) -> usize {
        NUM_LINEAR_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for Poseidon2LinearLayerChip {
    type Record = ExecutionRecord<F>;

    type Program = crate::RecursionProgram<F>;

    type Error = crate::RecursionChipError;

    fn name(&self) -> String {
        "Poseidon2LinearLayer".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_LINEAR_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let nb_rows = instrs_len.div_ceil(NUM_LINEAR_ENTRIES_PER_ROW);
        let fixed_num_rows = program.fixed_num_rows(self);
        Some(match fixed_num_rows {
            Some(num_rows) => num_rows,
            None => next_multiple_of_32(
                nb_rows,
                None,
                <Poseidon2LinearLayerChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Poseidon2LinearLayer(x) => Some(x.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>();

        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_LINEAR_PREPROCESSED_COLS];

        let populate_len = instrs.len() * NUM_LINEAR_ACCESS_COLS;
        values[..populate_len].par_chunks_mut(NUM_LINEAR_ACCESS_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let Poseidon2LinearLayerInstr { addrs, mults, external } = instr;
                let access: &mut Poseidon2LinearLayerAccessCols<_> = row.borrow_mut();
                access.addrs = addrs.to_owned();
                for mult in mults {
                    assert_eq!(
                        *mult,
                        F::ONE,
                        "Poseidon2LinearLayer output must be consumed exactly once"
                    );
                }
                if *external {
                    access.external = F::ONE;
                    access.internal = F::ZERO;
                } else {
                    access.external = F::ZERO;
                    access.internal = F::ONE;
                }
            },
        );

        Some(RowMajorMatrix::new(values, NUM_LINEAR_PREPROCESSED_COLS))
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
        let events = &input.poseidon2_linear_layer_events;
        Some(match input.fixed_num_rows(self) {
            Some(num_rows) => num_rows,
            None => next_multiple_of_32(
                events.len(),
                None,
                <Poseidon2LinearLayerChip as MachineAir<F>>::name(self).as_str(),
            ),
        })
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        let events = &input.poseidon2_linear_layer_events;
        let padded_nb_rows = self.num_rows(input).unwrap();
        let mut values = vec![F::ZERO; padded_nb_rows * NUM_LINEAR_COLS];

        let populate_len = events.len() * NUM_LINEAR_VALUE_COLS;
        values[..populate_len].par_chunks_mut(NUM_LINEAR_VALUE_COLS).zip_eq(events).for_each(
            |(row, event)| {
                let cols: &mut Poseidon2LinearLayerValueCols<_> = row.borrow_mut();
                cols.input = event.input;
            },
        );

        Ok(RowMajorMatrix::new(values, NUM_LINEAR_COLS))
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for Poseidon2LinearLayerChip
where
    AB: ZKMRecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2LinearLayerCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &Poseidon2LinearLayerPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (
            Poseidon2LinearLayerValueCols { input },
            Poseidon2LinearLayerAccessCols { addrs, external, internal },
        ) in zip(local.values, prep_local.accesses)
        {
            // Check that the `external`, `internal` flags are boolean, and at most one is on.
            let is_real = external + internal;
            builder.assert_bool(external);
            builder.assert_bool(internal);
            builder.assert_bool(is_real.clone());

            // Read the inputs from memory. The inputs are packed in extension elements.
            for i in 0..WIDTH / D {
                builder.receive_block(addrs.input[i], input[i], is_real.clone());
            }

            let mut state_external: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::ZERO);
            let mut state_internal: [AB::Expr; WIDTH] = core::array::from_fn(|_| AB::Expr::ZERO);

            // Unpack the extension elements into field elements.
            for i in 0..WIDTH / D {
                for j in 0..D {
                    state_external[i * D + j] = input[i].0[j].into();
                    state_internal[i * D + j] = input[i].0[j].into();
                }
            }

            // Apply the external/internal linear layer.
            external_linear_layer(&mut state_external);
            internal_linear_layer(&mut state_internal);

            // Write the output to memory for each case.
            for i in 0..WIDTH / D {
                builder.send_block(
                    Address(addrs.output[i].0.into()),
                    Block([
                        state_external[i * D].clone(),
                        state_external[i * D + 1].clone(),
                        state_external[i * D + 2].clone(),
                        state_external[i * D + 3].clone(),
                    ]),
                    external,
                );
                builder.send_block(
                    Address(addrs.output[i].0.into()),
                    Block([
                        state_internal[i * D].clone(),
                        state_internal[i * D + 1].clone(),
                        state_internal[i * D + 2].clone(),
                        state_internal[i * D + 3].clone(),
                    ]),
                    internal,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    #[test]
    fn generate_trace() {
        type F = KoalaBear;

        let input: [Block<F>; WIDTH / D] = core::array::from_fn(|i| {
            Block(core::array::from_fn(|j| F::from_canonical_u32((i * D + j) as u32)))
        });

        let shard = ExecutionRecord {
            poseidon2_linear_layer_events: vec![Poseidon2LinearLayerIo {
                input,
                output: input,
            }],
            ..Default::default()
        };
        let chip = Poseidon2LinearLayerChip;
        let trace: RowMajorMatrix<F> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }

    #[test]
    pub fn prove_poseidon2_linear_layer() {
        use crate::{machine::tests::run_recursion_test_machines, runtime::instruction as instr};

        type F = KoalaBear;

        let mut addr = 0u32;
        let mut instructions = Vec::new();

        for external in [true, false] {
            let input: [u32; WIDTH] = core::array::from_fn(|i| (i + 1) as u32);
            let in_addrs: [u32; WIDTH / D] = core::array::from_fn(|i| addr + i as u32);
            addr += (WIDTH / D) as u32;
            let out_addrs: [u32; WIDTH / D] = core::array::from_fn(|i| addr + i as u32);
            addr += (WIDTH / D) as u32;

            for i in 0..WIDTH / D {
                let block = Block(core::array::from_fn(|j| {
                    F::from_canonical_u32(input[i * D + j])
                }));
                instructions.push(instr::mem_block(MemAccessKind::Write, 1, in_addrs[i], block));
            }

            let mut state: [F; WIDTH] = input.map(F::from_canonical_u32);
            if external {
                external_linear_layer(&mut state);
            } else {
                internal_linear_layer(&mut state);
            }

            instructions.push(instr::poseidon2_linear_layer(
                external,
                [1; WIDTH / D],
                out_addrs,
                in_addrs,
            ));

            for i in 0..WIDTH / D {
                let block = Block(core::array::from_fn(|j| state[i * D + j]));
                instructions.push(instr::mem_block(MemAccessKind::Read, 1, out_addrs[i], block));
            }
        }

        let program = RecursionProgram { instructions, ..Default::default() };
        run_recursion_test_machines(program);
    }
}
