pub mod instruction;
mod memory;
mod opcode;
mod program;
mod record;

// Avoid triggering annoying branch of thiserror derive macro.
use backtrace::Backtrace as Trace;
use hashbrown::HashMap;
use instruction::HintAddCurveInstr;
pub use instruction::Instruction;
use instruction::{FieldEltType, HintBitsInstr, HintExt2FeltsInstr, HintInstr, PrintInstr};
use itertools::Itertools;
use machine::RecursionAirEventCount;
use memory::*;
pub use opcode::*;
pub use program::*;
pub use record::*;

use std::{
    array,
    borrow::Borrow,
    collections::VecDeque,
    fmt::Debug,
    io::{stdout, Write},
    iter::zip,
    marker::PhantomData,
    sync::Arc,
};

use p3_field::{ExtensionField, FieldAlgebra, FieldExtensionAlgebra, PrimeField32};
use p3_koala_bear::Poseidon2ExternalLayerKoalaBear;
use p3_poseidon2::Poseidon2;
use p3_symmetric::{CryptographicPermutation, Permutation};
use thiserror::Error;

use zkm_hypercube::septic_curve::SepticCurve;
use zkm_hypercube::septic_extension::SepticExtension;

use crate::air::{Block, RECURSIVE_PROOF_NUM_PV_ELTS};
use crate::chips::poseidon2_wide::{external_linear_layer, internal_linear_layer};

/// TODO expand glob import once things are organized enough
use crate::*;

pub const STACK_SIZE: usize = 1 << 24;
pub const MEMORY_SIZE: usize = 1 << 28;

/// The heap pointer address.
pub const HEAP_PTR: i32 = -4;
pub const HEAP_START_ADDRESS: usize = STACK_SIZE + 4;

/// The width of the Poseidon2 permutation.
pub const PERMUTATION_WIDTH: usize = 16;
pub const POSEIDON2_SBOX_DEGREE: u64 = 3;
pub const HASH_RATE: usize = 8;

/// The current verifier implementation assumes that we are using a 256-bit hash with 32-bit
/// elements.
pub const DIGEST_SIZE: usize = 8;

pub const NUM_BITS: usize = 31;

pub const D: usize = 4;

#[derive(Debug, Clone, Default)]
pub struct CycleTrackerEntry {
    pub span_entered: bool,
    pub span_enter_cycle: usize,
    pub cumulative_cycles: usize,
}

/// TODO fully document.
/// Taken from [`zkm_recursion_core::runtime::Runtime`].
/// Many missing things (compared to the old `Runtime`) will need to be implemented.
pub struct Runtime<'a, F: PrimeField32, EF: ExtensionField<F>, Diffusion> {
    pub timestamp: usize,

    pub nb_poseidons: usize,

    pub nb_wide_poseidons: usize,

    pub nb_bit_decompositions: usize,

    pub nb_ext_ops: usize,

    pub nb_base_ops: usize,

    pub nb_memory_ops: usize,

    pub nb_branch_ops: usize,

    pub nb_select: usize,

    pub nb_prefix_sum_checks: usize,

    pub nb_print_f: usize,

    pub nb_print_e: usize,

    /// The current clock.
    pub clk: F,

    /// The program counter.
    pub pc: F,

    /// The program.
    pub program: Arc<RecursionProgram<F>>,

    /// Memory. From canonical usize of an Address to a MemoryEntry.
    pub memory: MemVecMap<F>,

    /// The execution record.
    pub record: ExecutionRecord<F>,

    pub witness_stream: VecDeque<Block<F>>,

    pub cycle_tracker: HashMap<String, CycleTrackerEntry>,

    /// The stream that print statements write to.
    pub debug_stdout: Box<dyn Write + 'a>,

    /// Entries for dealing with the Poseidon2 hash state.
    perm: Option<
        Poseidon2<
            F::Packing,
            Poseidon2ExternalLayerKoalaBear<16>,
            Diffusion,
            PERMUTATION_WIDTH,
            POSEIDON2_SBOX_DEGREE,
        >,
    >,

    _marker_ef: PhantomData<EF>,

    _marker_diffusion: PhantomData<Diffusion>,
}

#[derive(Error, Debug)]
pub enum RuntimeError<F: Debug, EF: Debug> {
    #[error(
        "attempted to perform base field division {in1:?}/{in2:?} \
        from instruction {instr:?} at pc {pc:?}\nnearest pc with backtrace:\n{trace:?}"
    )]
    DivFOutOfDomain {
        in1: F,
        in2: F,
        instr: BaseAluInstr<F>,
        pc: usize,
        trace: Option<(usize, Trace)>,
    },
    #[error(
        "attempted to perform extension field division {in1:?}/{in2:?} \
        from instruction {instr:?} at pc {pc:?}\nnearest pc with backtrace:\n{trace:?}"
    )]
    DivEOutOfDomain {
        in1: EF,
        in2: EF,
        instr: ExtAluInstr<F>,
        pc: usize,
        trace: Option<(usize, Trace)>,
    },
    #[error("failed to print to `debug_stdout`: {0}")]
    DebugPrint(#[from] std::io::Error),
    #[error("attempted to read from empty witness stream")]
    EmptyWitnessStream,
}

impl<F: PrimeField32, EF: ExtensionField<F>, Diffusion> Runtime<'_, F, EF, Diffusion>
where
    Poseidon2<
        F::Packing,
        Poseidon2ExternalLayerKoalaBear<16>,
        Diffusion,
        PERMUTATION_WIDTH,
        POSEIDON2_SBOX_DEGREE,
    >: CryptographicPermutation<[F; PERMUTATION_WIDTH]>,
{
    pub fn new(
        program: Arc<RecursionProgram<F>>,
        perm: Poseidon2<
            F::Packing,
            Poseidon2ExternalLayerKoalaBear<16>,
            Diffusion,
            PERMUTATION_WIDTH,
            POSEIDON2_SBOX_DEGREE,
        >,
    ) -> Self {
        let record = ExecutionRecord::<F> { program: program.clone(), ..Default::default() };
        let memory = Memory::with_capacity(program.total_memory);
        Self {
            timestamp: 0,
            nb_poseidons: 0,
            nb_wide_poseidons: 0,
            nb_bit_decompositions: 0,
            nb_select: 0,
            nb_ext_ops: 0,
            nb_base_ops: 0,
            nb_memory_ops: 0,
            nb_branch_ops: 0,
            nb_prefix_sum_checks: 0,
            nb_print_f: 0,
            nb_print_e: 0,
            clk: F::ZERO,
            program,
            pc: F::ZERO,
            memory,
            record,
            witness_stream: VecDeque::new(),
            cycle_tracker: HashMap::new(),
            debug_stdout: Box::new(stdout()),
            perm: Some(perm),
            _marker_ef: PhantomData,
            _marker_diffusion: PhantomData,
        }
    }

    pub fn print_stats(&self) {
        tracing::debug!("Total Cycles: {}", self.timestamp);
        tracing::debug!("Poseidon Skinny Operations: {}", self.nb_poseidons);
        tracing::debug!("Poseidon Wide Operations: {}", self.nb_wide_poseidons);
        tracing::debug!("Field Operations: {}", self.nb_base_ops);
        tracing::debug!("Select Operations: {}", self.nb_select);
        tracing::debug!("Extension Operations: {}", self.nb_ext_ops);
        tracing::debug!("PrefixSumChecks Operations: {}", self.nb_prefix_sum_checks);
        tracing::debug!("Memory Operations: {}", self.nb_memory_ops);
        tracing::debug!("Branch Operations: {}", self.nb_branch_ops);
        for (name, entry) in self.cycle_tracker.iter().sorted_by_key(|(name, _)| *name) {
            tracing::debug!("> {}: {}", name, entry.cumulative_cycles);
        }
    }

    fn nearest_pc_backtrace(&mut self) -> Option<(usize, Trace)> {
        let trap_pc = self.pc.as_canonical_u32() as usize;
        let trace = self.program.traces.get(trap_pc).cloned()?;
        if let Some(mut trace) = trace {
            trace.resolve();
            Some((trap_pc, trace))
        } else {
            (0..trap_pc)
                .rev()
                .filter_map(|nearby_pc| {
                    let mut trace = self.program.traces.get(nearby_pc)?.clone()?;
                    trace.resolve();
                    Some((nearby_pc, trace))
                })
                .next()
        }
    }

    /// Compare to [zkm_recursion_core::runtime::Runtime::run].
    pub fn run(&mut self) -> Result<(), RuntimeError<F, EF>> {
        let early_exit_ts = std::env::var("RECURSION_EARLY_EXIT_TS")
            .map_or(usize::MAX, |ts: String| ts.parse().unwrap());
        self.preallocate_record();
        while self.pc < F::from_canonical_u32(self.program.instructions.len() as u32) {
            let idx = self.pc.as_canonical_u32() as usize;
            let instruction = self.program.instructions[idx].clone();

            let next_clk = self.clk + F::from_canonical_u32(4);
            let next_pc = self.pc + F::ONE;
            match instruction {
                Instruction::BaseAlu(instr @ BaseAluInstr { opcode, mult, addrs }) => {
                    self.nb_base_ops += 1;
                    let in1 = self.memory.mr(addrs.in1).val[0];
                    let in2 = self.memory.mr(addrs.in2).val[0];
                    // Do the computation.
                    let out = match opcode {
                        BaseAluOpcode::AddF => in1 + in2,
                        BaseAluOpcode::SubF => in1 - in2,
                        BaseAluOpcode::MulF => in1 * in2,
                        BaseAluOpcode::DivF => match in2.try_inverse().map(|x| x * in1) {
                            Some(x) => x,
                            None => {
                                // Check for division exceptions and error. Note that 0/0 is defined
                                // to be 1.
                                if in1.is_zero() {
                                    FieldAlgebra::ONE
                                } else {
                                    return Err(RuntimeError::DivFOutOfDomain {
                                        in1,
                                        in2,
                                        instr,
                                        pc: self.pc.as_canonical_u32() as usize,
                                        trace: self.nearest_pc_backtrace(),
                                    });
                                }
                            }
                        },
                    };
                    self.memory.mw(addrs.out, Block::from(out), mult);
                    self.record.base_alu_events.push(BaseAluEvent { out, in1, in2 });
                }
                Instruction::ExtAlu(instr @ ExtAluInstr { opcode, mult, addrs }) => {
                    self.nb_ext_ops += 1;
                    let in1 = self.memory.mr(addrs.in1).val;
                    let in2 = self.memory.mr(addrs.in2).val;
                    // Do the computation.
                    let in1_ef = EF::from_base_slice(&in1.0);
                    let in2_ef = EF::from_base_slice(&in2.0);
                    let out_ef = match opcode {
                        ExtAluOpcode::AddE => in1_ef + in2_ef,
                        ExtAluOpcode::SubE => in1_ef - in2_ef,
                        ExtAluOpcode::MulE => in1_ef * in2_ef,
                        ExtAluOpcode::DivE => match in2_ef.try_inverse().map(|x| x * in1_ef) {
                            Some(x) => x,
                            None => {
                                // Check for division exceptions and error. Note that 0/0 is defined
                                // to be 1.
                                if in1_ef.is_zero() {
                                    FieldAlgebra::ONE
                                } else {
                                    return Err(RuntimeError::DivEOutOfDomain {
                                        in1: in1_ef,
                                        in2: in2_ef,
                                        instr,
                                        pc: self.pc.as_canonical_u32() as usize,
                                        trace: self.nearest_pc_backtrace(),
                                    });
                                }
                            }
                        },
                    };
                    let out = Block::from(out_ef.as_base_slice());
                    self.memory.mw(addrs.out, out, mult);
                    self.record.ext_alu_events.push(ExtAluEvent { out, in1, in2 });
                }
                Instruction::Mem(MemInstr {
                    addrs: MemIo { inner: addr },
                    vals: MemIo { inner: val },
                    mult,
                    kind,
                }) => {
                    self.nb_memory_ops += 1;
                    match kind {
                        MemAccessKind::Read => {
                            let mem_entry = self.memory.mr_mult(addr, mult);
                            assert_eq!(
                                mem_entry.val, val,
                                "stored memory value should be the specified value"
                            );
                        }
                        MemAccessKind::Write => drop(self.memory.mw(addr, val, mult)),
                    }
                    self.record.mem_const_count += 1;
                }
                Instruction::Poseidon2(instr) => {
                    let Poseidon2Instr { addrs: Poseidon2Io { input, output }, mults } = *instr;
                    self.nb_poseidons += 1;
                    let in_vals = std::array::from_fn(|i| self.memory.mr(input[i]).val[0]);
                    let perm_output = self.perm.as_ref().unwrap().permute(in_vals);

                    perm_output.iter().zip(output).zip(mults).for_each(|((&val, addr), mult)| {
                        self.memory.mw(addr, Block::from(val), mult);
                    });
                    self.record
                        .poseidon2_events
                        .push(Poseidon2Event { input: in_vals, output: perm_output });
                }
                Instruction::Poseidon2LinearLayer(instr) => {
                    let Poseidon2LinearLayerInstr {
                        addrs: Poseidon2LinearLayerIo { input, output },
                        mults,
                        external,
                    } = *instr;
                    let mut state = [F::ZERO; PERMUTATION_WIDTH];
                    let mut io_input = [Block::from(F::ZERO); PERMUTATION_WIDTH / D];
                    let mut io_output = [Block::from(F::ZERO); PERMUTATION_WIDTH / D];
                    for i in 0..PERMUTATION_WIDTH / D {
                        io_input[i] = self.memory.mr(input[i]).val;
                        for j in 0..D {
                            state[i * D + j] = io_input[i].0[j];
                        }
                    }
                    if external {
                        external_linear_layer(&mut state);
                    } else {
                        internal_linear_layer(&mut state);
                    }
                    for i in 0..PERMUTATION_WIDTH / D {
                        io_output[i] = Block(state[i * D..i * D + D].try_into().unwrap());
                        self.memory.mw(output[i], io_output[i], mults[i]);
                    }
                    self.record
                        .poseidon2_linear_layer_events
                        .push(Poseidon2LinearLayerEvent { input: io_input, output: io_output });
                }
                Instruction::Poseidon2SBox(Poseidon2SBoxInstr {
                    addrs: Poseidon2SBoxIo { input, output },
                    mult,
                    external,
                }) => {
                    let io_input = self.memory.mr(input).val;
                    let cube = |x: F| x * x * x;

                    let io_output = if external {
                        Block([
                            cube(io_input.0[0]),
                            cube(io_input.0[1]),
                            cube(io_input.0[2]),
                            cube(io_input.0[3]),
                        ])
                    } else {
                        Block([cube(io_input.0[0]), io_input.0[1], io_input.0[2], io_input.0[3]])
                    };
                    self.memory.mw(output, io_output, mult);
                    self.record
                        .poseidon2_sbox_events
                        .push(Poseidon2SBoxEvent { input: io_input, output: io_output });
                }
                Instruction::Select(SelectInstr {
                    addrs: SelectIo { bit, out1, out2, in1, in2 },
                    mult1,
                    mult2,
                }) => {
                    self.nb_select += 1;
                    let bit = self.memory.mr(bit).val[0];
                    let in1 = self.memory.mr(in1).val[0];
                    let in2 = self.memory.mr(in2).val[0];
                    let out1_val = bit * in2 + (F::ONE - bit) * in1;
                    let out2_val = bit * in1 + (F::ONE - bit) * in2;
                    self.memory.mw(out1, Block::from(out1_val), mult1);
                    self.memory.mw(out2, Block::from(out2_val), mult2);
                    self.record.select_events.push(SelectEvent {
                        bit,
                        out1: out1_val,
                        out2: out2_val,
                        in1,
                        in2,
                    })
                }
                Instruction::HintBits(HintBitsInstr { output_addrs_mults, input_addr }) => {
                    self.nb_bit_decompositions += 1;
                    let num = self.memory.mr_mult(input_addr, F::ZERO).val[0].as_canonical_u32();
                    // Decompose the num into LE bits.
                    let bits = (0..output_addrs_mults.len())
                        .map(|i| Block::from(F::from_canonical_u32((num >> i) & 1)))
                        .collect::<Vec<_>>();
                    // Write the bits to the array at dst.
                    for (bit, (addr, mult)) in bits.into_iter().zip(output_addrs_mults) {
                        self.memory.mw(addr, bit, mult);
                        self.record.mem_var_events.push(MemEvent { inner: bit });
                    }
                }
                Instruction::HintAddCurve(HintAddCurveInstr {
                    output_x_addrs_mults,
                    output_y_addrs_mults,
                    input1_x_addrs,
                    input1_y_addrs,
                    input2_x_addrs,
                    input2_y_addrs,
                }) => {
                    let input1_x = SepticExtension::<F>::from_base_fn(|i| {
                        self.memory.mr_mult(input1_x_addrs[i], F::ZERO).val[0]
                    });
                    let input1_y = SepticExtension::<F>::from_base_fn(|i| {
                        self.memory.mr_mult(input1_y_addrs[i], F::ZERO).val[0]
                    });
                    let input2_x = SepticExtension::<F>::from_base_fn(|i| {
                        self.memory.mr_mult(input2_x_addrs[i], F::ZERO).val[0]
                    });
                    let input2_y = SepticExtension::<F>::from_base_fn(|i| {
                        self.memory.mr_mult(input2_y_addrs[i], F::ZERO).val[0]
                    });
                    let point1 = SepticCurve { x: input1_x, y: input1_y };
                    let point2 = SepticCurve { x: input2_x, y: input2_y };
                    let output = point1.add_incomplete(point2);

                    for (val, (addr, mult)) in
                        output.x.0.into_iter().zip(output_x_addrs_mults.into_iter())
                    {
                        self.memory.mw(addr, Block::from(val), mult);
                        self.record.mem_var_events.push(MemEvent { inner: Block::from(val) });
                    }
                    for (val, (addr, mult)) in
                        output.y.0.into_iter().zip(output_y_addrs_mults.into_iter())
                    {
                        self.memory.mw(addr, Block::from(val), mult);
                        self.record.mem_var_events.push(MemEvent { inner: Block::from(val) });
                    }
                }

                Instruction::PrefixSumChecks(instr) => {
                    let PrefixSumChecksInstr {
                        addrs: PrefixSumChecksIo { zero, one, x1, x2, accs, field_accs },
                        acc_mults,
                        field_acc_mults,
                    } = *instr;

                    let zero_val = self.memory.mr(zero).val[0];
                    let one_val: EF = self.memory.mr(one).val.ext();
                    let x1_vals =
                        x1.iter().map(|addr| self.memory.mr(*addr).val[0]).collect_vec();
                    let x2_vals: Vec<EF> =
                        x2.iter().map(|addr| self.memory.mr(*addr).val.ext()).collect_vec();

                    self.nb_prefix_sum_checks += x1_vals.len();

                    let mut acc = one_val;
                    let mut field_acc = zero_val;
                    for m in 0..x1_vals.len() {
                        let product = EF::from_base(x1_vals[m]) * x2_vals[m];
                        let lagrange_term =
                            EF::ONE - EF::from_base(x1_vals[m]) - x2_vals[m] + product + product;
                        let new_acc = acc * lagrange_term;
                        let new_field_acc = x1_vals[m] + field_acc * F::from_canonical_u32(2);

                        self.record.prefix_sum_checks_events.push(PrefixSumChecksEvent {
                            x1: x1_vals[m],
                            x2: Block::from(x2_vals[m].as_base_slice()),
                            zero: zero_val,
                            one: Block::from(one_val.as_base_slice()),
                            acc: Block::from(acc.as_base_slice()),
                            new_acc: Block::from(new_acc.as_base_slice()),
                            field_acc,
                            new_field_acc,
                        });

                        acc = new_acc;
                        field_acc = new_field_acc;

                        let _ = self.memory.mw(
                            accs[m],
                            Block::from(acc.as_base_slice()),
                            acc_mults[m],
                        );
                        let _ =
                            self.memory.mw(field_accs[m], Block::from(field_acc), field_acc_mults[m]);
                    }
                }
                Instruction::CommitPublicValues(instr) => {
                    let pv_addrs = instr.pv_addrs.as_array();
                    let pv_values: [F; RECURSIVE_PROOF_NUM_PV_ELTS] =
                        array::from_fn(|i| self.memory.mr(pv_addrs[i]).val[0]);
                    self.record.public_values = *pv_values.as_slice().borrow();
                    self.record
                        .commit_pv_hash_events
                        .push(CommitPublicValuesEvent { public_values: self.record.public_values });
                }

                Instruction::Print(PrintInstr { field_elt_type, addr }) => match field_elt_type {
                    FieldEltType::Base => {
                        self.nb_print_f += 1;
                        let f = self.memory.mr_mult(addr, F::ZERO).val[0];
                        writeln!(self.debug_stdout, "PRINTF={f}")
                    }
                    FieldEltType::Extension => {
                        self.nb_print_e += 1;
                        let ef = self.memory.mr_mult(addr, F::ZERO).val;
                        writeln!(self.debug_stdout, "PRINTEF={ef:?}")
                    }
                }
                .map_err(RuntimeError::DebugPrint)?,
                Instruction::HintExt2Felts(HintExt2FeltsInstr {
                    output_addrs_mults,
                    input_addr,
                }) => {
                    self.nb_bit_decompositions += 1;
                    let fs = self.memory.mr_mult(input_addr, F::ZERO).val;
                    // Write the bits to the array at dst.
                    for (f, (addr, mult)) in fs.into_iter().zip(output_addrs_mults) {
                        let felt = Block::from(f);
                        self.memory.mw(addr, felt, mult);
                        self.record.mem_var_events.push(MemEvent { inner: felt });
                    }
                }
                Instruction::Hint(HintInstr { output_addrs_mults }) => {
                    // Check that enough Blocks can be read, so `drain` does not panic.
                    if self.witness_stream.len() < output_addrs_mults.len() {
                        return Err(RuntimeError::EmptyWitnessStream);
                    }
                    let witness = self.witness_stream.drain(0..output_addrs_mults.len());
                    for ((addr, mult), val) in zip(output_addrs_mults, witness) {
                        // Inline [`Self::mw`] to mutably borrow multiple fields of `self`.
                        self.memory.mw(addr, val, mult);
                        self.record.mem_var_events.push(MemEvent { inner: val });
                    }
                }
            }

            self.pc = next_pc;
            self.clk = next_clk;
            self.timestamp += 1;

            if self.timestamp >= early_exit_ts {
                break;
            }
        }
        Ok(())
    }

    pub fn preallocate_record(&mut self) {
        let event_counts = self
            .program
            .instructions
            .iter()
            .fold(RecursionAirEventCount::default(), |heights, instruction| heights + instruction);
        self.record.poseidon2_events.reserve(event_counts.poseidon2_wide_events);
        self.record.mem_var_events.reserve(event_counts.mem_var_events);
        self.record.base_alu_events.reserve(event_counts.base_alu_events);
        self.record.ext_alu_events.reserve(event_counts.ext_alu_events);
        self.record.select_events.reserve(event_counts.select_events);
    }
}
