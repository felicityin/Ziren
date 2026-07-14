use std::borrow::Borrow;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instruction<F> {
    BaseAlu(BaseAluInstr<F>),
    ExtAlu(ExtAluInstr<F>),
    Mem(MemInstr<F>),
    Poseidon2(Box<Poseidon2Instr<F>>),
    Poseidon2LinearLayer(Box<Poseidon2LinearLayerInstr<F>>),
    Poseidon2SBox(Poseidon2SBoxInstr<F>),
    ExtFelt(ExtFeltInstr<F>),
    Select(SelectInstr<F>),
    HintBits(HintBitsInstr<F>),
    HintAddCurve(HintAddCurveInstr<F>),
    PrefixSumChecks(Box<PrefixSumChecksInstr<F>>),
    Print(PrintInstr<F>),
    HintExt2Felts(HintExt2FeltsInstr<F>),
    CommitPublicValues(Box<CommitPublicValuesInstr<F>>),
    Hint(HintInstr<F>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintBitsInstr<F> {
    /// Addresses and mults of the output bits.
    pub output_addrs_mults: Vec<(Address<F>, F)>,
    /// Input value to decompose.
    pub input_addr: Address<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrintInstr<F> {
    pub field_elt_type: FieldEltType,
    pub addr: Address<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintAddCurveInstr<F> {
    pub output_x_addrs_mults: Vec<(Address<F>, F)>,
    pub output_y_addrs_mults: Vec<(Address<F>, F)>,
    pub input1_x_addrs: Vec<Address<F>>,
    pub input1_y_addrs: Vec<Address<F>>,
    pub input2_x_addrs: Vec<Address<F>>,
    pub input2_y_addrs: Vec<Address<F>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintInstr<F> {
    /// Addresses and mults of the output felts.
    pub output_addrs_mults: Vec<(Address<F>, F)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintExt2FeltsInstr<F> {
    /// Addresses and mults of the output bits.
    pub output_addrs_mults: [(Address<F>, F); D],
    /// Input value to decompose.
    pub input_addr: Address<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FieldEltType {
    Base,
    Extension,
}

pub fn base_alu<F: FieldAlgebra>(
    opcode: BaseAluOpcode,
    mult: u32,
    out: u32,
    in1: u32,
    in2: u32,
) -> Instruction<F> {
    Instruction::BaseAlu(BaseAluInstr {
        opcode,
        mult: F::from_canonical_u32(mult),
        addrs: BaseAluIo {
            out: Address(F::from_canonical_u32(out)),
            in1: Address(F::from_canonical_u32(in1)),
            in2: Address(F::from_canonical_u32(in2)),
        },
    })
}

pub fn ext_alu<F: FieldAlgebra>(
    opcode: ExtAluOpcode,
    mult: u32,
    out: u32,
    in1: u32,
    in2: u32,
) -> Instruction<F> {
    Instruction::ExtAlu(ExtAluInstr {
        opcode,
        mult: F::from_canonical_u32(mult),
        addrs: ExtAluIo {
            out: Address(F::from_canonical_u32(out)),
            in1: Address(F::from_canonical_u32(in1)),
            in2: Address(F::from_canonical_u32(in2)),
        },
    })
}

pub fn mem<F: FieldAlgebra>(kind: MemAccessKind, mult: u32, addr: u32, val: u32) -> Instruction<F> {
    mem_single(kind, mult, addr, F::from_canonical_u32(val))
}

pub fn mem_single<F: FieldAlgebra>(
    kind: MemAccessKind,
    mult: u32,
    addr: u32,
    val: F,
) -> Instruction<F> {
    mem_block(kind, mult, addr, Block::from(val))
}

pub fn mem_ext<F: FieldAlgebra + Copy, EF: FieldExtensionAlgebra<F>>(
    kind: MemAccessKind,
    mult: u32,
    addr: u32,
    val: EF,
) -> Instruction<F> {
    mem_block(kind, mult, addr, val.as_base_slice().into())
}

pub fn mem_block<F: FieldAlgebra>(
    kind: MemAccessKind,
    mult: u32,
    addr: u32,
    val: Block<F>,
) -> Instruction<F> {
    Instruction::Mem(MemInstr {
        addrs: MemIo { inner: Address(F::from_canonical_u32(addr)) },
        vals: MemIo { inner: val },
        mult: F::from_canonical_u32(mult),
        kind,
    })
}

pub fn poseidon2<F: FieldAlgebra>(
    mults: [u32; WIDTH],
    output: [u32; WIDTH],
    input: [u32; WIDTH],
) -> Instruction<F> {
    Instruction::Poseidon2(Box::new(Poseidon2Instr {
        mults: mults.map(F::from_canonical_u32),
        addrs: Poseidon2Io {
            output: output.map(F::from_canonical_u32).map(Address),
            input: input.map(F::from_canonical_u32).map(Address),
        },
    }))
}

pub fn poseidon2_linear_layer<F: FieldAlgebra>(
    external: bool,
    mults: [u32; WIDTH / D],
    output: [u32; WIDTH / D],
    input: [u32; WIDTH / D],
) -> Instruction<F> {
    Instruction::Poseidon2LinearLayer(Box::new(Poseidon2LinearLayerInstr {
        mults: mults.map(F::from_canonical_u32),
        addrs: Poseidon2LinearLayerIo {
            output: output.map(F::from_canonical_u32).map(Address),
            input: input.map(F::from_canonical_u32).map(Address),
        },
        external,
    }))
}

pub fn poseidon2_sbox<F: FieldAlgebra>(
    external: bool,
    mult: u32,
    output: u32,
    input: u32,
) -> Instruction<F> {
    Instruction::Poseidon2SBox(Poseidon2SBoxInstr {
        mult: F::from_canonical_u32(mult),
        addrs: Poseidon2SBoxIo {
            output: Address(F::from_canonical_u32(output)),
            input: Address(F::from_canonical_u32(input)),
        },
        external,
    })
}

pub fn ext_felt<F: FieldAlgebra>(
    ext2felt: bool,
    mults: [u32; 5],
    addrs: [u32; 5],
) -> Instruction<F> {
    Instruction::ExtFelt(ExtFeltInstr {
        mults: mults.map(F::from_canonical_u32),
        addrs: addrs.map(F::from_canonical_u32).map(Address),
        ext2felt,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn select<F: FieldAlgebra>(
    mult1: u32,
    mult2: u32,
    bit: u32,
    out1: u32,
    out2: u32,
    in1: u32,
    in2: u32,
) -> Instruction<F> {
    Instruction::Select(SelectInstr {
        mult1: F::from_canonical_u32(mult1),
        mult2: F::from_canonical_u32(mult2),
        addrs: SelectIo {
            bit: Address(F::from_canonical_u32(bit)),
            out1: Address(F::from_canonical_u32(out1)),
            out2: Address(F::from_canonical_u32(out2)),
            in1: Address(F::from_canonical_u32(in1)),
            in2: Address(F::from_canonical_u32(in2)),
        },
    })
}


#[allow(clippy::too_many_arguments)]
pub fn prefix_sum_checks<F: FieldAlgebra>(
    zero: u32,
    one: u32,
    x1: Vec<u32>,
    x2: Vec<u32>,
    accs: Vec<u32>,
    field_accs: Vec<u32>,
    acc_mults: Vec<u32>,
    field_acc_mults: Vec<u32>,
) -> Instruction<F> {
    Instruction::PrefixSumChecks(Box::new(PrefixSumChecksInstr {
        addrs: PrefixSumChecksIo {
            zero: Address(F::from_canonical_u32(zero)),
            one: Address(F::from_canonical_u32(one)),
            x1: x1.iter().map(|elm| Address(F::from_canonical_u32(*elm))).collect(),
            x2: x2.iter().map(|elm| Address(F::from_canonical_u32(*elm))).collect(),
            accs: accs.iter().map(|elm| Address(F::from_canonical_u32(*elm))).collect(),
            field_accs: field_accs.iter().map(|elm| Address(F::from_canonical_u32(*elm))).collect(),
        },
        acc_mults: acc_mults.iter().map(|mult| F::from_canonical_u32(*mult)).collect(),
        field_acc_mults: field_acc_mults.iter().map(|mult| F::from_canonical_u32(*mult)).collect(),
    }))
}

pub fn commit_public_values<F: FieldAlgebra>(
    public_values_a: &RecursionPublicValues<u32>,
) -> Instruction<F> {
    let pv_a = public_values_a.as_array().map(|pv| Address(F::from_canonical_u32(pv)));
    let pv_address: &RecursionPublicValues<Address<F>> = pv_a.as_slice().borrow();

    Instruction::CommitPublicValues(Box::new(CommitPublicValuesInstr {
        pv_addrs: pv_address.clone(),
    }))
}
