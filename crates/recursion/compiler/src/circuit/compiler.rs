use chips::poseidon2_skinny::WIDTH;
use core::fmt::Debug;
use instruction::{
    FieldEltType, HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr, PrintInstr,
};
use itertools::Itertools;
use p3_field::{
    Field, FieldAlgebra, FieldExtensionAlgebra, PrimeField, PrimeField64, TwoAdicField,
};
use std::{borrow::Borrow, collections::HashMap, mem::transmute};
use vec_map::VecMap;
use zkm_core_machine::utils::{zkm_debug_mode, SpanBuilder};
use zkm_recursion_core::{
    air::{Block, RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS},
    BaseAluInstr, BaseAluOpcode,
};
use zkm_hypercube::septic_curve::SepticCurve;

use zkm_recursion_core::*;

use crate::prelude::*;

/// The number of instructions to preallocate in a recursion program
const PREALLOC_INSTRUCTIONS: usize = 10000000;

/// The backend for the circuit compiler.
#[derive(Debug, Clone, Default)]
pub struct AsmCompiler<C: Config> {
    pub next_addr: C::F,
    /// Map the frame pointers of the variables to the "physical" addresses.
    pub virtual_to_physical: VecMap<Address<C::F>>,
    /// Map base or extension field constants to "physical" addresses and mults.
    pub consts: HashMap<Imm<C::F, C::EF>, (Address<C::F>, C::F)>,
    /// Map each "physical" address to its read count.
    pub addr_to_mult: VecMap<C::F>,
}

impl<C: Config> AsmCompiler<C>
where
    C::F: PrimeField64,
{
    /// Allocate a fresh address. Checks that the address space is not full.
    pub fn alloc(next_addr: &mut C::F) -> Address<C::F> {
        let id = Address(*next_addr);
        *next_addr += C::F::ONE;
        if next_addr.is_zero() {
            panic!("out of address space");
        }
        id
    }

    /// Map `fp` to its existing address without changing its mult.
    ///
    /// Ensures that `fp` has already been assigned an address.
    pub fn read_ghost_vaddr(&mut self, vaddr: usize) -> Address<C::F> {
        self.read_vaddr_internal(vaddr, false)
    }

    /// Map `fp` to its existing address and increment its mult.
    ///
    /// Ensures that `fp` has already been assigned an address.
    pub fn read_vaddr(&mut self, vaddr: usize) -> Address<C::F> {
        self.read_vaddr_internal(vaddr, true)
    }

    pub fn read_vaddr_internal(&mut self, vaddr: usize, increment_mult: bool) -> Address<C::F> {
        use vec_map::Entry;
        match self.virtual_to_physical.entry(vaddr) {
            Entry::Vacant(_) => panic!("expected entry: virtual_physical[{vaddr:?}]"),
            Entry::Occupied(entry) => {
                if increment_mult {
                    // This is a read, so we increment the mult.
                    match self.addr_to_mult.get_mut(entry.get().as_usize()) {
                        Some(mult) => *mult += C::F::ONE,
                        None => panic!("expected entry: virtual_physical[{vaddr:?}]"),
                    }
                }
                *entry.into_mut()
            }
        }
    }

    /// Map `fp` to a fresh address and initialize the mult to 0.
    ///
    /// Ensures that `fp` has not already been written to.
    pub fn write_fp(&mut self, vaddr: usize) -> Address<C::F> {
        use vec_map::Entry;
        match self.virtual_to_physical.entry(vaddr) {
            Entry::Vacant(entry) => {
                let addr = Self::alloc(&mut self.next_addr);
                // This is a write, so we set the mult to zero.
                if let Some(x) = self.addr_to_mult.insert(addr.as_usize(), C::F::ZERO) {
                    panic!("unexpected entry in addr_to_mult: {x:?}");
                }
                *entry.insert(addr)
            }
            Entry::Occupied(entry) => {
                panic!("unexpected entry: virtual_to_physical[{:?}] = {:?}", vaddr, entry.get())
            }
        }
    }

    /// Increment the existing `mult` associated with `addr`.
    ///
    /// Ensures that `addr` has already been assigned a `mult`.
    pub fn read_addr(&mut self, addr: Address<C::F>) -> &mut C::F {
        self.read_addr_internal(addr, true)
    }

    /// Retrieves `mult` associated with `addr`.
    ///
    /// Ensures that `addr` has already been assigned a `mult`.
    pub fn read_ghost_addr(&mut self, addr: Address<C::F>) -> &mut C::F {
        self.read_addr_internal(addr, true)
    }

    fn read_addr_internal(&mut self, addr: Address<C::F>, increment_mult: bool) -> &mut C::F {
        use vec_map::Entry;
        match self.addr_to_mult.entry(addr.as_usize()) {
            Entry::Vacant(_) => panic!("expected entry: addr_to_mult[{:?}]", addr.as_usize()),
            Entry::Occupied(entry) => {
                // This is a read, so we increment the mult.
                let mult = entry.into_mut();
                if increment_mult {
                    *mult += C::F::ONE;
                }
                mult
            }
        }
    }

    /// Associate a `mult` of zero with `addr`.
    ///
    /// Ensures that `addr` has not already been written to.
    pub fn write_addr(&mut self, addr: Address<C::F>) -> &mut C::F {
        use vec_map::Entry;
        match self.addr_to_mult.entry(addr.as_usize()) {
            Entry::Vacant(entry) => entry.insert(C::F::ZERO),
            Entry::Occupied(entry) => {
                panic!("unexpected entry: addr_to_mult[{:?}] = {:?}", addr.as_usize(), entry.get())
            }
        }
    }

    /// Read a constant (a.k.a. immediate).
    ///
    /// Increments the mult, first creating an entry if it does not yet exist.
    pub fn read_const(&mut self, imm: Imm<C::F, C::EF>) -> Address<C::F> {
        self.consts
            .entry(imm)
            .and_modify(|(_, x)| *x += C::F::ONE)
            .or_insert_with(|| (Self::alloc(&mut self.next_addr), C::F::ONE))
            .0
    }

    /// Read a constant (a.k.a. immediate).
    ///
    /// Does not increment the mult. Creates an entry if it does not yet exist.
    pub fn read_ghost_const(&mut self, imm: Imm<C::F, C::EF>) -> Address<C::F> {
        self.consts.entry(imm).or_insert_with(|| (Self::alloc(&mut self.next_addr), C::F::ZERO)).0
    }

    fn mem_write_const(&mut self, dst: impl Reg<C>, src: Imm<C::F, C::EF>) -> Instruction<C::F> {
        Instruction::Mem(MemInstr {
            addrs: MemIo { inner: dst.write(self) },
            vals: MemIo { inner: src.as_block() },
            mult: C::F::ZERO,
            kind: MemAccessKind::Write,
        })
    }

    fn base_alu(
        &mut self,
        opcode: BaseAluOpcode,
        dst: impl Reg<C>,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
    ) -> Instruction<C::F> {
        Instruction::BaseAlu(BaseAluInstr {
            opcode,
            mult: C::F::ZERO,
            addrs: BaseAluIo { out: dst.write(self), in1: lhs.read(self), in2: rhs.read(self) },
        })
    }

    fn ext_alu(
        &mut self,
        opcode: ExtAluOpcode,
        dst: impl Reg<C>,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
    ) -> Instruction<C::F> {
        Instruction::ExtAlu(ExtAluInstr {
            opcode,
            mult: C::F::ZERO,
            addrs: ExtAluIo { out: dst.write(self), in1: lhs.read(self), in2: rhs.read(self) },
        })
    }

    fn base_assert_eq(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use BaseAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));
        f(self.base_alu(SubF, diff, lhs, rhs));
        f(self.base_alu(DivF, out, diff, Imm::F(C::F::ZERO)));
    }

    fn base_assert_ne(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use BaseAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));

        f(self.base_alu(SubF, diff, lhs, rhs));
        f(self.base_alu(DivF, out, Imm::F(C::F::ONE), diff));
    }

    fn ext_assert_eq(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use ExtAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));

        f(self.ext_alu(SubE, diff, lhs, rhs));
        f(self.ext_alu(DivE, out, diff, Imm::EF(C::EF::ZERO)));
    }

    fn ext_assert_ne(
        &mut self,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
        mut f: impl FnMut(Instruction<C::F>),
    ) {
        use ExtAluOpcode::*;
        let [diff, out] = core::array::from_fn(|_| Self::alloc(&mut self.next_addr));

        f(self.ext_alu(SubE, diff, lhs, rhs));
        f(self.ext_alu(DivE, out, Imm::EF(C::EF::ONE), diff));
    }

    #[inline(always)]
    fn poseidon2_permute(
        &mut self,
        dst: [impl Reg<C>; WIDTH],
        src: [impl Reg<C>; WIDTH],
    ) -> Instruction<C::F> {
        Instruction::Poseidon2(Box::new(Poseidon2Instr {
            addrs: Poseidon2Io {
                input: src.map(|r| r.read(self)),
                output: dst.map(|r| r.write(self)),
            },
            mults: [C::F::ZERO; WIDTH],
        }))
    }

    #[inline(always)]
    fn select(
        &mut self,
        bit: impl Reg<C>,
        dst1: impl Reg<C>,
        dst2: impl Reg<C>,
        lhs: impl Reg<C>,
        rhs: impl Reg<C>,
    ) -> Instruction<C::F> {
        Instruction::Select(SelectInstr {
            addrs: SelectIo {
                bit: bit.read(self),
                out1: dst1.write(self),
                out2: dst2.write(self),
                in1: lhs.read(self),
                in2: rhs.read(self),
            },
            mult1: C::F::ZERO,
            mult2: C::F::ZERO,
        })
    }

    fn hint_bit_decomposition(
        &mut self,
        value: impl Reg<C>,
        output: impl IntoIterator<Item = impl Reg<C>>,
    ) -> Instruction<C::F> {
        Instruction::HintBits(HintBitsInstr {
            output_addrs_mults: output.into_iter().map(|r| (r.write(self), C::F::ZERO)).collect(),
            input_addr: value.read_ghost(self),
        })
    }

    fn add_curve(
        &mut self,
        output: SepticCurve<Felt<C::F>>,
        input1: SepticCurve<Felt<C::F>>,
        input2: SepticCurve<Felt<C::F>>,
    ) -> Instruction<C::F> {
        Instruction::HintAddCurve(HintAddCurveInstr {
            output_x_addrs_mults: output
                .x
                .0
                .into_iter()
                .map(|r| (r.write(self), C::F::ZERO))
                .collect(),
            output_y_addrs_mults: output
                .y
                .0
                .into_iter()
                .map(|r| (r.write(self), C::F::ZERO))
                .collect(),
            input1_x_addrs: input1.x.0.into_iter().map(|value| value.read_ghost(self)).collect(),
            input1_y_addrs: input1.y.0.into_iter().map(|value| value.read_ghost(self)).collect(),
            input2_x_addrs: input2.x.0.into_iter().map(|value| value.read_ghost(self)).collect(),
            input2_y_addrs: input2.y.0.into_iter().map(|value| value.read_ghost(self)).collect(),
        })
    }

    fn prefix_sum_checks(
        &mut self,
        zero: Felt<C::F>,
        one: Ext<C::F, C::EF>,
        accs: Vec<Ext<C::F, C::EF>>,
        field_accs: Vec<Felt<C::F>>,
        x1: Vec<Felt<C::F>>,
        x2: Vec<Ext<C::F, C::EF>>,
    ) -> Instruction<C::F> {
        // First, write to the addresses in `accs`/`field_accs`.
        let acc_write_addrs: Vec<_> = accs.clone().into_iter().map(|r| r.write(self)).collect();
        let field_acc_write_addrs: Vec<_> =
            field_accs.clone().into_iter().map(|r| r.write(self)).collect();
        // Then, read from all but the last address in `accs`/`field_accs`: the chip's own next
        // row consumes each intermediate accumulator (this is what makes the AIR row-local, see
        // `PrefixSumChecksChip`), while the final address is this instruction's true output,
        // consumed by whatever DSL code comes after.
        let _: Vec<_> = accs.iter().take(accs.len() - 1).map(|r| r.read(self)).collect();
        let _: Vec<_> =
            field_accs.iter().take(field_accs.len() - 1).map(|r| r.read(self)).collect();
        Instruction::PrefixSumChecks(Box::new(PrefixSumChecksInstr {
            addrs: PrefixSumChecksIo {
                zero: zero.read(self),
                one: one.read(self),
                x1: x1.into_iter().map(|r| r.read(self)).collect(),
                x2: x2.into_iter().map(|r| r.read(self)).collect(),
                accs: acc_write_addrs,
                field_accs: field_acc_write_addrs,
            },
            acc_mults: vec![C::F::ZERO; accs.len()],
            field_acc_mults: vec![C::F::ZERO; field_accs.len()],
        }))
    }

    fn commit_public_values(
        &mut self,
        public_values: &RecursionPublicValues<Felt<C::F>>,
    ) -> Instruction<C::F> {
        public_values.digest.iter().for_each(|x| {
            let _ = x.read(self);
        });
        let pv_addrs =
            unsafe {
                transmute::<
                    RecursionPublicValues<Felt<C::F>>,
                    [Felt<C::F>; RECURSIVE_PROOF_NUM_PV_ELTS],
                >(*public_values)
            }
            .map(|pv| pv.read_ghost(self));

        let public_values_a: &RecursionPublicValues<Address<C::F>> = pv_addrs.as_slice().borrow();
        Instruction::CommitPublicValues(Box::new(CommitPublicValuesInstr {
            pv_addrs: *public_values_a,
        }))
    }

    fn print_f(&mut self, addr: impl Reg<C>) -> Instruction<C::F> {
        Instruction::Print(PrintInstr {
            field_elt_type: FieldEltType::Base,
            addr: addr.read_ghost(self),
        })
    }

    fn print_e(&mut self, addr: impl Reg<C>) -> Instruction<C::F> {
        Instruction::Print(PrintInstr {
            field_elt_type: FieldEltType::Extension,
            addr: addr.read_ghost(self),
        })
    }

    fn ext2felts(&mut self, felts: [impl Reg<C>; D], ext: impl Reg<C>) -> Instruction<C::F> {
        Instruction::HintExt2Felts(HintExt2FeltsInstr {
            output_addrs_mults: felts.map(|r| (r.write(self), C::F::ZERO)),
            input_addr: ext.read_ghost(self),
        })
    }

    #[inline(always)]
    fn poseidon2_linear_layer(
        &mut self,
        external: bool,
        dst: [impl Reg<C>; WIDTH / D],
        src: [impl Reg<C>; WIDTH / D],
    ) -> Instruction<C::F> {
        Instruction::Poseidon2LinearLayer(Box::new(Poseidon2LinearLayerInstr {
            addrs: Poseidon2LinearLayerIo {
                input: src.map(|r| r.read(self)),
                output: dst.map(|r| r.write(self)),
            },
            mults: [C::F::ZERO; WIDTH / D],
            external,
        }))
    }

    #[inline(always)]
    fn poseidon2_sbox(
        &mut self,
        external: bool,
        dst: impl Reg<C>,
        src: impl Reg<C>,
    ) -> Instruction<C::F> {
        Instruction::Poseidon2SBox(Poseidon2SBoxInstr {
            addrs: Poseidon2SBoxIo { input: src.read(self), output: dst.write(self) },
            mult: C::F::ZERO,
            external,
        })
    }

    /// Converts an extension element to `D` felts using the row-local `ConvertChip`, as opposed
    /// to `ext2felts` (`HintExt2Felts`, a hint operation). Should be used for wrap.
    fn ext2felt_chip(&mut self, felts: [impl Reg<C>; D], ext: impl Reg<C>) -> Instruction<C::F> {
        let ext_addr = ext.read(self);
        let felt_addrs = felts.map(|r| r.write(self));
        Instruction::ExtFelt(ExtFeltInstr {
            addrs: [ext_addr, felt_addrs[0], felt_addrs[1], felt_addrs[2], felt_addrs[3]],
            mults: [C::F::ZERO; 5],
            ext2felt: true,
        })
    }

    /// Converts `D` felts to an extension element using the row-local `ConvertChip`. Should be
    /// used for wrap.
    fn felt2ext_chip(&mut self, ext: impl Reg<C>, felts: [impl Reg<C>; D]) -> Instruction<C::F> {
        let ext_addr = ext.write(self);
        let felt_addrs = felts.map(|r| r.read(self));
        Instruction::ExtFelt(ExtFeltInstr {
            addrs: [ext_addr, felt_addrs[0], felt_addrs[1], felt_addrs[2], felt_addrs[3]],
            mults: [C::F::ZERO; 5],
            ext2felt: false,
        })
    }

    fn hint(&mut self, output: &[impl Reg<C>]) -> Instruction<C::F> {
        Instruction::Hint(HintInstr {
            output_addrs_mults: output.iter().map(|r| (r.write(self), C::F::ZERO)).collect(),
        })
    }

    /// Compiles one instruction, passing one or more instructions to `consumer`.
    ///
    /// We do not simply return a `Vec` for performance reasons --- results would be immediately fed
    /// to `flat_map`, so we employ fusion/deforestation to eliminate intermediate data structures.
    #[inline]
    pub fn compile_one<F>(
        &mut self,
        ir_instr: DslIr<C>,
        mut consumer: impl FnMut(Result<Instruction<C::F>, CompileOneErr<C>>),
    ) where
        F: PrimeField + TwoAdicField,
        C: Config<N = F, F = F> + Debug,
    {
        // For readability. Avoids polluting outer scope.
        use BaseAluOpcode::*;
        use ExtAluOpcode::*;

        let mut f = |instr| consumer(Ok(instr));
        match ir_instr {
            DslIr::ImmV(dst, src) => f(self.mem_write_const(dst, Imm::F(src))),
            DslIr::ImmF(dst, src) => f(self.mem_write_const(dst, Imm::F(src))),
            DslIr::ImmE(dst, src) => f(self.mem_write_const(dst, Imm::EF(src))),

            DslIr::AddV(dst, lhs, rhs) => f(self.base_alu(AddF, dst, lhs, rhs)),
            DslIr::AddVI(dst, lhs, rhs) => f(self.base_alu(AddF, dst, lhs, Imm::F(rhs))),
            DslIr::AddF(dst, lhs, rhs) => f(self.base_alu(AddF, dst, lhs, rhs)),
            DslIr::AddFI(dst, lhs, rhs) => f(self.base_alu(AddF, dst, lhs, Imm::F(rhs))),
            DslIr::AddE(dst, lhs, rhs) => f(self.ext_alu(AddE, dst, lhs, rhs)),
            DslIr::AddEI(dst, lhs, rhs) => f(self.ext_alu(AddE, dst, lhs, Imm::EF(rhs))),
            DslIr::AddEF(dst, lhs, rhs) => f(self.ext_alu(AddE, dst, lhs, rhs)),
            DslIr::AddEFI(dst, lhs, rhs) => f(self.ext_alu(AddE, dst, lhs, Imm::F(rhs))),
            DslIr::AddEFFI(dst, lhs, rhs) => f(self.ext_alu(AddE, dst, lhs, Imm::EF(rhs))),

            DslIr::SubV(dst, lhs, rhs) => f(self.base_alu(SubF, dst, lhs, rhs)),
            DslIr::SubVI(dst, lhs, rhs) => f(self.base_alu(SubF, dst, lhs, Imm::F(rhs))),
            DslIr::SubVIN(dst, lhs, rhs) => f(self.base_alu(SubF, dst, Imm::F(lhs), rhs)),
            DslIr::SubF(dst, lhs, rhs) => f(self.base_alu(SubF, dst, lhs, rhs)),
            DslIr::SubFI(dst, lhs, rhs) => f(self.base_alu(SubF, dst, lhs, Imm::F(rhs))),
            DslIr::SubFIN(dst, lhs, rhs) => f(self.base_alu(SubF, dst, Imm::F(lhs), rhs)),
            DslIr::SubE(dst, lhs, rhs) => f(self.ext_alu(SubE, dst, lhs, rhs)),
            DslIr::SubEI(dst, lhs, rhs) => f(self.ext_alu(SubE, dst, lhs, Imm::EF(rhs))),
            DslIr::SubEIN(dst, lhs, rhs) => f(self.ext_alu(SubE, dst, Imm::EF(lhs), rhs)),
            DslIr::SubEFI(dst, lhs, rhs) => f(self.ext_alu(SubE, dst, lhs, Imm::F(rhs))),
            DslIr::SubEF(dst, lhs, rhs) => f(self.ext_alu(SubE, dst, lhs, rhs)),

            DslIr::MulV(dst, lhs, rhs) => f(self.base_alu(MulF, dst, lhs, rhs)),
            DslIr::MulVI(dst, lhs, rhs) => f(self.base_alu(MulF, dst, lhs, Imm::F(rhs))),
            DslIr::MulF(dst, lhs, rhs) => f(self.base_alu(MulF, dst, lhs, rhs)),
            DslIr::MulFI(dst, lhs, rhs) => f(self.base_alu(MulF, dst, lhs, Imm::F(rhs))),
            DslIr::MulE(dst, lhs, rhs) => f(self.ext_alu(MulE, dst, lhs, rhs)),
            DslIr::MulEI(dst, lhs, rhs) => f(self.ext_alu(MulE, dst, lhs, Imm::EF(rhs))),
            DslIr::MulEFI(dst, lhs, rhs) => f(self.ext_alu(MulE, dst, lhs, Imm::F(rhs))),
            DslIr::MulEF(dst, lhs, rhs) => f(self.ext_alu(MulE, dst, lhs, rhs)),

            DslIr::DivF(dst, lhs, rhs) => f(self.base_alu(DivF, dst, lhs, rhs)),
            DslIr::DivFI(dst, lhs, rhs) => f(self.base_alu(DivF, dst, lhs, Imm::F(rhs))),
            DslIr::DivFIN(dst, lhs, rhs) => f(self.base_alu(DivF, dst, Imm::F(lhs), rhs)),
            DslIr::DivE(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, lhs, rhs)),
            DslIr::DivEI(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, lhs, Imm::EF(rhs))),
            DslIr::DivEIN(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, Imm::EF(lhs), rhs)),
            DslIr::DivEFI(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, lhs, Imm::F(rhs))),
            DslIr::DivEFIN(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, Imm::F(lhs), rhs)),
            DslIr::DivEF(dst, lhs, rhs) => f(self.ext_alu(DivE, dst, lhs, rhs)),

            DslIr::NegV(dst, src) => f(self.base_alu(SubF, dst, Imm::F(C::F::ZERO), src)),
            DslIr::NegF(dst, src) => f(self.base_alu(SubF, dst, Imm::F(C::F::ZERO), src)),
            DslIr::NegE(dst, src) => f(self.ext_alu(SubE, dst, Imm::EF(C::EF::ZERO), src)),
            DslIr::InvV(dst, src) => f(self.base_alu(DivF, dst, Imm::F(C::F::ONE), src)),
            DslIr::InvF(dst, src) => f(self.base_alu(DivF, dst, Imm::F(C::F::ONE), src)),
            DslIr::InvE(dst, src) => f(self.ext_alu(DivE, dst, Imm::F(C::F::ONE), src)),

            DslIr::Select(bit, dst1, dst2, lhs, rhs) => f(self.select(bit, dst1, dst2, lhs, rhs)),

            DslIr::AssertEqV(lhs, rhs) => self.base_assert_eq(lhs, rhs, f),
            DslIr::AssertEqF(lhs, rhs) => self.base_assert_eq(lhs, rhs, f),
            DslIr::AssertEqE(lhs, rhs) => self.ext_assert_eq(lhs, rhs, f),
            DslIr::AssertEqVI(lhs, rhs) => self.base_assert_eq(lhs, Imm::F(rhs), f),
            DslIr::AssertEqFI(lhs, rhs) => self.base_assert_eq(lhs, Imm::F(rhs), f),
            DslIr::AssertEqEI(lhs, rhs) => self.ext_assert_eq(lhs, Imm::EF(rhs), f),

            DslIr::AssertNeV(lhs, rhs) => self.base_assert_ne(lhs, rhs, f),
            DslIr::AssertNeF(lhs, rhs) => self.base_assert_ne(lhs, rhs, f),
            DslIr::AssertNeE(lhs, rhs) => self.ext_assert_ne(lhs, rhs, f),
            DslIr::AssertNeVI(lhs, rhs) => self.base_assert_ne(lhs, Imm::F(rhs), f),
            DslIr::AssertNeFI(lhs, rhs) => self.base_assert_ne(lhs, Imm::F(rhs), f),
            DslIr::AssertNeEI(lhs, rhs) => self.ext_assert_ne(lhs, Imm::EF(rhs), f),

            DslIr::CircuitV2Poseidon2PermuteKoalaBear(data) => {
                f(self.poseidon2_permute(data.0, data.1))
            }
            DslIr::Poseidon2ExternalLinearLayer(data) => {
                f(self.poseidon2_linear_layer(true, data.0, data.1))
            }
            DslIr::Poseidon2InternalLinearLayer(data) => {
                f(self.poseidon2_linear_layer(false, data.0, data.1))
            }
            DslIr::Poseidon2ExternalSBOX(dst, src) => f(self.poseidon2_sbox(true, dst, src)),
            DslIr::Poseidon2InternalSBOX(dst, src) => f(self.poseidon2_sbox(false, dst, src)),
            DslIr::CircuitChipExt2Felt(felts, ext) => f(self.ext2felt_chip(felts, ext)),
            DslIr::CircuitChipFelt2Ext(ext, felts) => f(self.felt2ext_chip(ext, felts)),
            DslIr::CircuitV2HintBitsF(output, value) => {
                f(self.hint_bit_decomposition(value, output))
            }
            DslIr::CircuitV2PrefixSumChecks(data) => {
                f(self.prefix_sum_checks(data.0, data.1, data.2, data.3, data.4, data.5))
            }
            DslIr::CircuitV2CommitPublicValues(public_values) => {
                f(self.commit_public_values(&public_values))
            }
            DslIr::CircuitV2HintAddCurve(output, point1, point2) => {
                f(self.add_curve(output, point1, point2))
            }

            DslIr::PrintV(dst) => f(self.print_f(dst)),
            DslIr::PrintF(dst) => f(self.print_f(dst)),
            DslIr::PrintE(dst) => f(self.print_e(dst)),
            DslIr::CircuitV2HintFelts(output) => f(self.hint(&output)),
            DslIr::CircuitV2HintExts(output) => f(self.hint(&output)),
            DslIr::CircuitExt2Felt(felts, ext) => f(self.ext2felts(felts, ext)),
            DslIr::CycleTrackerV2Enter(name) => {
                consumer(Err(CompileOneErr::CycleTrackerEnter(name)))
            }
            DslIr::CycleTrackerV2Exit => consumer(Err(CompileOneErr::CycleTrackerExit)),
            DslIr::ReduceE(_) => {}
            instr => consumer(Err(CompileOneErr::Unsupported(instr))),
        }
    }

    /// Emit the instructions from a list of operations in the DSL.
    pub fn compile<F>(&mut self, operations: TracedVec<DslIr<C>>) -> RecursionProgram<C::F>
    where
        F: PrimeField + TwoAdicField,
        C: Config<N = F, F = F> + Debug,
    {
        // In debug mode, we perform cycle tracking and keep track of backtraces.
        // Otherwise, we ignore cycle tracking instructions and pass around an empty Vec of traces.
        let debug_mode = zkm_debug_mode();
        // Compile each IR instruction into a list of ASM instructions, then combine them.
        // This step also counts the number of times each address is read from.
        let (mut instrs, traces) = tracing::debug_span!("compile_one loop").in_scope(|| {
            let mut instrs = Vec::with_capacity(PREALLOC_INSTRUCTIONS);
            let mut traces = vec![];
            if debug_mode {
                let mut span_builder =
                    SpanBuilder::<_, &'static str>::new("cycle_tracker".to_string());
                for (ir_instr, trace) in operations {
                    self.compile_one(ir_instr, &mut |item| match item {
                        Ok(instr) => {
                            span_builder.item(instr_name(&instr));
                            instrs.push(instr);
                            #[cfg(feature = "debug")]
                            traces.push(trace.clone());
                        }
                        Err(CompileOneErr::CycleTrackerEnter(name)) => {
                            span_builder.enter(name);
                        }
                        Err(CompileOneErr::CycleTrackerExit) => {
                            span_builder.exit().unwrap();
                        }
                        Err(CompileOneErr::Unsupported(instr)) => {
                            panic!("unsupported instruction: {instr:?}\nbacktrace: {trace:?}")
                        }
                    });
                }
                let cycle_tracker_root_span = span_builder.finish().unwrap();
                for line in cycle_tracker_root_span.lines() {
                    tracing::info!("{}", line);
                }
            } else {
                for (ir_instr, trace) in operations {
                    self.compile_one(ir_instr, &mut |item| match item {
                        Ok(instr) => instrs.push(instr),
                        Err(
                            CompileOneErr::CycleTrackerEnter(_) | CompileOneErr::CycleTrackerExit,
                        ) => (),
                        Err(CompileOneErr::Unsupported(instr)) => {
                            panic!("unsupported instruction: {instr:?}\nbacktrace: {trace:?}")
                        }
                    });
                }
            }
            (instrs, traces)
        });

        // Replace the mults using the address count data gathered in this previous.
        // Exhaustive match for refactoring purposes.
        let total_memory = self.addr_to_mult.len() + self.consts.len();
        let mut backfill = |(mult, addr): (&mut F, &Address<F>)| {
            *mult = self.addr_to_mult.remove(addr.as_usize()).unwrap()
        };
        tracing::debug_span!("backfill mult").in_scope(|| {
            for asm_instr in instrs.iter_mut() {
                match asm_instr {
                    Instruction::BaseAlu(BaseAluInstr {
                        mult,
                        addrs: BaseAluIo { out: ref addr, .. },
                        ..
                    }) => backfill((mult, addr)),
                    Instruction::ExtAlu(ExtAluInstr {
                        mult,
                        addrs: ExtAluIo { out: ref addr, .. },
                        ..
                    }) => backfill((mult, addr)),
                    Instruction::Mem(MemInstr {
                        addrs: MemIo { inner: ref addr },
                        mult,
                        kind: MemAccessKind::Write,
                        ..
                    }) => backfill((mult, addr)),
                    Instruction::Poseidon2(instr) => {
                        let Poseidon2SkinnyInstr {
                            addrs: Poseidon2Io { output: ref addrs, .. },
                            mults,
                        } = instr.as_mut();
                        mults.iter_mut().zip(addrs).for_each(&mut backfill);
                    }
                    Instruction::Poseidon2LinearLayer(instr) => {
                        let Poseidon2LinearLayerInstr {
                            addrs: Poseidon2LinearLayerIo { output: ref addrs, .. },
                            mults,
                            ..
                        } = instr.as_mut();
                        mults.iter_mut().zip(addrs).for_each(&mut backfill);
                    }
                    Instruction::Poseidon2SBox(Poseidon2SBoxInstr {
                        addrs: Poseidon2SBoxIo { output: ref addr, .. },
                        mult,
                        ..
                    }) => backfill((mult, addr)),
                    Instruction::ExtFelt(ExtFeltInstr { addrs, mults, ext2felt }) => {
                        if *ext2felt {
                            mults[1..].iter_mut().zip(&addrs[1..]).for_each(&mut backfill);
                        } else {
                            backfill((&mut mults[0], &addrs[0]));
                        }
                    }
                    Instruction::Select(SelectInstr {
                        addrs: SelectIo { out1: ref addr1, out2: ref addr2, .. },
                        mult1,
                        mult2,
                    }) => {
                        backfill((mult1, addr1));
                        backfill((mult2, addr2));
                    }
                    Instruction::HintBits(HintBitsInstr { output_addrs_mults, .. })
                    | Instruction::Hint(HintInstr { output_addrs_mults, .. }) => {
                        output_addrs_mults
                            .iter_mut()
                            .for_each(|(addr, mult)| backfill((mult, addr)));
                    }
                    Instruction::PrefixSumChecks(instr) => {
                        let PrefixSumChecksInstr {
                            addrs: PrefixSumChecksIo { ref accs, ref field_accs, .. },
                            acc_mults,
                            field_acc_mults,
                        } = instr.as_mut();
                        acc_mults.iter_mut().zip(accs).for_each(&mut backfill);
                        field_acc_mults.iter_mut().zip(field_accs).for_each(&mut backfill);
                    }
                    Instruction::HintExt2Felts(HintExt2FeltsInstr {
                        output_addrs_mults, ..
                    }) => {
                        output_addrs_mults
                            .iter_mut()
                            .for_each(|(addr, mult)| backfill((mult, addr)));
                    }
                    Instruction::HintAddCurve(HintAddCurveInstr {
                        output_x_addrs_mults,
                        output_y_addrs_mults,
                        ..
                    }) => {
                        output_x_addrs_mults
                            .iter_mut()
                            .for_each(|(addr, mult)| backfill((mult, addr)));
                        output_y_addrs_mults
                            .iter_mut()
                            .for_each(|(addr, mult)| backfill((mult, addr)));
                    }
                    // Instructions that do not write to memory.
                    Instruction::Mem(MemInstr { kind: MemAccessKind::Read, .. })
                    | Instruction::CommitPublicValues(_)
                    | Instruction::Print(_) => (),
                }
            }
        });
        debug_assert!(self.addr_to_mult.is_empty());
        // Initialize constants.
        let total_consts = self.consts.len();
        let instrs_consts =
            self.consts.drain().sorted_by_key(|x| x.1 .0 .0).map(|(imm, (addr, mult))| {
                Instruction::Mem(MemInstr {
                    addrs: MemIo { inner: addr },
                    vals: MemIo { inner: imm.as_block() },
                    mult,
                    kind: MemAccessKind::Write,
                })
            });
        tracing::debug!("number of consts to initialize: {}", instrs_consts.len());
        // Reset the other fields.
        self.next_addr = Default::default();
        self.virtual_to_physical.clear();
        // Place constant-initializing instructions at the top.
        let (instructions, traces) = tracing::debug_span!("construct program").in_scope(|| {
            if debug_mode {
                let instrs_all = instrs_consts.chain(instrs);
                let traces_all = std::iter::repeat_n(None, total_consts).chain(traces);
                (instrs_all.collect(), traces_all.collect())
            } else {
                (instrs_consts.chain(instrs).collect(), traces)
            }
        });
        RecursionProgram { instructions, total_memory, traces, shape: None }
    }
}

/// Used for cycle tracking.
const fn instr_name<F>(instr: &Instruction<F>) -> &'static str {
    match instr {
        Instruction::BaseAlu(_) => "BaseAlu",
        Instruction::ExtAlu(_) => "ExtAlu",
        Instruction::Mem(_) => "Mem",
        Instruction::Poseidon2(_) => "Poseidon2",
        Instruction::Poseidon2LinearLayer(_) => "Poseidon2LinearLayer",
        Instruction::Poseidon2SBox(_) => "Poseidon2SBox",
        Instruction::ExtFelt(_) => "ExtFelt",
        Instruction::Select(_) => "Select",
        Instruction::HintBits(_) => "HintBits",
        Instruction::PrefixSumChecks(_) => "PrefixSumChecks",
        Instruction::Print(_) => "Print",
        Instruction::HintExt2Felts(_) => "HintExt2Felts",
        Instruction::Hint(_) => "Hint",
        Instruction::HintAddCurve(_) => "HintAddCurve",
        Instruction::CommitPublicValues(_) => "CommitPublicValues",
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum CompileOneErr<C: Config> {
    Unsupported(DslIr<C>),
    CycleTrackerEnter(String),
    CycleTrackerExit,
}

/// Immediate (i.e. constant) field element.
///
/// Required to distinguish a base and extension field element at the type level,
/// since the IR's instructions do not provide this information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Imm<F, EF> {
    /// Element of the base field `F`.
    F(F),
    /// Element of the extension field `EF`.
    EF(EF),
}

impl<F, EF> Imm<F, EF>
where
    F: FieldAlgebra + Copy,
    EF: FieldExtensionAlgebra<F>,
{
    // Get a `Block` of memory representing this immediate.
    pub fn as_block(&self) -> Block<F> {
        match self {
            Imm::F(f) => Block::from(*f),
            Imm::EF(ef) => ef.as_base_slice().into(),
        }
    }
}

/// Utility functions for various register types.
trait Reg<C: Config> {
    /// Mark the register as to be read from, returning the "physical" address.
    fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F>;

    /// Get the "physical" address of the register, assigning a new address if necessary.
    fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F>;

    /// Mark the register as to be written to, returning the "physical" address.
    fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F>;
}

macro_rules! impl_reg_borrowed {
    ($a:ty) => {
        impl<C, T> Reg<C> for $a
        where
            C: Config,
            T: Reg<C> + ?Sized,
        {
            fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                (**self).read(compiler)
            }

            fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                (**self).read_ghost(compiler)
            }

            fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                (**self).write(compiler)
            }
        }
    };
}

// Allow for more flexibility in arguments.
impl_reg_borrowed!(&T);
impl_reg_borrowed!(&mut T);
impl_reg_borrowed!(Box<T>);

macro_rules! impl_reg_vaddr {
    ($a:ty) => {
        impl<C: Config<F: PrimeField64>> Reg<C> for $a {
            fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                compiler.read_vaddr(self.idx as usize)
            }
            fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                compiler.read_ghost_vaddr(self.idx as usize)
            }
            fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
                compiler.write_fp(self.idx as usize)
            }
        }
    };
}

// These three types wrap a `u32` but they don't share a trait.
impl_reg_vaddr!(Var<C::F>);
impl_reg_vaddr!(Felt<C::F>);
impl_reg_vaddr!(Ext<C::F, C::EF>);

impl<C: Config<F: PrimeField64>> Reg<C> for Imm<C::F, C::EF> {
    fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_const(*self)
    }

    fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_ghost_const(*self)
    }

    fn write(&self, _compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        panic!("cannot write to immediate in register: {self:?}")
    }
}

impl<C: Config<F: PrimeField64>> Reg<C> for Address<C::F> {
    fn read(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_addr(*self);
        *self
    }

    fn read_ghost(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.read_ghost_addr(*self);
        *self
    }

    fn write(&self, compiler: &mut AsmCompiler<C>) -> Address<C::F> {
        compiler.write_addr(*self);
        *self
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, io::BufRead, iter::zip, sync::Arc};

    use p3_field::{Field, PrimeField32};
    use p3_koala_bear::Poseidon2InternalLayerKoalaBear;
    use p3_symmetric::{CryptographicHasher, Permutation};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use slop_challenger::IopCtx;

    use zkm_core_machine::utils::setup_logger;
    use zkm_hypercube::{
        config::{default_fri_config, ZkmGlobalContext},
        prover::{AirProver, ProverSemaphore, ZkmShardProver},
        Machine, ShardVerifier,
    };
    use zkm_recursion_core::{machine::RecursionAir, RecursionProgram, Runtime};
    use zkm_stark::{
        inner_perm, koala_bear_poseidon2::KoalaBearPoseidon2, InnerHash, KoalaBearPoseidon2Inner,
        StarkGenericConfig,
    };

    use crate::circuit::{AsmBuilder, AsmConfig, CircuitV2Builder};

    use super::*;

    type SC = KoalaBearPoseidon2;
    type F = <SC as StarkGenericConfig>::Val;
    type EF = <SC as StarkGenericConfig>::Challenge;

    /// The log2 of the number of rows each stacked-PCS column is grouped into. Mirrors
    /// `zkm_recursion_core::machine::tests::RECURSION_LOG_STACKING_HEIGHT`, kept in sync by
    /// convention.
    const RECURSION_LOG_STACKING_HEIGHT: u32 = 4;

    fn test_operations(operations: TracedVec<DslIr<AsmConfig<F, EF>>>) {
        test_operations_with_runner(operations, |program| {
            let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
                program,
                KoalaBearPoseidon2Inner::new().perm,
            );
            runtime.run().unwrap();
            runtime.record
        });
    }

    fn prove_and_verify_recursion_shard<const DEGREE: usize>(
        machine: Machine<F, RecursionAir<F, DEGREE>>,
        program: Arc<RecursionProgram<F>>,
        record: ExecutionRecord<F>,
    ) {
        let max_log_row_count = zkm_stark::ZKMCoreOpts::recursion().shard_size.ilog2() as usize;
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

    fn test_operations_with_runner(
        operations: TracedVec<DslIr<AsmConfig<F, EF>>>,
        run: impl FnOnce(Arc<RecursionProgram<F>>) -> ExecutionRecord<F>,
    ) {
        let mut compiler = super::AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = Arc::new(compiler.compile(operations));
        let record = run(program.clone());

        // Run with the poseidon2 wide chip.
        //
        // TODO(zkm-hypercube): the skinny (DEGREE=9) machine is deliberately not exercised here
        // -- see the matching TODO on `zkm_recursion_core::machine::tests::
        // run_recursion_test_machines` (task #25) for why it still exceeds
        // `zkm_hypercube::chip::MAX_CONSTRAINT_DEGREE`.
        let wide_machine = RecursionAir::<_, 3>::machine_wide_with_all_chips();
        prove_and_verify_recursion_shard(wide_machine, program, record);
    }

    #[test]
    fn test_poseidon2() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng = StdRng::seed_from_u64(0xCAFEDA7E)
            .sample_iter::<[F; WIDTH], _>(rand::distributions::Standard);
        for _ in 0..1 {
            let input_1: [F; WIDTH] = rng.next().unwrap();
            let output_1 = inner_perm().permute(input_1);

            let input_1_felts = input_1.map(|x| builder.eval(x));
            let output_1_felts = builder.poseidon2_permute_v2(input_1_felts);
            let expected: [Felt<_>; WIDTH] = output_1.map(|x| builder.eval(x));
            for (lhs, rhs) in output_1_felts.into_iter().zip(expected) {
                builder.assert_felt_eq(lhs, rhs);
            }
        }

        test_operations(builder.into_operations());
    }

    #[test]
    fn test_poseidon2_hash() {
        let perm = inner_perm();
        let hasher = InnerHash::new(perm.clone());

        let input: [F; 26] = [
            F::from_canonical_u32(0),
            F::from_canonical_u32(1),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(2),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
            F::from_canonical_u32(3),
        ];
        let expected = hasher.hash_iter(input);
        println!("{expected:?}");

        let mut builder = AsmBuilder::<F, EF>::default();
        let input_felts: [Felt<_>; 26] = input.map(|x| builder.eval(x));
        let result = builder.poseidon2_hash_v2(&input_felts);

        for (actual_f, expected_f) in zip(result, expected) {
            builder.assert_felt_eq(actual_f, expected_f);
        }
    }

    #[test]
    fn test_hint_bit_decomposition() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0xC0FFEE7AB1E).sample_iter::<F, _>(rand::distributions::Standard);
        for _ in 0..100 {
            let input_f = rng.next().unwrap();
            let input = input_f.as_canonical_u32();
            let output = (0..NUM_BITS).map(|i| (input >> i) & 1).collect::<Vec<_>>();

            let input_felt = builder.eval(input_f);
            let output_felts = builder.num2bits_v2_f(input_felt, NUM_BITS);
            let expected: Vec<Felt<_>> =
                output.into_iter().map(|x| builder.eval(F::from_canonical_u32(x))).collect();
            for (lhs, rhs) in output_felts.into_iter().zip(expected) {
                builder.assert_felt_eq(lhs, rhs);
            }
        }
        test_operations(builder.into_operations());
    }

    #[test]
    fn test_print_and_cycle_tracker() {
        const ITERS: usize = 5;

        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();

        let input_fs = StdRng::seed_from_u64(0xC0FFEE7AB1E)
            .sample_iter::<F, _>(rand::distributions::Standard)
            .take(ITERS)
            .collect::<Vec<_>>();

        let input_efs = StdRng::seed_from_u64(0x7EA7AB1E)
            .sample_iter::<[F; 4], _>(rand::distributions::Standard)
            .take(ITERS)
            .collect::<Vec<_>>();

        let mut buf = VecDeque::<u8>::new();

        builder.cycle_tracker_v2_enter("printing felts".to_string());
        for (i, &input_f) in input_fs.iter().enumerate() {
            builder.cycle_tracker_v2_enter(format!("printing felt {i}"));
            let input_felt = builder.eval(input_f);
            builder.print_f(input_felt);
            builder.cycle_tracker_v2_exit();
        }
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("printing exts".to_string());
        for (i, input_block) in input_efs.iter().enumerate() {
            builder.cycle_tracker_v2_enter(format!("printing ext {i}"));
            let input_ext = builder.eval(EF::from_base_slice(input_block).cons());
            builder.print_e(input_ext);
            builder.cycle_tracker_v2_exit();
        }
        builder.cycle_tracker_v2_exit();

        test_operations_with_runner(builder.into_operations(), |program| {
            let mut runtime = Runtime::<F, EF, Poseidon2InternalLayerKoalaBear<16>>::new(
                program,
                KoalaBearPoseidon2Inner::new().perm,
            );
            runtime.debug_stdout = Box::new(&mut buf);
            runtime.run().unwrap();
            runtime.record
        });

        let input_str_fs = input_fs.into_iter().map(|elt| format!("{elt}"));
        let input_str_efs = input_efs.into_iter().map(|elt| format!("{elt:?}"));
        let input_strs = input_str_fs.chain(input_str_efs);

        for (input_str, line) in zip(input_strs, buf.lines()) {
            let line = line.unwrap();
            assert!(line.contains(&input_str));
        }
    }

    #[test]
    fn test_ext2felts() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0x3264).sample_iter::<[F; 4], _>(rand::distributions::Standard);
        let mut random_ext = move || EF::from_base_slice(&rng.next().unwrap());
        for _ in 0..100 {
            let input = random_ext();
            let output: &[F] = input.as_base_slice();

            let input_ext = builder.eval(input.cons());
            let output_felts = builder.ext2felt_v2(input_ext);
            let expected: Vec<Felt<_>> = output.iter().map(|&x| builder.eval(x)).collect();
            for (lhs, rhs) in output_felts.into_iter().zip(expected) {
                builder.assert_felt_eq(lhs, rhs);
            }
        }
        test_operations(builder.into_operations());
    }

    #[test]
    fn test_poseidon2_linear_layer_chip() {
        use zkm_recursion_core::chips::poseidon2_wide::{external_linear_layer, internal_linear_layer};

        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0x1157E).sample_iter::<[F; 16], _>(rand::distributions::Standard);
        for _ in 0..20 {
            let input: [F; 16] = rng.next().unwrap();
            let input_exts: [Ext<F, EF>; 4] =
                core::array::from_fn(|i| builder.eval(EF::from_base_slice(&input[i * 4..i * 4 + 4]).cons()));

            let mut external_expected = input;
            external_linear_layer(&mut external_expected);
            let external_output = builder.poseidon2_external_linear_layer_v2(input_exts);
            for (i, out) in external_output.into_iter().enumerate() {
                let expected: Ext<F, EF> =
                    builder.eval(EF::from_base_slice(&external_expected[i * 4..i * 4 + 4]).cons());
                builder.assert_ext_eq(out, expected);
            }

            let mut internal_expected = input;
            internal_linear_layer(&mut internal_expected);
            let internal_output = builder.poseidon2_internal_linear_layer_v2(input_exts);
            for (i, out) in internal_output.into_iter().enumerate() {
                let expected: Ext<F, EF> =
                    builder.eval(EF::from_base_slice(&internal_expected[i * 4..i * 4 + 4]).cons());
                builder.assert_ext_eq(out, expected);
            }
        }
        test_operations(builder.into_operations());
    }

    #[test]
    fn test_poseidon2_sbox_chip() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0x5B0F).sample_iter::<[F; 4], _>(rand::distributions::Standard);
        for _ in 0..20 {
            let input: [F; 4] = rng.next().unwrap();
            let input_ext: Ext<F, EF> = builder.eval(EF::from_base_slice(&input).cons());
            let cube = |x: F| x * x * x;

            let external_expected = input.map(cube);
            let external_output = builder.poseidon2_external_sbox_v2(input_ext);
            let external_expected_ext: Ext<F, EF> =
                builder.eval(EF::from_base_slice(&external_expected).cons());
            builder.assert_ext_eq(external_output, external_expected_ext);

            let internal_expected = [cube(input[0]), input[1], input[2], input[3]];
            let internal_output = builder.poseidon2_internal_sbox_v2(input_ext);
            let internal_expected_ext: Ext<F, EF> =
                builder.eval(EF::from_base_slice(&internal_expected).cons());
            builder.assert_ext_eq(internal_output, internal_expected_ext);
        }
        test_operations(builder.into_operations());
    }

    #[test]
    fn test_ext_felt_convert_chip() {
        setup_logger();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut rng =
            StdRng::seed_from_u64(0xC0117E27).sample_iter::<[F; 4], _>(rand::distributions::Standard);
        for _ in 0..20 {
            let input: [F; 4] = rng.next().unwrap();
            let input_ext: Ext<F, EF> = builder.eval(EF::from_base_slice(&input).cons());

            let felts = builder.ext2felt_chip_v2(input_ext);
            let expected_felts: Vec<Felt<F>> = input.iter().map(|&x| builder.eval(x)).collect();
            for (lhs, rhs) in felts.into_iter().zip(expected_felts) {
                builder.assert_felt_eq(lhs, rhs);
            }

            let round_trip = builder.felt2ext_chip_v2(felts);
            builder.assert_ext_eq(round_trip, input_ext);
        }
        test_operations(builder.into_operations());
    }

    macro_rules! test_assert_fixture {
        ($assert_felt:ident, $assert_ext:ident, $should_offset:literal) => {
            {
                use std::convert::identity;
                let mut builder = AsmBuilder::<F, EF>::default();
                test_assert_fixture!(builder, identity, F, Felt<_>, 0xDEADBEEF, $assert_felt, $should_offset);
                test_assert_fixture!(builder, EF::cons, EF, Ext<_, _>, 0xABADCAFE, $assert_ext, $should_offset);
                test_operations(builder.into_operations());
            }
        };
        ($builder:ident, $wrap:path, $t:ty, $u:ty, $seed:expr, $assert:ident, $should_offset:expr) => {
            {
                let mut elts = StdRng::seed_from_u64($seed)
                    .sample_iter::<$t, _>(rand::distributions::Standard);
                for _ in 0..100 {
                    let a = elts.next().unwrap();
                    let b = elts.next().unwrap();
                    let c = a + b;
                    let ar: $u = $builder.eval($wrap(a));
                    let br: $u = $builder.eval($wrap(b));
                    let cr: $u = $builder.eval(ar + br);
                    let cm = if $should_offset {
                        c + elts.find(|x| !x.is_zero()).unwrap()
                    } else {
                        c
                    };
                    $builder.$assert(cr, $wrap(cm));
                }
            }
        };
    }

    #[test]
    fn test_assert_eq_noop() {
        test_assert_fixture!(assert_felt_eq, assert_ext_eq, false);
    }

    #[test]
    #[should_panic]
    fn test_assert_eq_panics() {
        test_assert_fixture!(assert_felt_eq, assert_ext_eq, true);
    }

    #[test]
    fn test_assert_ne_noop() {
        test_assert_fixture!(assert_felt_ne, assert_ext_ne, true);
    }

    #[test]
    #[should_panic]
    fn test_assert_ne_panics() {
        test_assert_fixture!(assert_felt_ne, assert_ext_ne, false);
    }
}
