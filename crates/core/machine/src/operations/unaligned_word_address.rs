use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, NUM_REGISTERS,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::word::Word;

use crate::{
    air::ZKMCoreAirBuilder,
    operations::{AddOperation, IsZeroOperation, KoalaBearWordRangeChecker},
};

/// Computes and validates a (possibly unaligned) memory address (`base + offset`), shared by
/// every chip that reads or writes a sub-word or unaligned quantity (`LoadByteChip`/
/// `LoadHalfChip`/`LoadWordUnalignedChip`/`StoreByteChip`/`StoreHalfChip`/
/// `StoreWordUnalignedChip`). Like [`super::WordAddressOperation`], verifies the sum locally via
/// an embedded `AddOperation` (no cross-chip lookup), range-checks it into the field, and checks
/// it's at least `NUM_REGISTERS` -- but instead of asserting word alignment, it witnesses the low
/// 2 bits (`addr_ls_two_bits`) and the aligned address (`addr_aligned`) so callers can mux the
/// accessed byte/halfword within the aligned word.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct UnalignedWordAddressOperation<T: Copy> {
    /// The computed, range-checked memory address (`op_b + op_c`).
    pub add_operation: AddOperation<T>,
    /// Gadget to verify that `addr_word` is within the Koala-Bear field.
    pub addr_word_range_checker: KoalaBearWordRangeChecker<T>,
    /// Used to check that the address is at least `NUM_REGISTERS`, i.e. doesn't alias the
    /// register file.
    pub most_sig_bytes_zero: IsZeroOperation<T>,
    /// The word-aligned address (`addr_word - addr_ls_two_bits`).
    pub addr_aligned: T,
    /// The address's least significant two bits (`addr_word % 4`).
    pub addr_ls_two_bits: T,
    /// Whether `addr_ls_two_bits == 1`.
    pub ls_bits_is_one: T,
    /// Whether `addr_ls_two_bits == 2`.
    pub ls_bits_is_two: T,
    /// Whether `addr_ls_two_bits == 3`.
    pub ls_bits_is_three: T,
}

impl<F: PrimeField32> UnalignedWordAddressOperation<F> {
    /// Populates the columns for `base + offset`, returning the computed (possibly unaligned)
    /// address.
    pub fn populate(&mut self, blu: &mut impl ByteRecord, base: u32, offset: u32) -> u32 {
        let memory_addr = self.add_operation.populate(blu, base, offset);
        self.addr_word_range_checker.populate(memory_addr);

        let addr_ls_two_bits = (memory_addr % 4) as u8;
        let aligned_addr = memory_addr - addr_ls_two_bits as u32;
        self.addr_aligned = F::from_canonical_u32(aligned_addr);
        self.addr_ls_two_bits = F::from_canonical_u8(addr_ls_two_bits);
        self.ls_bits_is_one = F::from_bool(addr_ls_two_bits == 1);
        self.ls_bits_is_two = F::from_bool(addr_ls_two_bits == 2);
        self.ls_bits_is_three = F::from_bool(addr_ls_two_bits == 3);

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: addr_ls_two_bits as u16,
            a2: 0,
            b: memory_addr.to_le_bytes()[0],
            c: 0b11,
        });

        let addr_bytes = memory_addr.to_le_bytes();
        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::U8Range,
            a1: 0,
            a2: 0,
            b: addr_bytes[1],
            c: addr_bytes[2],
        });

        let addr_word = self.add_operation.value;
        self.most_sig_bytes_zero
            .populate_from_field_element(addr_word[1] + addr_word[2] + addr_word[3]);
        if self.most_sig_bytes_zero.result == F::ONE {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: NUM_REGISTERS as u8 - 1,
                c: addr_word[0].as_canonical_u32() as u8,
            });
        }

        memory_addr
    }
}

impl<F: Field> UnalignedWordAddressOperation<F> {
    /// Evaluates the address computation/validation, returning the witnessed `addr_aligned` for
    /// the caller's own `eval_memory_access` on the aligned RAM word. The witnessed
    /// `ls_bits_is_one/two/three` (accessible on `cols`) are for the caller's own byte/halfword
    /// mux; `1 - ls_bits_is_one - ls_bits_is_two - ls_bits_is_three` is `offset_is_zero`.
    pub fn eval<AB: ZKMCoreAirBuilder>(
        builder: &mut AB,
        op_b_val: Word<AB::Var>,
        op_c: Word<AB::Var>,
        cols: UnalignedWordAddressOperation<AB::Var>,
        is_real: AB::Expr,
    ) -> AB::Var {
        AddOperation::<AB::F>::eval(builder, op_b_val, op_c, cols.add_operation, is_real.clone());
        let addr_word = cols.add_operation.value;
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            addr_word,
            cols.addr_word_range_checker,
            is_real.clone(),
        );
        builder.slice_range_check_u8(&addr_word.0[1..3], is_real.clone());

        // `addr_word >= NUM_REGISTERS`: if the most significant three bytes are zero, the least
        // significant byte alone must already clear `NUM_REGISTERS - 1`.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            AB::Expr::from_canonical_u8(NUM_REGISTERS as u8 - 1),
            addr_word[0],
            cols.most_sig_bytes_zero.result,
        );
        builder.when(cols.most_sig_bytes_zero.result).assert_one(is_real.clone());
        IsZeroOperation::<AB::F>::eval(
            builder,
            addr_word[1] + addr_word[2] + addr_word[3],
            cols.most_sig_bytes_zero,
            is_real.clone(),
        );

        // Witness the low 2 bits via the AND table (unlike `WordAddressOperation`, the result
        // isn't forced to 0 -- this address need not be word-aligned).
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            cols.addr_ls_two_bits,
            addr_word[0],
            AB::Expr::from_canonical_u8(0b11),
            is_real.clone(),
        );

        // Assert that `reduce(addr_word) == addr_aligned + addr_ls_two_bits`.
        builder.when(is_real.clone()).assert_eq::<AB::Expr, AB::Expr>(
            cols.addr_aligned.into() + cols.addr_ls_two_bits.into(),
            addr_word.reduce::<AB>(),
        );

        // Assert that the offset flags are boolean and bind them to `addr_ls_two_bits`.
        let offset_is_zero = AB::Expr::one()
            - cols.ls_bits_is_one.into()
            - cols.ls_bits_is_two.into()
            - cols.ls_bits_is_three.into();
        builder.assert_bool(cols.ls_bits_is_one);
        builder.assert_bool(cols.ls_bits_is_two);
        builder.assert_bool(cols.ls_bits_is_three);
        builder.assert_bool(offset_is_zero.clone());

        builder.when(offset_is_zero).assert_zero(cols.addr_ls_two_bits);
        builder.when(cols.ls_bits_is_one).assert_one(cols.addr_ls_two_bits);
        builder.when(cols.ls_bits_is_two).assert_eq(cols.addr_ls_two_bits, AB::Expr::two());
        builder
            .when(cols.ls_bits_is_three)
            .assert_eq(cols.addr_ls_two_bits, AB::Expr::from_canonical_u8(3));

        cols.addr_aligned
    }
}
