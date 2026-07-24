use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, Opcode, NUM_REGISTERS,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::word::Word;

use crate::{
    air::ZKMCoreAirBuilder,
    operations::{IsZeroOperation, KoalaBearWordRangeChecker},
};

/// Computes and validates a word-aligned memory address (`base + offset`), shared by every chip
/// that reads or writes a whole aligned word (`LoadWordChip`/`StoreWordChip`/`LoadX0Chip`):
/// verifies the sum via the ALU table, range-checks it into the field, checks it's at least
/// `NUM_REGISTERS` (so it can't alias the register file), and checks word alignment.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct WordAddressOperation<T: Copy> {
    /// The computed, unaligned-checked memory address (`op_b + op_c`).
    pub addr_word: Word<T>,
    /// Gadget to verify that `addr_word` is within the Koala-Bear field.
    pub addr_word_range_checker: KoalaBearWordRangeChecker<T>,
    /// Used to check that the address is at least `NUM_REGISTERS`, i.e. doesn't alias the
    /// register file.
    pub most_sig_bytes_zero: IsZeroOperation<T>,
}

impl<F: PrimeField32> WordAddressOperation<F> {
    /// Populates the columns for `base + offset`, returning the computed address.
    pub fn populate(&mut self, blu: &mut impl ByteRecord, base: u32, offset: u32) -> u32 {
        let memory_addr = base.wrapping_add(offset);
        self.addr_word = memory_addr.into();
        self.addr_word_range_checker.populate(memory_addr);
        assert!(memory_addr.is_multiple_of(4), "a real word-aligned memory access must be word-aligned");

        blu.add_byte_lookup_event(ByteLookupEvent {
            opcode: ByteOpcode::AND,
            a1: 0,
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

        self.most_sig_bytes_zero.populate_from_field_element(
            self.addr_word[1] + self.addr_word[2] + self.addr_word[3],
        );
        if self.most_sig_bytes_zero.result == F::ONE {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a1: 1,
                a2: 0,
                b: NUM_REGISTERS as u8 - 1,
                c: self.addr_word[0].as_canonical_u32() as u8,
            });
        }

        memory_addr
    }
}

impl<F: Field> WordAddressOperation<F> {
    /// Evaluates the address computation/validation, returning the witnessed `addr_word` for the
    /// caller's own `eval_memory_access` on the RAM word.
    pub fn eval<AB: ZKMCoreAirBuilder>(
        builder: &mut AB,
        op_b_val: Word<AB::Expr>,
        op_c: Word<AB::Expr>,
        cols: WordAddressOperation<AB::Var>,
        is_real: AB::Expr,
    ) -> Word<AB::Var> {
        builder.send_alu(
            AB::Expr::from_canonical_u32(Opcode::ADD as u32),
            cols.addr_word,
            op_b_val,
            op_c,
            is_real.clone(),
        );
        KoalaBearWordRangeChecker::<AB::F>::range_check(
            builder,
            cols.addr_word,
            cols.addr_word_range_checker,
            is_real.clone(),
        );
        builder.slice_range_check_u8(&cols.addr_word.0[1..3], is_real.clone());

        // `addr_word >= NUM_REGISTERS`: if the most significant three bytes are zero, the least
        // significant byte alone must already clear `NUM_REGISTERS - 1`.
        builder.send_byte(
            ByteOpcode::LTU.as_field::<AB::F>(),
            AB::Expr::one(),
            AB::Expr::from_canonical_u8(NUM_REGISTERS as u8 - 1),
            cols.addr_word[0],
            cols.most_sig_bytes_zero.result,
        );
        builder.when(cols.most_sig_bytes_zero.result).assert_one(is_real.clone());
        IsZeroOperation::<AB::F>::eval(
            builder,
            cols.addr_word[1] + cols.addr_word[2] + cols.addr_word[3],
            cols.most_sig_bytes_zero,
            is_real.clone(),
        );

        // A real word-aligned access's address is always word-aligned: `addr_word[0] & 0b11 == 0`.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            AB::Expr::zero(),
            cols.addr_word[0],
            AB::Expr::from_canonical_u8(0b11),
            is_real,
        );

        cols.addr_word
    }
}
