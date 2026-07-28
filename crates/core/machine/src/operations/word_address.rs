use p3_air::AirBuilder;
use p3_field::{Field, FieldAlgebra, PrimeField32};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, NUM_REGISTERS,
};
use zkm_derive::AlignedBorrow;
use zkm_hypercube::{air::BaseAirBuilder, word::Word};

use crate::{
    air::ZKMCoreAirBuilder,
    operations::{AddOperation, IsZeroOperation},
};

/// The KoalaBear modulus's top 16 bits (`P = TOP_LIMB * 2^16 + 1`).
const TOP_LIMB: u16 = 0x7F00;

/// Computes and validates a word-aligned memory address (`base + offset`), shared by every chip
/// that reads or writes a whole aligned word (`LoadWordChip`/`StoreWordChip`/`LoadX0Chip`):
/// verifies the sum locally via an embedded `AddOperation` (no cross-chip lookup), range-checks
/// it into the field, checks it's at least `NUM_REGISTERS` (so it can't alias the register file),
/// and checks word alignment.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct WordAddressOperation<T: Copy> {
    /// The computed, unaligned-checked memory address (`op_b + op_c`).
    pub add_operation: AddOperation<T>,
    /// 1 iff `addr_word`'s top 16 bits are strictly less than `TOP_LIMB`, which (together with a
    /// low-16-bits-zero check in the -- for a real address, unreachable -- equality case) proves
    /// `addr_word` reduces to a canonical value below the Koala-Bear modulus. See `eval`'s doc
    /// comment for why this single comparison suffices in place of a full byte decomposition.
    pub addr_high16_lt_top_limb: T,
    /// Used to check that the address is at least `NUM_REGISTERS`, i.e. doesn't alias the
    /// register file.
    pub most_sig_bytes_zero: IsZeroOperation<T>,
}

impl<F: PrimeField32> WordAddressOperation<F> {
    /// Populates the columns for `base + offset`, returning the computed address.
    pub fn populate(&mut self, blu: &mut impl ByteRecord, base: u32, offset: u32) -> u32 {
        let memory_addr = self.add_operation.populate(blu, base, offset);
        assert!(memory_addr.is_multiple_of(4), "a real word-aligned memory access must be word-aligned");

        let high16 = (memory_addr >> 16) as u16;
        let lt = high16 < TOP_LIMB;
        self.addr_high16_lt_top_limb = F::from_bool(lt);
        if lt {
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U16Range,
                a1: TOP_LIMB - 1 - high16,
                a2: 0,
                b: 0,
                c: 0,
            });
        } else {
            debug_assert_eq!(high16, TOP_LIMB, "a real memory address can't reach the modulus");
            debug_assert_eq!(memory_addr & 0xFFFF, 0);
        }

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

impl<F: Field> WordAddressOperation<F> {
    /// Evaluates the address computation/validation, returning the witnessed `addr_word` for the
    /// caller's own `eval_memory_access` on the RAM word.
    pub fn eval<AB: ZKMCoreAirBuilder>(
        builder: &mut AB,
        op_b_val: Word<AB::Var>,
        op_c: Word<AB::Var>,
        cols: WordAddressOperation<AB::Var>,
        is_real: AB::Expr,
    ) -> Word<AB::Var> {
        AddOperation::<AB::F>::eval(builder, op_b_val, op_c, cols.add_operation, is_real.clone());
        let addr_word = cols.add_operation.value;

        // Range-check `addr_word` into the field: `reduce(addr_word) < P` where
        // `P = TOP_LIMB * 2^16 + 1` is the Koala-Bear modulus. Grouping the top two bytes into
        // one 16-bit `high16` turns this into a single less-than-`TOP_LIMB` check instead of a
        // full most-significant-byte bit decomposition.
        let high16 = addr_word[2].into() + addr_word[3].into() * AB::Expr::from_canonical_u32(256);
        let low16 = addr_word[0].into() + addr_word[1].into() * AB::Expr::from_canonical_u32(256);
        let lt = cols.addr_high16_lt_top_limb;
        builder.when(is_real.clone()).assert_bool(lt);

        // `lt == 1`: prove `high16 < TOP_LIMB` via one range-check on the slack -- if a cheating
        // prover set `high16 >= TOP_LIMB`, the slack underflows in the field to a value far
        // outside `[0, 2^16)`, which has no matching row in the (shared, already-paid-for)
        // `U16Range` table. `lt` alone (not `is_real * lt`) is the multiplicity, matching the
        // `most_sig_bytes_zero`-gated send below; the safety constraint right after forces
        // `lt == 0` on padding rows so it can't be exploited for a free lookup there.
        builder.send_byte(
            ByteOpcode::U16Range.as_field::<AB::F>(),
            AB::Expr::from_canonical_u32(u32::from(TOP_LIMB) - 1) - high16.clone(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            lt,
        );
        builder.when(lt).assert_one(is_real.clone());

        // `lt == 0`: the only way `high16` isn't `< TOP_LIMB` while `addr_word` still reduces
        // below `P` is `high16 == TOP_LIMB` exactly, which further requires the low 16 bits to be
        // exactly 0 (never true for a real, `MAX_MEMORY`-bounded address).
        builder
            .when(is_real.clone())
            .when_not(lt)
            .assert_eq(high16, AB::Expr::from_canonical_u32(u32::from(TOP_LIMB)));
        builder.when(is_real.clone()).when_not(lt).assert_zero(low16);

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

        // A real word-aligned access's address is always word-aligned: `addr_word[0] & 0b11 == 0`.
        builder.send_byte(
            ByteOpcode::AND.as_field::<AB::F>(),
            AB::Expr::zero(),
            addr_word[0],
            AB::Expr::from_canonical_u8(0b11),
            is_real,
        );

        addr_word
    }
}
