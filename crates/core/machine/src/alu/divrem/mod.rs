//! Division and remainder verification.
//!
//! This module implements the verification logic for division and remainder operations. It ensures
//! that for any given inputs b and c and outputs quotient and remainder, the equation
//!
//! b = c * quotient + remainder
//!
//! holds true, while also ensuring that the signs of `b` and `remainder` match.
//!
//! A critical aspect of this implementation is the use of 64-bit arithmetic for result calculation.
//! This choice is driven by the need to make the solution unique: in 32-bit arithmetic,
//! `c * quotient + remainder` could overflow, leading to results that are congruent modulo 2^{32}
//! and thus not uniquely defined. The 64-bit approach avoids this overflow, ensuring that each
//! valid input combination maps to a unique result.
//!
//! Implementation:
//!
//! # Use the multiplication ALU table. result is 64 bits.
//! result = quotient * c.
//!
//! # Add sign-extended remainder to result. Propagate carry to handle overflow within bytes.
//! base = pow(2, 8)
//! carry = 0
//! for i in range(8):
//!     x = result[i] + remainder[i] + carry
//!     result[i] = x % base
//!     carry = x // base
//!
//! # The number represented by c * quotient + remainder in 64 bits must equal b in 32 bits.
//!
//! # Assert the lower 32 bits of result match b.
//! assert result[0..4] == b[0..4]
//!
//! # Assert the upper 32 bits of result match the sign of b.
//! if (b == -2^{31}) and (c == -1):
//!     # This is the only exception as this is the only case where it overflows.
//!     assert result[4..8] == [0, 0, 0, 0]
//! elif b < 0:
//!     assert result[4..8] == [0xff, 0xff, 0xff, 0xff]
//! else:
//!     assert result[4..8] == [0, 0, 0, 0]
//!
//! # Check a = quotient or remainder.
//! assert a == (quotient if opcode == division else remainder)
//!
//! # remainder and b must have the same sign.
//! if remainder < 0:
//!     assert b <= 0
//! if remainder > 0:
//!     assert b >= 0
//!
//! # abs(remainder) < abs(c)
//! if c < 0:
//!    assert c < remainder <= 0
//! elif c > 0:
//!    assert 0 <= remainder < c
//!
//! # Division by zero is undefined per the MIPS spec and is rejected by the executor
//! # (it traps), so an honest trace never contains a div-by-zero event. The AIR enforces
//! # the same to stay in agreement with the executor.
//! assert not is_c_0   # i.e. c != 0 on every real row

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use p3_air::AirBuilder;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use slop_air::{Air, AirBuilderWithPublicValues, BaseAir};
use zkm_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemoryAccessPosition, MemoryRecordEnum},
    get_msb, get_quotient_and_remainder, is_signed_operation, ByteOpcode, ExecutionRecord, Opcode,
    Program,
};

use crate::{memory::MemoryReadWriteCols, CoreChipError};
use zkm_derive::AlignedBorrow;
#[cfg(feature = "picus")]
use zkm_derive::PicusAnnotations;
#[cfg(feature = "picus")]
use zkm_hypercube::air::PicusInfo;
use zkm_hypercube::{air::MachineAir, word::Word};
use zkm_primitives::consts::WORD_SIZE;

use crate::{
    adapter::InstructionCols,
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, CpuState},
    air::{WordAirBuilder, ZKMCoreAirBuilder},
    memory::{MemoryCols, RegisterAccessCols, RegisterWriteAccessCols},
    operations::{AddOperation, IsEqualWordOperation, IsZeroWordOperation, LtOperation, MulOperation},
    utils::next_power_of_two,
};

/// The number of main trace columns for `DivRemChip`.
pub const NUM_DIVREM_COLS: usize = size_of::<DivRemCols<u8>>();

/// The size of a byte in bits.
const BYTE_SIZE: usize = 8;

/// The size of a 64-bit in bytes.
const LONG_WORD_SIZE: usize = 2 * WORD_SIZE;

/// A chip that implements addition for the opcodes DIV/REM.
///
/// Unlike `AddChip`/`MulChip`, no other chip ever emits a synthetic dependency row into
/// `divrem_events`. DivRem's own internal checks (the `c * quotient` product, the `abs`
/// computations, and the `abs(remainder) < max(abs(c), 1)` comparison) are all verified locally
/// via embedded `MulOperation`/`AddOperation`/`LtOperation` copies -- no cross-chip `send_alu`
/// lookups into `MulChip`/`AddChip`/`LtChip` at all. Every row here is therefore a real, retired
/// instruction -- no `is_real_instruction` split is needed, unlike the other migrated ALU chips.
///
/// `op_b`/`op_c` are always registers (MIPS has no DIVI), so they use the cheap
/// [`RegisterAccessCols`] scheme. `op_a`'s write value is `quotient` for DIV/DIVU and `remainder`
/// for MOD/MODU -- a masked mux between two stored columns, degree 2, which can't be sent
/// directly as a lookup value the way `RTypeReader` does (see `RegisterWriteAccessCols`'s doc
/// comment), so `op_a` uses that instead, with the mux separately asserted against its witnessed
/// `value`. DIV/DIVU always decode with `op_a=32` (MIPS's HI/LO-style divide) and always also
/// write HI (the remainder) -- unlike every other `AluX0Chip`-routed opcode, that HI write stays
/// observable even when the LO destination is discarded, so DIV/DIVU can never be routed away and
/// this chip must always fully verify their division. MOD/MODU have no such second register, so
/// their `op_a==0` case *is* routed to `AluX0Chip` (see its doc comment), same reasoning as every
/// other migrated ALU chip; this chip's own `op_a` is therefore always guaranteed non-zero and
/// needs no `op_a_0` masking of its own.
#[derive(Default)]
pub struct DivRemChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "picus", derive(PicusAnnotations))]
#[repr(C)]
pub struct DivRemCols<T: Copy> {
    /// The current shard and clk. Only meaningful when `is_real` is set.
    pub state: CpuState<T>,

    /// The current/next pc, used for instruction lookup table.
    pub pc: T,
    pub next_pc: T,

    /// The register index of `op_a` (written; DIV/DIVU always 32, MOD/MODU any register but
    /// never register 0 -- see this chip's doc comment).
    pub op_a: T,
    pub op_a_access: RegisterWriteAccessCols<T>,
    /// The register index of `op_b` (read).
    pub op_b: T,
    pub op_b_access: RegisterAccessCols<T>,
    /// The register index of `op_c` (read).
    pub op_c: T,
    pub op_c_access: RegisterAccessCols<T>,

    /// Results of dividing `b` by `c`.
    pub quotient: Word<T>,

    /// Remainder when dividing `b` by `c`.
    pub remainder: Word<T>,

    /// `abs(remainder)`, used to check `abs(remainder) < abs(c)`.
    pub abs_remainder: Word<T>,

    /// `abs(c)`, used to check `abs(remainder) < abs(c)`.
    pub abs_c: Word<T>,

    /// `max(abs(c), 1)`, used to check `abs(remainder) < abs(c)`.
    pub max_abs_c_or_1: Word<T>,

    /// Verifies `0 == c + abs_c` (only meaningful when `c_neg`), computed locally (no cross-chip
    /// lookup into `AddChip`).
    pub add_operation_abs_c: AddOperation<T>,

    /// Verifies `0 == remainder + abs_remainder` (only meaningful when `rem_neg`), computed
    /// locally (no cross-chip lookup into `AddChip`).
    pub add_operation_abs_remainder: AddOperation<T>,

    /// The `c * quotient` product, computed locally (no cross-chip lookup into `MulChip`).
    pub mul_operation: MulOperation<T>,

    /// Carry propagated when adding `remainder` by `c * quotient`.
    pub carry: [T; LONG_WORD_SIZE],

    /// Flag to indicate division by 0.
    pub is_c_0: IsZeroWordOperation<T>,

    /// Flag to indicate whether the opcode is DIV.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_div: T,

    /// Flag to indicate whether the opcode is DIVU.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_divu: T,

    /// Flag to indicate whether the opcode is MOD.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_mod: T,

    /// Flag to indicate whether the opcode is MODU.
    #[cfg_attr(feature = "picus", picus(selector))]
    pub is_modu: T,

    /// Flag to indicate whether the division operation overflows.
    ///
    /// Overflow occurs in a specific case of signed 32-bit integer division: when `b` is the
    /// minimum representable value (`-2^31`, the smallest negative number) and `c` is `-1`. In
    /// this case, the division result exceeds the maximum positive value representable by a
    /// 32-bit signed integer.
    pub is_overflow: T,

    /// Flag for whether the value of `b` matches the unique overflow case `b = -2^31` and `c =
    /// -1`.
    pub is_overflow_b: IsEqualWordOperation<T>,

    /// Flag for whether the value of `c` matches the unique overflow case `b = -2^31` and `c =
    /// -1`.
    pub is_overflow_c: IsEqualWordOperation<T>,

    /// The most significant bit of `b`.
    pub b_msb: T,

    /// The most significant bit of remainder.
    pub rem_msb: T,

    /// The most significant bit of `c`.
    pub c_msb: T,

    /// Flag to indicate whether `b` is negative.
    pub b_neg: T,

    /// Flag to indicate whether `rem_neg` is negative.
    pub rem_neg: T,

    /// Flag to indicate whether `c` is negative.
    pub c_neg: T,

    /// Column to modify multiplicity for remainder range check event.
    pub remainder_check_multiplicity: T,

    /// Verifies `abs(remainder) < max(abs(c), 1)`, computed locally (no cross-chip lookup into
    /// `LtChip`).
    pub remainder_check: LtOperation<T>,

    /// Access to hi register
    pub op_hi_access: MemoryReadWriteCols<T>,
}

impl<F: PrimeField32> MachineAir<F> for DivRemChip {
    type Record = ExecutionRecord;

    type Program = Program;

    type Error = CoreChipError;

    fn name(&self) -> String {
        "DivRem".to_string()
    }

    #[cfg(feature = "picus")]
    fn picus_info(&self) -> PicusInfo {
        DivRemCols::<u8>::picus_info()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.divrem_events.len(),
            None,
            <DivRemChip as MachineAir<F>>::name(self).as_str(),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> Result<RowMajorMatrix<F>, Self::Error> {
        // Generate the trace rows for each event.
        let padded_nb_rows = <DivRemChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let nb_rows = input.divrem_events.len();
        let mut rows: Vec<[F; NUM_DIVREM_COLS]> = Vec::with_capacity(padded_nb_rows);
        let divrem_events = input.divrem_events.clone();
        for event in divrem_events.iter() {
            assert!(
                event.opcode == Opcode::DIVU
                    || event.opcode == Opcode::DIV
                    || event.opcode == Opcode::MODU
                    || event.opcode == Opcode::MOD
            );
            let mut row = [F::ZERO; NUM_DIVREM_COLS];
            let cols: &mut DivRemCols<F> = row.as_mut_slice().borrow_mut();

            // Initialize cols with basic operands and flags derived from the current event.
            {
                cols.pc = F::from_canonical_u32(event.pc);
                cols.next_pc = F::from_canonical_u32(event.next_pc);
                cols.is_divu = F::from_bool(event.opcode == Opcode::DIVU);
                cols.is_div = F::from_bool(event.opcode == Opcode::DIV);
                cols.is_modu = F::from_bool(event.opcode == Opcode::MODU);
                cols.is_mod = F::from_bool(event.opcode == Opcode::MOD);
                cols.is_c_0.populate(event.c);

                cols.state.populate(output, event.clk);

                let instruction = input.program.fetch(event.pc);
                cols.op_a = F::from_canonical_u8(instruction.op_a);
                if let Some(record) = event.a_record {
                    cols.op_a_access.populate(record, output);
                }

                cols.op_b = F::from_canonical_u32(instruction.op_b);
                if let Some(record) = event.b_record {
                    cols.op_b_access.populate(record, output);
                }

                cols.op_c = F::from_canonical_u32(instruction.op_c);
                if let Some(record) = event.c_record {
                    cols.op_c_access.populate(record, output);
                }

                if event.opcode == Opcode::DIVU || event.opcode == Opcode::DIV {
                    // DivRem Chip is only used for DIV and DIVU instruction currently.
                    let mut blu_events: Vec<ByteLookupEvent> = vec![];
                    cols.op_hi_access
                        .populate(MemoryRecordEnum::Write(event.hi_record), &mut blu_events);
                    output.add_byte_lookup_events(blu_events);
                }
            }

            let (quotient, remainder) = get_quotient_and_remainder(event.b, event.c, event.opcode);
            cols.quotient = Word::from(quotient);
            cols.remainder = Word::from(remainder);

            // Calculate flags for sign detection.
            {
                cols.rem_msb = F::from_canonical_u8(get_msb(remainder));
                cols.b_msb = F::from_canonical_u8(get_msb(event.b));
                cols.c_msb = F::from_canonical_u8(get_msb(event.c));
                cols.is_overflow_b.populate(event.b, i32::MIN as u32);
                cols.is_overflow_c.populate(event.c, -1i32 as u32);
                let (c_neg, rem_neg, abs_c, abs_remainder) = if is_signed_operation(event.opcode) {
                    let abs_remainder = (remainder as i32).unsigned_abs();
                    let abs_c = (event.c as i32).unsigned_abs();

                    cols.rem_neg = cols.rem_msb;
                    cols.b_neg = cols.b_msb;
                    cols.c_neg = cols.c_msb;
                    cols.is_overflow =
                        F::from_bool(event.b as i32 == i32::MIN && event.c as i32 == -1);
                    cols.abs_remainder = Word::from(abs_remainder);
                    cols.abs_c = Word::from(abs_c);
                    cols.max_abs_c_or_1 = Word::from(u32::max(1, abs_c));
                    (get_msb(event.c) == 1, get_msb(remainder) == 1, abs_c, abs_remainder)
                } else {
                    cols.abs_remainder = cols.remainder;
                    cols.abs_c = Word::from(event.c);
                    cols.max_abs_c_or_1 = Word::from(u32::max(1, event.c));
                    (false, false, event.c, remainder)
                };

                // Verify `abs(remainder) < max(abs(c), 1)`, computed locally (no cross-chip
                // lookup into `LtChip`). `c != 0` is enforced architecturally (division by zero
                // traps in the executor), so this is unconditional for every real row.
                let max_abs_c_or_1 = u32::max(1, abs_c);
                let result = cols.remainder_check.populate(output, Opcode::SLTU, abs_remainder, max_abs_c_or_1);
                debug_assert_eq!(result, 1);

                // `0 == c + abs_c` / `0 == remainder + abs_remainder`, computed locally (no
                // cross-chip lookup into `AddChip`). Only populated (and its byte-range-check
                // dependency events only recorded) when actually negative, matching the AIR's
                // `c_neg`/`rem_neg`-gated `AddOperation::eval` -- populating unconditionally would
                // record BLU events with no matching send, an interaction imbalance.
                if c_neg {
                    cols.add_operation_abs_c.populate(output, event.c, abs_c);
                }
                if rem_neg {
                    cols.add_operation_abs_remainder.populate(output, remainder, abs_remainder);
                }

                // Insert the MSB lookup events.
                {
                    let words = [event.b, event.c, remainder];
                    let mut blu_events: Vec<ByteLookupEvent> = vec![];
                    for word in words.iter() {
                        let most_significant_byte = word.to_le_bytes()[WORD_SIZE - 1];
                        blu_events.push(ByteLookupEvent {
                            opcode: ByteOpcode::MSB,
                            a1: get_msb(*word) as u16,
                            a2: 0,
                            b: most_significant_byte,
                            c: 0,
                        });
                    }
                    output.add_byte_lookup_events(blu_events);
                }
            }

            // Calculate the modified multiplicity
            {
                cols.remainder_check_multiplicity = F::ONE - cols.is_c_0.result;
            }

            // Calculate c * quotient + remainder.
            {
                let (lo, hi) =
                    cols.mul_operation.populate(output, quotient, event.c, is_signed_operation(event.opcode));
                let mut c_times_quotient = [0u8; LONG_WORD_SIZE];
                c_times_quotient[..WORD_SIZE].copy_from_slice(&lo.to_le_bytes());
                c_times_quotient[WORD_SIZE..].copy_from_slice(&hi.to_le_bytes());

                let remainder_bytes = {
                    if is_signed_operation(event.opcode) {
                        ((remainder as i32) as i64).to_le_bytes()
                    } else {
                        (remainder as u64).to_le_bytes()
                    }
                };

                // Add remainder to product.
                let mut carry = [0u32; 8];
                let base = 1 << BYTE_SIZE;
                for i in 0..LONG_WORD_SIZE {
                    let mut x = c_times_quotient[i] as u32 + remainder_bytes[i] as u32;
                    if i > 0 {
                        x += carry[i - 1];
                    }
                    carry[i] = x / base;
                    cols.carry[i] = F::from_canonical_u32(carry[i]);
                }

                // Range check. (`c_times_quotient`'s bytes are already range-checked by
                // `mul_operation.populate` above.)
                {
                    output.add_u8_range_checks(&event.b.to_le_bytes());
                    output.add_u8_range_checks(&event.c.to_le_bytes());
                    output.add_u8_range_checks(&quotient.to_le_bytes());
                    output.add_u8_range_checks(&remainder.to_le_bytes());
                }
            }

            rows.push(row);
        }

        // Pad the trace to a power of two depending on the proof shape in `input`. A padding row
        // is left all-zero: `is_div`/`is_divu`/`is_mod`/`is_modu` (and thus `is_real`, their sum)
        // default to 0, which gates every interaction below to zero multiplicity on its own --
        // unlike the generic `RegisterReader`, this needs no separate "force immediate flags"
        // workaround.
        rows.resize(padded_nb_rows, [F::ZERO; NUM_DIVREM_COLS]);
        debug_assert_eq!(rows.len(), padded_nb_rows);
        let _ = nb_rows;

        // Convert the trace to a row major matrix.
        Ok(RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_DIVREM_COLS))
    }

    fn included(&self, shard: &Self::Record) -> bool {
        !shard.divrem_events.is_empty()
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<F> BaseAir<F> for DivRemChip {
    fn width(&self) -> usize {
        NUM_DIVREM_COLS
    }
}

impl<AB> Air<AB> for DivRemChip
where
    AB: ZKMCoreAirBuilder + AirBuilderWithPublicValues,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &DivRemCols<AB::Var> = (*local).borrow();
        let base = AB::F::from_canonical_u32(1 << 8);
        let one: AB::Expr = AB::F::ONE.into();
        let zero: AB::Expr = AB::F::ZERO.into();

        let is_real = local.is_div + local.is_divu + local.is_mod + local.is_modu;
        let op_b_val = local.op_b_access.prev_value;
        let op_c_val = local.op_c_access.prev_value;

        // Calculate whether b, remainder, and c are negative.
        {
            // Negative if and only if op code is signed & MSB = 1.
            let msb_sign_pairs = [
                (local.b_msb, local.b_neg),
                (local.rem_msb, local.rem_neg),
                (local.c_msb, local.c_neg),
            ];

            for msb_sign_pair in msb_sign_pairs.iter() {
                let msb = msb_sign_pair.0;
                let is_negative = msb_sign_pair.1;
                builder.assert_eq(msb * (local.is_div + local.is_mod), is_negative);
            }
        }

        // Compute c * quotient locally (no cross-chip lookup into `MulChip`). DIV/MOD treat their
        // operands as signed; DIVU/MODU as unsigned.
        let (c_times_quotient_lo, c_times_quotient_hi) = MulOperation::<AB::F>::eval(
            builder,
            local.quotient,
            op_c_val,
            local.mul_operation,
            local.is_div + local.is_mod,
            is_real.clone(),
        );
        let c_times_quotient: [AB::Var; LONG_WORD_SIZE] = [
            c_times_quotient_lo[0],
            c_times_quotient_lo[1],
            c_times_quotient_lo[2],
            c_times_quotient_lo[3],
            c_times_quotient_hi[0],
            c_times_quotient_hi[1],
            c_times_quotient_hi[2],
            c_times_quotient_hi[3],
        ];

        // Calculate is_overflow. is_overflow = is_equal(b, -2^{31}) * is_equal(c, -1) * is_signed
        {
            IsEqualWordOperation::<AB::F>::eval(
                builder,
                op_b_val.map(|x| x.into()),
                Word::from(i32::MIN as u32).map(|x: AB::F| x.into()),
                local.is_overflow_b,
                is_real.clone(),
            );

            IsEqualWordOperation::<AB::F>::eval(
                builder,
                op_c_val.map(|x| x.into()),
                Word::from(-1i32 as u32).map(|x: AB::F| x.into()),
                local.is_overflow_c,
                is_real.clone(),
            );

            builder.assert_eq(
                local.is_overflow,
                local.is_overflow_b.is_diff_zero.result
                    * local.is_overflow_c.is_diff_zero.result
                    * (local.is_div + local.is_mod),
            );
        }

        // Add remainder to product c * quotient, and compare it to b.
        {
            let sign_extension = local.rem_neg * AB::F::from_canonical_u8(u8::MAX);
            let mut c_times_quotient_plus_remainder: Vec<AB::Expr> =
                vec![AB::F::ZERO.into(); LONG_WORD_SIZE];

            // Add remainder to c_times_quotient and propagate carry.
            for i in 0..LONG_WORD_SIZE {
                c_times_quotient_plus_remainder[i] = c_times_quotient[i].into();

                // Add remainder.
                if i < WORD_SIZE {
                    c_times_quotient_plus_remainder[i] =
                        c_times_quotient_plus_remainder[i].clone() + local.remainder[i].into();
                } else {
                    // If rem is negative, add 0xff to the upper 4 bytes.
                    c_times_quotient_plus_remainder[i] =
                        c_times_quotient_plus_remainder[i].clone() + sign_extension.clone();
                }

                // Propagate carry.
                c_times_quotient_plus_remainder[i] =
                    c_times_quotient_plus_remainder[i].clone() - local.carry[i] * base;
                if i > 0 {
                    c_times_quotient_plus_remainder[i] =
                        c_times_quotient_plus_remainder[i].clone() + local.carry[i - 1].into();
                }
            }

            // Compare c_times_quotient_plus_remainder to b by checking each limb.
            for i in 0..LONG_WORD_SIZE {
                if i < WORD_SIZE {
                    // The lower 4 bytes of the result must match the corresponding bytes in b.
                    builder.assert_eq(op_b_val[i], c_times_quotient_plus_remainder[i].clone());
                } else {
                    // The upper 4 bytes must reflect the sign of b in two's complement:
                    // - All 1s (0xff) for negative b.
                    // - All 0s for non-negative b.
                    let not_overflow = one.clone() - local.is_overflow;
                    builder.when(not_overflow.clone()).when(local.b_neg).assert_eq(
                        c_times_quotient_plus_remainder[i].clone(),
                        AB::F::from_canonical_u8(u8::MAX),
                    );
                    builder
                        .when(not_overflow.clone())
                        .when_ne(one.clone(), local.b_neg)
                        .assert_zero(c_times_quotient_plus_remainder[i].clone());

                    // The only exception to the upper-4-byte check is the overflow case.
                    builder
                        .when(local.is_overflow)
                        .assert_zero(c_times_quotient_plus_remainder[i].clone());
                }
            }
        }

        // remainder and b must have the same sign. Due to the intricate nature of sign logic in ZK,
        // we will check a slightly stronger condition:
        //
        // 1. If remainder < 0, then b < 0.
        // 2. If remainder > 0, then b >= 0.
        {
            // A number is 0 if and only if the sum of the 4 limbs equals to 0.
            let mut rem_byte_sum = zero.clone();
            let mut b_byte_sum = zero.clone();
            for i in 0..WORD_SIZE {
                rem_byte_sum = rem_byte_sum.clone() + local.remainder[i].into();
                b_byte_sum = b_byte_sum + op_b_val[i].into();
            }

            // 1. If remainder < 0, then b < 0.
            builder
                .when(local.rem_neg) // rem is negative.
                .assert_one(local.b_neg); // b is negative.

            // 2. If remainder > 0, then b >= 0.
            builder
                .when(rem_byte_sum.clone()) // remainder is nonzero.
                .when(one.clone() - local.rem_neg) // rem is not negative.
                .assert_zero(local.b_neg); // b is not negative.
        }

        // Division by zero is architecturally undefined: the executor traps on it
        // (`ExecutionError::ExceptionOrTrap`, see `execute_alu`), so an honest trace never
        // contains a divrem event with c == 0. Enforce the same here so the AIR rejects
        // div-by-zero rows rather than accepting them with a forced quotient. This keeps the
        // executor and AIR in agreement and removes a path for injecting controlled
        // quotient/remainder values into the trace.
        {
            // Calculate whether c is 0.
            IsZeroWordOperation::<AB::F>::eval(
                builder,
                op_c_val.map(|x| x.into()),
                local.is_c_0,
                is_real.clone(),
            );

            // c must be non-zero on every real divrem row.
            builder.when(is_real.clone()).assert_zero(local.is_c_0.result);
        }

        // Range check remainder. (i.e., |remainder| < |c| when not is_c_0)
        {
            // For each of `c` and `rem`, assert that the absolute value is equal to the original
            // value, if the original value is non-negative or the minimum i32.
            for i in 0..WORD_SIZE {
                builder.when_not(local.c_neg).assert_eq(op_c_val[i], local.abs_c[i]);
                builder
                    .when_not(local.rem_neg)
                    .assert_eq(local.remainder[i], local.abs_remainder[i]);
            }
            // In the case that `c` or `rem` is negative, instead check that their sum is zero,
            // computed locally (no cross-chip lookup into `AddChip`).
            AddOperation::<AB::F>::eval(
                builder,
                op_c_val,
                local.abs_c,
                local.add_operation_abs_c,
                local.c_neg.into(),
            );
            builder.when(local.c_neg).assert_word_zero(local.add_operation_abs_c.value);
            AddOperation::<AB::F>::eval(
                builder,
                local.remainder,
                local.abs_remainder,
                local.add_operation_abs_remainder,
                local.rem_neg.into(),
            );
            builder.when(local.rem_neg).assert_word_zero(local.add_operation_abs_remainder.value);

            // max(abs(c), 1) = abs(c) * (1 - is_c_0) + 1 * is_c_0
            let max_abs_c_or_1: Word<AB::Expr> = {
                let mut v = vec![zero.clone(); WORD_SIZE];

                // Set the least significant byte to 1 if is_c_0 is true.
                v[0] = local.is_c_0.result * one.clone()
                    + (one.clone() - local.is_c_0.result) * local.abs_c[0];

                // Set the remaining bytes to 0 if is_c_0 is true.
                for i in 1..WORD_SIZE {
                    v[i] = (one.clone() - local.is_c_0.result) * local.abs_c[i];
                }
                Word(v.try_into().unwrap_or_else(|_| panic!("Incorrect length")))
            };
            for i in 0..WORD_SIZE {
                builder
                    .when(is_real.clone())
                    .assert_eq(local.max_abs_c_or_1[i], max_abs_c_or_1[i].clone());
            }

            // Handle cases:
            // - If is_real == 0 then remainder_check_multiplicity == 0 is forced.
            // - If is_real == 1 then is_c_0_result must be the expected one, so
            //   remainder_check_multiplicity = (1 - is_c_0_result) * is_real.
            builder.assert_eq(
                (AB::Expr::one() - local.is_c_0.result) * is_real.clone(),
                local.remainder_check_multiplicity,
            );

            // Verify abs(remainder) < max(abs(c), 1), computed locally (no cross-chip lookup into
            // `LtChip`); this is equivalent to abs(remainder) < abs(c) if not division by 0.
            LtOperation::<AB::F>::eval(
                builder,
                local.abs_remainder.map(Into::into),
                local.max_abs_c_or_1.map(Into::into),
                local.remainder_check,
                local.remainder_check_multiplicity.into(),
            );
            builder
                .when(local.remainder_check_multiplicity)
                .assert_one(local.remainder_check.a[0]);
        }

        // Check that the MSBs are correct.
        {
            let msb_pairs = [
                (local.b_msb, op_b_val[WORD_SIZE - 1]),
                (local.c_msb, op_c_val[WORD_SIZE - 1]),
                (local.rem_msb, local.remainder[WORD_SIZE - 1]),
            ];
            let opcode = AB::F::from_canonical_u32(ByteOpcode::MSB as u32);
            for msb_pair in msb_pairs.iter() {
                let msb = msb_pair.0;
                let byte = msb_pair.1;
                builder.send_byte(opcode, msb, byte, zero.clone(), is_real.clone());
            }
        }

        // Range check all the bytes.
        {
            // Constrain operands to byte limbs so extracted standalone modules
            // cannot pick non-byte witness values for word inputs.
            builder.slice_range_check_u8(&op_b_val.0, is_real.clone());
            builder.slice_range_check_u8(&op_c_val.0, is_real.clone());
            builder.slice_range_check_u8(&local.quotient.0, is_real.clone());
            builder.slice_range_check_u8(&local.remainder.0, is_real.clone());

            local.carry.iter().for_each(|carry| {
                builder.assert_bool(*carry);
            });

            // `c_times_quotient`'s bytes are already range-checked by `MulOperation::eval` above.
        }

        // Check that the flags are boolean.
        {
            let bool_flags = [
                local.is_div,
                local.is_divu,
                local.is_mod,
                local.is_modu,
                local.is_overflow,
                local.b_msb,
                local.rem_msb,
                local.c_msb,
                local.b_neg,
                local.rem_neg,
                local.c_neg,
            ];

            for flag in bool_flags.into_iter() {
                builder.assert_bool(flag);
            }
        }

        // Exactly one of the opcode flags must be on.
        builder
            .when(is_real.clone())
            .assert_eq(one.clone(), local.is_divu + local.is_div + local.is_mod + local.is_modu);

        // ---- Real-instruction path: program lookup, state chain, register access. ----
        // No `AddChip`/`MulChip`-style synthetic-row split is needed here: nothing ever
        // produces a synthetic `divrem_events` row (see this chip's doc comment), so `is_real`
        // already means "real instruction".
        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: `opcode` is a degree-1
        // linear combination of the one-hot selectors above, `op_a_0`/`imm_b`/`imm_c` are
        // compile-time constants (this chip only ever sees a non-zero destination and
        // register-register DIV/DIVU/MOD/MODU -- see this chip's doc comment), and `op_b`/`op_c`
        // are zero-extended from their register-index columns.
        let opcode = local.is_div * Opcode::DIV.as_field::<AB::F>()
            + local.is_divu * Opcode::DIVU.as_field::<AB::F>()
            + local.is_mod * Opcode::MOD.as_field::<AB::F>()
            + local.is_modu * Opcode::MODU.as_field::<AB::F>();
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode,
            op_a: local.op_a.into(),
            op_b: Word::extend_var::<AB>(local.op_b),
            op_c: Word::extend_var::<AB>(local.op_c),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::zero(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        // The register write is `quotient` for DIV/DIVU and `remainder` for MOD/MODU -- a masked
        // mux between two stored columns (degree 2), so it needs `RegisterWriteAccessCols`'s own
        // witnessed `value` rather than a direct lookup-value feed (see this chip's doc comment).
        let is_div_or_divu: AB::Expr = local.is_div + local.is_divu;
        let is_mod_or_modu: AB::Expr = local.is_mod + local.is_modu;
        let op_a_computed_value: Word<AB::Expr> = Word(core::array::from_fn(|i| {
            is_div_or_divu.clone() * local.quotient[i].into()
                + is_mod_or_modu.clone() * local.remainder[i].into()
        }));
        let written_value: Word<AB::Expr> = local.op_a_access.value.map(Into::into);
        builder.when(is_real.clone()).assert_word_eq(op_a_computed_value, written_value);

        // Register positions must be read/written in the order C, B, A (see
        // `MemoryAccessPosition`'s doc comment); each gets its own `clk_low` offset, matching the
        // executor's own `rr_traced`/`rw_traced` timestamps for these accesses.
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
            local.op_c.into(),
            &local.op_c_access,
            is_real.clone(),
        );
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            local.op_b.into(),
            &local.op_b_access,
            is_real.clone(),
        );
        builder.eval_register_access_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            local.op_a.into(),
            &local.op_a_access,
            is_real.clone(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.clone());

        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk_high.clone(),
            clk_low.clone(),
            local.pc.into(),
            local.next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            AB::Expr::from_canonical_u32(5),
            is_real.clone(),
        );

        // Write the HI register, the register can only be Register::HI（33）.
        builder.eval_memory_access(
            clk_high,
            clk_low + AB::F::from_canonical_u32(MemoryAccessPosition::HI as u32),
            AB::F::from_canonical_u32(33),
            &local.op_hi_access,
            local.is_div + local.is_divu,
        );
        builder
            .when(local.is_div + local.is_divu)
            .assert_word_eq(local.remainder, *local.op_hi_access.value());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::RowMajorMatrix;
    use zkm_core_executor::{events::CompAluEvent, ExecutionRecord, Instruction, Opcode, Program};

    use super::DivRemChip;
    use zkm_hypercube::air::MachineAir;

    #[test]
    fn generate_trace() {
        // Every `divrem_events` row is a real, retired instruction (see this chip's doc
        // comment), so trace-gen always does a real program lookup -- unlike the other migrated
        // ALU chips' tests, this needs an actual single-instruction `Program`, not `UNUSED_PC`.
        let program = Arc::new(Program {
            instructions: vec![Instruction::new(Opcode::DIVU, 32, 0, 0, false, false)],
            pc_start: 0,
            pc_base: 0,
            ..Default::default()
        });
        let shard = ExecutionRecord {
            program,
            divrem_events: vec![CompAluEvent::new(0, Opcode::DIVU, 2, 17, 3)],
            ..Default::default()
        };
        let chip = DivRemChip::default();
        let trace: RowMajorMatrix<KoalaBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default()).unwrap();
        println!("{:?}", trace.values)
    }
}
