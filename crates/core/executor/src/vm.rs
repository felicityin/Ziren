//! Pure-compute cores shared by `MinimalExecutor` and the oracle-driven `CoreVM`, plus `CoreVM`
//! itself, the shared replay engine `SplicingVM`/`TracingVM` are built on.
//!
//! The pure functions in the first half of this file are each a *pure* function of already-known
//! operand values -- no register file, no memory, no `Executor`/`CoreVM` state. This is the
//! boundary between "recomputed live in every pass" (ALU/branch/jump math, HI/LO writes) and
//! "must be logged in the oracle trace" (RAM contents, syscall/precompile writes, hint lengths).
//! Every function is transcribed directly from the corresponding private `Executor::execute_*`
//! method in `executor.rs` (cited in each doc comment) and is covered by a differential test in
//! the `tests` module below that drives the *real* `Executor` through the public API
//! (`Executor::new`/`run`/`register`) over randomized operands and asserts byte-identical results
//! -- this is the only way to compare against `executor.rs`'s private methods without weakening
//! their visibility.
//!
//! `CoreVM` has no backing RAM at all: every RAM access is answered by popping the next entry off
//! a [`crate::trace::MinimalTrace`]'s oracle log, in exactly the order `MinimalExecutor` produced
//! it. Registers stay plain live state (never oracle-logged); only the RAM-log cursor is threaded
//! through.

#![allow(dead_code)]

use std::sync::Arc;

use crate::{
    events::MemoryAccessPosition,
    opcode::Opcode,
    register::{Register, NUM_REGISTERS},
    syscalls::SyscallCode,
    trace::{MemReads, MemValue, MinimalTrace},
    ExecutionError, Instruction, Program, CORE_SHARD_CLK_LIMIT,
};
use num::{BigUint, Integer};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_koala_bear::KoalaBear;
use p3_symmetric::Permutation;
use typenum::Unsigned;
use zkm_curves::{
    curve25519_dalek::CompressedEdwardsY,
    edwards::{
        ed25519::{decompress as ed25519_decompress_point, Ed25519},
        WORDS_FIELD_ELEMENT,
    },
    params::{NumLimbs, NumWords},
    weierstrass::{
        bls12_381::{bls12381_decompress, Bls12381, Bls12381BaseField},
        bn254::{Bn254, Bn254BaseField},
        secp256k1::{secp256k1_decompress, Secp256k1},
        secp256r1::{secp256r1_decompress, Secp256r1},
        FpOpField,
    },
    AffinePoint, CurveError, CurveType, EllipticCurve, COMPRESSED_POINT_BYTES,
    NUM_BYTES_FIELD_ELEMENT,
};
use zkm_primitives::{
    consts::fd::{FD_HINT, FD_PUBLIC_VALUES, FD_STDERR, FD_STDIN, FD_STDOUT},
    poseidon2_init,
};

use crate::events::FieldOperation;

/// Number of u64 lanes in a Keccak-f\[1600\] state.
pub(crate) const KECCAK_STATE_SIZE_U64S: usize = 25;
/// Rate of the general (fixed-parameter) Keccak sponge precompile, in u64 lanes per block.
pub(crate) const KECCAK_GENERAL_BLOCK_SIZE_U64S: usize = 18;
/// Number of u64 lanes read back out as the sponge's output.
pub(crate) const KECCAK_GENERAL_OUTPUT_U64S: usize = 8;

/// ALU family (`Executor::execute_alu`, `executor.rs`): every opcode reachable via
/// `Instruction::is_alu_instruction()`. Returns `(a, hi)` exactly as the source's inner match
/// does; the caller decides (via `Opcode::is_use_lo_hi_alu()`) whether `hi` is meaningful or `a`
/// alone is the result.
///
/// # Errors
///
/// Returns [`ExecutionError::ExceptionOrTrap`] for DIV/DIVU/MOD/MODU by zero, mirroring
/// `execute_alu`'s guard (checked *before* the source's own match, same as here).
pub(crate) fn alu_compute(opcode: Opcode, b: u32, c: u32) -> Result<(u32, u32), ExecutionError> {
    if matches!(opcode, Opcode::DIV | Opcode::DIVU | Opcode::MOD | Opcode::MODU) && c == 0 {
        return Err(ExecutionError::ExceptionOrTrap());
    }

    Ok(match opcode {
        Opcode::ADD => (b.overflowing_add(c).0, 0),
        Opcode::SUB => (b.overflowing_sub(c).0, 0),
        Opcode::SLL => (b << (c & 0x1f), 0),
        Opcode::SRL => (b >> (c & 0x1F), 0),
        Opcode::SRA => {
            let sin = b as i32;
            let sout = sin >> (c & 0x1f);
            (sout as u32, 0)
        }
        Opcode::ROR => {
            let sin = (b as u64) + ((b as u64) << 32);
            let sout = sin >> (c & 0x1f);
            (sout as u32, 0)
        }
        Opcode::MUL => (b.overflowing_mul(c).0, 0),
        Opcode::SLTU => {
            if b < c {
                (1, 0)
            } else {
                (0, 0)
            }
        }
        Opcode::SLT => {
            if (b as i32) < (c as i32) {
                (1, 0)
            } else {
                (0, 0)
            }
        }
        Opcode::MULT => {
            let out = (((b as i32) as i64) * ((c as i32) as i64)) as u64;
            (out as u32, (out >> 32) as u32)
        }
        Opcode::MULTU => {
            let out = b as u64 * c as u64;
            (out as u32, (out >> 32) as u32)
        }
        Opcode::DIV => (((b as i32) / (c as i32)) as u32, ((b as i32) % (c as i32)) as u32),
        Opcode::DIVU => (b / c, b % c),
        Opcode::MOD => (((b as i32) % (c as i32)) as u32, 0),
        Opcode::MODU => (b % c, 0),
        Opcode::AND => (b & c, 0),
        Opcode::OR => (b | c, 0),
        Opcode::XOR => (b ^ c, 0),
        Opcode::NOR => (!(b | c), 0),
        Opcode::CLZ => (b.leading_zeros(), 0),
        Opcode::CLO => (b.leading_ones(), 0),
        _ => unreachable!("not an ALU opcode: {opcode:?}"),
    })
}

/// Branch family (`Executor::execute_branch`, `executor.rs`): whether the branch is taken.
#[must_use]
pub(crate) fn branch_taken(opcode: Opcode, src1: u32, src2: u32) -> bool {
    match opcode {
        Opcode::BEQ => src1 == src2,
        Opcode::BNE => src1 != src2,
        Opcode::BGEZ => (src1 as i32) >= 0,
        Opcode::BLEZ => (src1 as i32) <= 0,
        Opcode::BGTZ => (src1 as i32) > 0,
        Opcode::BLTZ => (src1 as i32) < 0,
        _ => unreachable!("not a branch opcode: {opcode:?}"),
    }
}

/// Branch family: the taken-branch target (`Executor::execute_branch`'s
/// `next_next_pc = offset.wrapping_add(next_pc)` assignment).
#[must_use]
pub(crate) fn branch_target(next_pc: u32, offset: u32) -> u32 {
    offset.wrapping_add(next_pc)
}

/// Jump family 1/3, JR/JALR (`Executor::execute_jump`): `target_pc` is a live register value
/// (read by the caller); returns `(return_pc, target_pc)`.
#[must_use]
pub(crate) fn jump_jr_result(next_pc: u32, target_pc: u32) -> (u32, u32) {
    (next_pc.wrapping_add(4), target_pc)
}

/// Jump family 2/3, J/JAL (`Executor::execute_jumpi`): `target_pc` is the instruction's own
/// encoded immediate; returns `(return_pc, target_pc)`.
#[must_use]
pub(crate) fn jump_jumpi_result(next_pc: u32, target_pc: u32) -> (u32, u32) {
    (next_pc.wrapping_add(4), target_pc)
}

/// Jump family 3/3, BAL (`Executor::execute_jump_direct`): `target_pc = offset.wrapping_add(next_pc)`;
/// returns `(return_pc, target_pc)`.
#[must_use]
pub(crate) fn jump_direct_result(next_pc: u32, offset: u32) -> (u32, u32) {
    (next_pc.wrapping_add(4), offset.wrapping_add(next_pc))
}

/// Condmov family (`Executor::execute_condmov`): MEQ/MNE. `prev_a` is `rd`'s pre-instruction
/// value (the no-op case's result).
#[must_use]
pub(crate) fn condmov_result(opcode: Opcode, prev_a: u32, b: u32, c: u32) -> u32 {
    let mov = match opcode {
        Opcode::MEQ => c == 0,
        Opcode::MNE => c != 0,
        _ => unreachable!("not a condmov opcode: {opcode:?}"),
    };
    if mov {
        b
    } else {
        prev_a
    }
}

/// WSBH (`Executor::execute_wsbh`) -- dispatched via the "misc" execute branch in `executor.rs`
/// but its event is emitted into `movcond_events` alongside MEQ/MNE (see `emit_misc_event`), so it
/// is grouped here with the condmov family rather than the "true" misc family below.
#[must_use]
pub(crate) fn wsbh(b: u32) -> u32 {
    (((b >> 16) & 0xFF) << 24) | (((b >> 24) & 0xFF) << 16) | ((b & 0xFF) << 8) | ((b >> 8) & 0xFF)
}

/// Misc family 1/5, SEXT (`Executor::execute_sext`).
#[must_use]
pub(crate) fn sext(b: u32, c: u32) -> u32 {
    if c > 0 {
        (b & 0xffff) as i16 as i32 as u32
    } else {
        (b & 0xff) as i8 as i32 as u32
    }
}

/// Misc family 2/5, EXT (`Executor::execute_ext`).
///
/// # Errors
///
/// Returns [`ExecutionError::ExceptionOrTrap`] when `lsb + msbd >= 32` (undefined encoding),
/// mirroring the source's guard.
pub(crate) fn ext(b: u32, c: u32) -> Result<u32, ExecutionError> {
    let msbd = c >> 5;
    let lsb = c & 0x1f;
    if msbd + lsb >= 32 {
        return Err(ExecutionError::ExceptionOrTrap());
    }
    let mask_msb = if msbd + lsb + 1 == 32 { 0xFFFF_FFFF } else { (1u32 << (msbd + lsb + 1)) - 1 };
    Ok((b & mask_msb) >> lsb)
}

/// Misc family 3/5, INS (`Executor::execute_ins`). `a` is `rd`'s pre-instruction value (the
/// bits outside the inserted field are preserved from it).
///
/// # Errors
///
/// Returns [`ExecutionError::ExceptionOrTrap`] when `msb < lsb` (undefined encoding), mirroring
/// the source's guard.
pub(crate) fn ins(a: u32, b: u32, c: u32) -> Result<u32, ExecutionError> {
    let msb = c >> 5;
    let lsb = c & 0x1f;
    if msb < lsb {
        return Err(ExecutionError::ExceptionOrTrap());
    }
    let mask = if msb - lsb + 1 == 32 { 0xFFFF_FFFF } else { (1u32 << (msb - lsb + 1)) - 1 };
    let mask_field = mask << lsb;
    Ok((a & !mask_field) | ((b << lsb) & mask_field))
}

/// Misc family 4/5, TEQ (`Executor::execute_teq`).
///
/// # Errors
///
/// Returns [`ExecutionError::ExceptionOrTrap`] when `src1 == src2` (the trap condition itself).
pub(crate) fn teq(src1: u32, src2: u32) -> Result<(), ExecutionError> {
    if src1 == src2 {
        return Err(ExecutionError::ExceptionOrTrap());
    }
    Ok(())
}

/// Misc family 5/5, the MADD/MADDU/MSUB/MSUBU quartet (`Executor::execute_{maddu,msubu,madd,msub}`).
/// `lo`/`hi` are the pre-instruction values of registers LO/HI (the running accumulator);
/// returns the new `(lo, hi)` pair.
#[must_use]
pub(crate) fn maddu(b: u32, c: u32, lo: u32, hi: u32) -> (u32, u32) {
    let multiply = b as u64 * c as u64;
    let addend = ((hi as u64) << 32) + lo as u64;
    let out = multiply.wrapping_add(addend);
    (out as u32, (out >> 32) as u32)
}

/// See [`maddu`].
#[must_use]
pub(crate) fn msubu(b: u32, c: u32, lo: u32, hi: u32) -> (u32, u32) {
    let multiply = b as u64 * c as u64;
    let addend = ((hi as u64) << 32) + lo as u64;
    let out = addend.wrapping_sub(multiply);
    (out as u32, (out >> 32) as u32)
}

/// See [`maddu`]. Signed variant (`MADD`).
#[must_use]
pub(crate) fn madd(b: u32, c: u32, lo: u32, hi: u32) -> (u32, u32) {
    let multiply = (b as i32 as i64) * (c as i32 as i64);
    let addend = ((hi as u64) << 32) + lo as u64;
    let out = multiply.wrapping_add(addend as i64) as u64;
    (out as u32, (out >> 32) as u32)
}

/// See [`maddu`]. Signed variant (`MSUB`).
#[must_use]
pub(crate) fn msub(b: u32, c: u32, lo: u32, hi: u32) -> (u32, u32) {
    let multiply = (b as i32 as i64) * (c as i32 as i64);
    let addend = ((hi as u64) << 32) + lo as u64;
    let out = (addend as i64).wrapping_sub(multiply) as u64;
    (out as u32, (out >> 32) as u32)
}

/// Pure function of `(clk, max_syscall_cycles)` -- shared by `MinimalExecutor` and `CoreVM` so
/// they can never drift apart on it. Must budget for an instruction's *full* clk advance
/// (`5 + num_extra_cycles`, up to `max_syscall_cycles` in the worst case), not just the widest
/// single `MemoryAccessPosition` offset -- mirrors `Executor::bump_clk_high_if_need` exactly. An
/// earlier version of this budget accounted only for `MemoryAccessPosition::HI`, which
/// under-budgeted for syscalls and could silently break the `LookupKind::State` interaction chain
/// at a `clk_high` boundary (fixed by `7df6fd1d`).
#[must_use]
pub(crate) fn bump_clk_high_if_need(clk: u64, max_syscall_cycles: u32) -> u64 {
    let window_start = clk & !(CORE_SHARD_CLK_LIMIT - 1);
    let window_remaining = window_start + CORE_SHARD_CLK_LIMIT - clk;
    let max_total_advance = MemoryAccessPosition::HI as u64 + 1 + u64::from(max_syscall_cycles);
    if window_remaining > max_total_advance {
        clk
    } else {
        window_start + CORE_SHARD_CLK_LIMIT
    }
}

/// SHA-256 round constants (`syscalls/precompiles/sha256/compress.rs::SHA_COMPRESS_K`).
pub(crate) const SHA_COMPRESS_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 compress, given the 8 already-read `h` words and 64 already-read `w` words. Pure
/// function of its inputs -- mirrors `Sha256CompressSyscall::execute`'s compute loop
/// (`syscalls/precompiles/sha256/compress.rs`) exactly, split from the memory accesses that
/// gather `h`/`w` so it can be shared between a real backing store and an oracle replay.
#[must_use]
pub(crate) fn sha256_compress(h: [u32; 8], w: &[u32; 64]) -> [u32; 8] {
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] =
        [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]];
    for (i, &w_i) in w.iter().enumerate() {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp1 =
            hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA_COMPRESS_K[i]).wrapping_add(w_i);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }
    let v = [a, b, c, d, e, f, g, hh];
    std::array::from_fn(|i| h[i].wrapping_add(v[i]))
}

/// SHA-256 extend, one word. Pure function of the 4 already-read predecessor words -- mirrors
/// `Sha256ExtendSyscall::execute`'s per-`w[i]` computation (`syscalls/precompiles/sha256/extend.rs`)
/// exactly.
#[must_use]
pub(crate) fn sha256_extend_word(
    w_i_minus_15: u32,
    w_i_minus_2: u32,
    w_i_minus_16: u32,
    w_i_minus_7: u32,
) -> u32 {
    let s0 = w_i_minus_15.rotate_right(7) ^ w_i_minus_15.rotate_right(18) ^ (w_i_minus_15 >> 3);
    let s1 = w_i_minus_2.rotate_right(17) ^ w_i_minus_2.rotate_right(19) ^ (w_i_minus_2 >> 10);
    s1.wrapping_add(w_i_minus_16).wrapping_add(s0).wrapping_add(w_i_minus_7)
}

/// XORs one rate-sized block into a Keccak sponge state, in place -- mirrors
/// `KeccakSpongeSyscall::execute`'s per-block XOR step (`syscalls/precompiles/keccak/sponge.rs`)
/// exactly. The caller applies the Keccak-f\[1600\] permutation (`keccakf`, re-exported alongside
/// this function) afterwards; kept as a separate step (not folded into one "absorb" call) because
/// `TracingVM` needs the post-XOR, pre-permutation state as part of its event.
pub(crate) fn keccak_xor_block(state: &mut [u64; KECCAK_STATE_SIZE_U64S], block: &[u64]) {
    for (i, value) in block.iter().enumerate() {
        state[i] ^= *value;
    }
}

pub(crate) use tiny_keccak::keccakf;

/// Number of u32 words in a curve point's `(x, y)` representation for `E`.
pub(crate) fn ec_num_words<E: EllipticCurve>() -> usize {
    <E::BaseField as NumWords>::WordsCurvePoint::USIZE
}

/// Adds two curve points, given as little-endian word slices -- mirrors `create_ec_add_event`'s
/// compute step (`events/precompiles/ec.rs`) exactly, split from the memory accesses that gather
/// `p`/`q`.
pub(crate) fn ec_add<E: EllipticCurve>(p: &[u32], q: &[u32]) -> Vec<u32> {
    (AffinePoint::<E>::from_words_le(p) + AffinePoint::<E>::from_words_le(q)).to_words_le()
}

/// Doubles a curve point, given as a little-endian word slice -- mirrors
/// `create_ec_double_event`'s compute step exactly.
pub(crate) fn ec_double<E: EllipticCurve>(p: &[u32]) -> Vec<u32> {
    E::ec_double(&AffinePoint::<E>::from_words_le(p)).to_words_le()
}

/// Number of u32 words in a base-field element for `E`.
pub(crate) fn ec_num_limb_words<E: EllipticCurve>() -> usize {
    <E::BaseField as NumLimbs>::Limbs::USIZE / 4
}

/// Decompresses a point's `y` coordinate from its `x` coordinate (big-endian bytes) and sign bit
/// -- mirrors `create_ec_decompress_event`'s compute step exactly, split from the memory accesses
/// that gather `x_bytes_be`. Returns the `y` coordinate as little-endian bytes, padded to
/// `E::BaseField`'s limb width.
///
/// # Errors
///
/// Returns [`CurveError`] if `x_bytes_be`/`sign_bit` don't decode to a valid point on the curve,
/// or if `E` isn't one of the curves with a known decompression routine.
pub(crate) fn ec_decompress<E: EllipticCurve>(
    x_bytes_be: &[u8],
    sign_bit: u32,
) -> Result<Vec<u8>, CurveError> {
    let decompress_fn = match E::CURVE_TYPE {
        CurveType::Secp256k1 => secp256k1_decompress::<E>,
        CurveType::Secp256r1 => secp256r1_decompress::<E>,
        CurveType::Bls12381 => bls12381_decompress::<E>,
        _ => return Err(CurveError::UnsupportedCurve(E::CURVE_TYPE.to_string())),
    };
    let computed_point = decompress_fn(x_bytes_be, sign_bit)?;
    let num_limbs = <E::BaseField as NumLimbs>::Limbs::USIZE;
    let mut y_bytes = computed_point.y.to_bytes_le();
    y_bytes.resize(num_limbs, 0u8);
    Ok(y_bytes)
}

/// Decompresses an Ed25519 point's `x` coordinate from its compressed `y` (already has the sign
/// bit re-inserted at bit 255 by the caller) -- mirrors `EdwardsDecompressSyscall::execute`'s
/// compute step (`syscalls/precompiles/edwards/decompress.rs`) exactly, split from the memory
/// accesses that gather `y_bytes`.
pub(crate) fn ed25519_decompress(
    y_bytes: [u8; COMPRESSED_POINT_BYTES],
    sign: u32,
) -> Result<[u8; NUM_BYTES_FIELD_ELEMENT], CurveError> {
    let mut compressed_edwards_y = y_bytes;
    let last = compressed_edwards_y.len() - 1;
    compressed_edwards_y[last] &= 0b0111_1111;
    compressed_edwards_y[last] |= (sign as u8) << 7;

    let decompressed = ed25519_decompress_point(&CompressedEdwardsY(compressed_edwards_y))?;
    let mut decompressed_x_bytes = decompressed.x.to_bytes_le();
    decompressed_x_bytes.resize(NUM_BYTES_FIELD_ELEMENT, 0u8);
    Ok(decompressed_x_bytes.try_into().unwrap())
}

/// Number of u32 words in one `P`-field element.
pub(crate) fn fp_num_words<P: FpOpField>() -> usize {
    <P as NumWords>::WordsFieldElement::USIZE
}

/// `x op y mod P::MODULUS`, given as little-endian word slices -- mirrors `FpOpSyscall::execute`'s
/// compute step (`syscalls/precompiles/fptower/fp.rs`) exactly, split from the memory accesses
/// that gather `x`/`y`. Only `Add`/`Sub`/`Mul` are valid; mirrors the source's own restriction.
pub(crate) fn fp_op<P: FpOpField>(x: &[u32], y: &[u32], op: FieldOperation) -> Vec<u32> {
    let num_words = fp_num_words::<P>();
    let modulus = &BigUint::from_bytes_le(P::MODULUS);
    let a = BigUint::from_slice(x) % modulus;
    let b = BigUint::from_slice(y) % modulus;
    let result = match op {
        FieldOperation::Add => (a + b) % modulus,
        FieldOperation::Sub => ((a + modulus) - b) % modulus,
        FieldOperation::Mul => (a * b) % modulus,
        FieldOperation::Div => unreachable!("fp_op only supports Add/Sub/Mul"),
    };
    let mut result = result.to_u32_digits();
    result.resize(num_words, 0);
    result
}

/// Number of u32 words in a `P`-field degree-2 extension element (`c0`, `c1` concatenated).
pub(crate) fn fp2_num_words<P: FpOpField>() -> usize {
    <P as NumWords>::WordsCurvePoint::USIZE
}

/// `x op y` in `P`'s degree-2 extension field -- mirrors `Fp2AddSubSyscall::execute`'s compute
/// step exactly. Only `Add`/`Sub` are valid; mirrors the source's own restriction.
pub(crate) fn fp2_addsub<P: FpOpField>(x: &[u32], y: &[u32], op: FieldOperation) -> Vec<u32> {
    let num_words = fp2_num_words::<P>();
    let (ac0, ac1) = x.split_at(x.len() / 2);
    let (bc0, bc1) = y.split_at(y.len() / 2);
    let ac0 = &BigUint::from_slice(ac0);
    let ac1 = &BigUint::from_slice(ac1);
    let bc0 = &BigUint::from_slice(bc0);
    let bc1 = &BigUint::from_slice(bc1);
    let modulus = &BigUint::from_bytes_le(P::MODULUS);
    let (c0, c1) = match op {
        FieldOperation::Add => ((ac0 + bc0) % modulus, (ac1 + bc1) % modulus),
        FieldOperation::Sub => {
            ((ac0 + modulus - bc0) % modulus, (ac1 + modulus - bc1) % modulus)
        }
        _ => unreachable!("fp2_addsub only supports Add/Sub"),
    };
    let mut result = c0.to_u32_digits();
    result.resize(num_words / 2, 0);
    result.extend_from_slice(&c1.to_u32_digits());
    result.resize(num_words, 0);
    result
}

/// `x * y` in `P`'s degree-2 extension field -- mirrors `Fp2MulSyscall::execute`'s compute step
/// exactly.
pub(crate) fn fp2_mul<P: FpOpField>(x: &[u32], y: &[u32]) -> Vec<u32> {
    let num_words = fp2_num_words::<P>();
    let (ac0, ac1) = x.split_at(x.len() / 2);
    let (bc0, bc1) = y.split_at(y.len() / 2);
    let ac0 = &BigUint::from_slice(ac0);
    let ac1 = &BigUint::from_slice(ac1);
    let bc0 = &BigUint::from_slice(bc0);
    let bc1 = &BigUint::from_slice(bc1);
    let modulus = &BigUint::from_bytes_le(P::MODULUS);
    let ac0_bc0 = (ac0 * bc0) % modulus;
    let ac1_bc1 = (ac1 * bc1) % modulus;
    let ac0_bc1 = (ac0 * bc1) % modulus;
    let ac1_bc0 = (ac1 * bc0) % modulus;
    let c0 = if ac0_bc0 < ac1_bc1 {
        ((modulus + &ac0_bc0) - &ac1_bc1) % modulus
    } else {
        (&ac0_bc0 - &ac1_bc1) % modulus
    };
    let c1 = (&ac0_bc1 + &ac1_bc0) % modulus;
    let mut result = c0.to_u32_digits();
    result.resize(num_words / 2, 0);
    result.extend_from_slice(&c1.to_u32_digits());
    result.resize(num_words, 0);
    result
}

/// `(x * y) mod modulus` as 256-bit little-endian word arrays -- mirrors `Uint256MulSyscall::execute`'s
/// compute step (`syscalls/precompiles/uint256.rs`) exactly, split from the memory accesses that
/// gather `x`/`y`/`modulus`. A zero `modulus` means "mod 2^256" (unconstrained multiplication),
/// mirroring the source's own convention.
pub(crate) fn uint256_mul(x: &[u32; 8], y: &[u32; 8], modulus: &[u32; 8]) -> [u32; 8] {
    let uint256_x = BigUint::from_bytes_le(&zkm_primitives::consts::words_to_bytes_le_vec(x));
    let uint256_y = BigUint::from_bytes_le(&zkm_primitives::consts::words_to_bytes_le_vec(y));
    let uint256_modulus =
        BigUint::from_bytes_le(&zkm_primitives::consts::words_to_bytes_le_vec(modulus));

    let result: BigUint = if num::Zero::is_zero(&uint256_modulus) {
        let modulus = <BigUint as num::One>::one() << 256;
        (uint256_x * uint256_y) % modulus
    } else {
        (uint256_x * uint256_y) % uint256_modulus
    };

    let mut result_bytes = result.to_bytes_le();
    result_bytes.resize(32, 0u8);
    zkm_primitives::consts::bytes_to_words_le::<8>(&result_bytes)
}

/// Number of u32 words in a 256-bit value.
pub(crate) const U256_NUM_WORDS: usize = 8;
/// Number of u32 words in a 2048-bit value.
pub(crate) const U2048_NUM_WORDS: usize = 64;

/// `a * b`, split into a 2048-bit low half and a 256-bit high half -- mirrors
/// `U256xU2048MulSyscall::execute`'s compute step (`syscalls/precompiles/u256x2048_mul.rs`)
/// exactly, split from the memory accesses that gather `a`/`b`. Returns `(lo, hi)`.
pub(crate) fn u256xu2048_mul(
    a: &[u32; U256_NUM_WORDS],
    b: &[u32; U2048_NUM_WORDS],
) -> ([u32; U2048_NUM_WORDS], [u32; U256_NUM_WORDS]) {
    let uint256_a = BigUint::from_bytes_le(&zkm_primitives::consts::words_to_bytes_le_vec(a));
    let uint2048_b = BigUint::from_bytes_le(&zkm_primitives::consts::words_to_bytes_le_vec(b));
    let result = uint256_a * uint2048_b;

    let two_to_2048 = <BigUint as num::One>::one() << 2048;
    let (hi, lo) = result.div_rem(&two_to_2048);

    let mut lo_bytes = lo.to_bytes_le();
    lo_bytes.resize(U2048_NUM_WORDS * 4, 0u8);
    let lo_words = zkm_primitives::consts::bytes_to_words_le::<U2048_NUM_WORDS>(&lo_bytes);

    let mut hi_bytes = hi.to_bytes_le();
    hi_bytes.resize(U256_NUM_WORDS * 4, 0u8);
    let hi_words = zkm_primitives::consts::bytes_to_words_le::<U256_NUM_WORDS>(&hi_bytes);

    (lo_words, hi_words)
}

/// Number of `KoalaBear` field elements in a Poseidon2 permutation state.
pub(crate) const POSEIDON2_STATE_SIZE: usize = 16;

/// Poseidon2 permutation over the `KoalaBear` field -- mirrors `Poseidon2PermuteSyscall::execute`'s
/// compute step (`syscalls/precompiles/poseidon2/permute.rs`) exactly, split from the memory
/// accesses that gather `pre_state`.
pub(crate) fn poseidon2_permute(
    pre_state: [u32; POSEIDON2_STATE_SIZE],
) -> [u32; POSEIDON2_STATE_SIZE] {
    let mut state = pre_state.map(KoalaBear::from_canonical_u32);
    let hasher = poseidon2_init();
    hasher.permute_mut(&mut state);
    state.map(|x| x.as_canonical_u32())
}

/// Return value for an unsupported/invalid Linux-syscall argument -- mirrors `sys_linux`'s shared
/// `MIPS_EBADF` convention across `sysfcntl.rs`/`sysread.rs`.
pub(crate) const MIPS_EBADF: u32 = 9;

/// Prover-side safety bound on heap growth from the program's initial `BRK` value, mirroring
/// `sysbrk.rs`'s identical constant.
const MAX_HEAP_SIZE: u32 = 0x4000_0000;

/// The highest `brk` value a program may resolve to, given its initial value -- mirrors
/// `sysbrk.rs::max_brk` exactly.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidSyscallArgs`] if `initial_brk + MAX_HEAP_SIZE` overflows.
pub(crate) fn max_brk(initial_brk: u32) -> Result<u32, ExecutionError> {
    let limit =
        initial_brk.checked_add(MAX_HEAP_SIZE).ok_or(ExecutionError::InvalidSyscallArgs())?;
    Ok(limit.min(crate::program::MAX_MEMORY as u32))
}

/// Resolves a `brk(requested_brk)` call -- mirrors `sysbrk.rs::resolve_brk` exactly.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidSyscallArgs`] if the resolved value would exceed
/// [`max_brk`]'s limit.
pub(crate) fn resolve_brk(
    initial_brk: u32,
    current_brk: u32,
    requested_brk: u32,
) -> Result<u32, ExecutionError> {
    let limit = max_brk(initial_brk)?;
    let v0 = requested_brk.max(current_brk);
    if v0 > limit {
        return Err(ExecutionError::InvalidSyscallArgs());
    }
    Ok(v0)
}

/// Rounds `size` up to the next page boundary -- mirrors `sysmmap.rs::align_size` exactly.
///
/// # Errors
///
/// Returns [`ExecutionError::InvalidSyscallArgs`] on overflow.
pub(crate) fn align_size(size: u32) -> Result<u32, ExecutionError> {
    const PAGE_ADDR_MASK: u32 = (1 << 12) - 1;
    const PAGE_SIZE: u32 = 1 << 12;
    if size & PAGE_ADDR_MASK == 0 {
        return Ok(size);
    }
    size.checked_add(PAGE_SIZE - (size & PAGE_ADDR_MASK)).ok_or(ExecutionError::InvalidSyscallArgs())
}

/// `(v0, a3)` for an `fcntl(fd, cmd)` call -- mirrors `sysfcntl.rs::SysFcntlSyscall::execute`'s
/// compute step exactly (`a1 == 3` is `F_GETFL`, `a1 == 1` is `F_GETFD`; anything else is
/// unsupported).
pub(crate) fn fcntl_result(fd: u32, cmd: u32) -> (u32, u32) {
    if cmd == 3 {
        match fd {
            FD_STDIN => (0, 0),
            FD_STDOUT | FD_STDERR => (1, 0),
            _ => (0xffff_ffff, MIPS_EBADF),
        }
    } else if cmd == 1 {
        match fd {
            FD_STDIN | FD_STDOUT | FD_STDERR => (fd, 0),
            _ => (0xffff_ffff, MIPS_EBADF),
        }
    } else {
        (0xffff_ffff, MIPS_EBADF)
    }
}

/// `(v0, a3)` for a `read(fd, ...)` call -- mirrors `sysread.rs::SysReadSyscall::execute`'s
/// compute step exactly (only `FD_STDIN` is supported; a read of zero bytes always "succeeds").
pub(crate) fn read_result(fd: u32) -> (u32, u32) {
    if fd == FD_STDIN {
        (0, 0)
    } else {
        (0xffff_ffff, MIPS_EBADF)
    }
}

/// `CoreVM` -- the shared oracle-driven replay engine `SplicingVM` and `TracingVM` are both built
/// on. Unlike `MinimalExecutor`, it has **no backing RAM at all**: every RAM access
/// (`mr`/`mw`-equivalent) is answered by popping the next entry off a [`MinimalTrace`]'s oracle
/// log, in exactly the order `MinimalExecutor` produced it -- this is the core determinism
/// invariant the whole split-pipeline design rests on. Registers stay plain live state (never
/// oracle-logged); only the RAM-log cursor is threaded through.
///
/// Deliberately **not** generic over an `ExecutionMode`/`UserMode` axis: the page-permission/
/// mprotect feature that axis would exist for is out of scope here. If it's ever wanted, add it
/// back as a type parameter then -- don't speculatively add it now.
pub(crate) struct CoreVM<'a> {
    registers: [u32; NUM_REGISTERS],
    /// The `clk` each register was last written at -- see `MinimalExecutor`'s identically-named
    /// field doc comment. Recomputed independently during replay (deterministic, given the same
    /// instruction sequence and starting baseline `MinimalExecutor` used), never oracle-logged.
    register_timestamps: [u64; NUM_REGISTERS],
    pc: u32,
    next_pc: u32,
    clk: u64,
    clk_end: u64,
    next_is_delayslot: bool,
    exited: bool,
    max_syscall_cycles: u32,
    mem_reads: MemReads<'a>,
    program: Arc<Program>,
}

/// The result of replaying up to a chunk/shard boundary. `TraceEnd` mirrors SP1's `CycleResult`:
/// the oracle log for this `MinimalTrace` is exhausted (`clk == clk_end`), but the program itself
/// has not halted -- a later `MinimalTrace`/chunk continues it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoreVMStatus {
    /// The program halted for real (explicit `HALT` syscall, or ran off the end of the program).
    Done,
    /// `clk` reached `clk_end`; this chunk's oracle log is exhausted.
    TraceEnd,
}

impl<'a> CoreVM<'a> {
    #[must_use]
    pub(crate) fn new<T: MinimalTrace>(
        trace: &'a T,
        program: Arc<Program>,
        max_syscall_cycles: u32,
    ) -> Self {
        let pc = trace.pc_start();
        Self {
            registers: trace.start_registers(),
            register_timestamps: trace.start_register_timestamps(),
            pc,
            // Always `pc_start + 4`: a chunk boundary is only ever legal when
            // `!next_is_delayslot`, which implies `next_pc == pc + 4` exactly. Verified with a
            // `debug_assert_eq!` at every constructed boundary in `SplicingVM`/`MinimalExecutor`.
            next_pc: pc.wrapping_add(4),
            clk: trace.clk_start(),
            clk_end: trace.clk_end(),
            next_is_delayslot: false,
            exited: false,
            max_syscall_cycles,
            mem_reads: trace.mem_reads(),
            program,
        }
    }

    #[must_use]
    pub(crate) fn registers(&self) -> [u32; NUM_REGISTERS] {
        self.registers
    }

    #[must_use]
    pub(crate) fn register_timestamps(&self) -> [u64; NUM_REGISTERS] {
        self.register_timestamps
    }

    #[must_use]
    pub(crate) fn pc(&self) -> u32 {
        self.pc
    }

    #[must_use]
    pub(crate) fn clk(&self) -> u64 {
        self.clk
    }

    #[must_use]
    pub(crate) fn clk_end(&self) -> u64 {
        self.clk_end
    }

    #[must_use]
    pub(crate) fn next_is_delayslot(&self) -> bool {
        self.next_is_delayslot
    }

    #[must_use]
    pub(crate) fn program(&self) -> &Arc<Program> {
        &self.program
    }

    /// Peeks a register's live value without any oracle interaction (registers are never
    /// oracle-logged) -- used by `SplicingVM` to classify a `SYSCALL` instruction's precompile
    /// cost *before* executing it (`V0` holds the syscall code).
    #[must_use]
    pub(crate) fn reg_peek(&self, r: Register) -> u32 {
        self.reg(r)
    }

    /// Remaining (unconsumed) oracle-log entries -- used by `SplicingVM` to compute how many
    /// entries the shard just spliced actually consumed.
    #[must_use]
    pub(crate) fn mem_reads_remaining(&self) -> usize {
        self.mem_reads.len()
    }

    pub(crate) fn is_halted(&self) -> bool {
        self.pc == 0
            || self.exited
            || self.pc.wrapping_sub(self.program.pc_base)
                >= (self.program.instructions.len() * 4) as u32
    }

    // ---- register file (never oracle-logged) ----

    pub(crate) fn reg(&self, r: Register) -> u32 {
        self.registers[r as usize]
    }

    /// The `clk` register `r` was last written at -- `0` means "never touched". Looked up
    /// *before* overwriting a register (via [`Self::set_reg`]/[`Self::set_reg_aux`]) to recover a
    /// real `MemoryReadRecord`/`MemoryWriteRecord.prev_timestamp` (`TracingVM`'s job; `CoreVM`'s
    /// own replay dispatch, used by `SplicingVM`, never needs this).
    #[must_use]
    pub(crate) fn reg_timestamp(&self, r: Register) -> u64 {
        self.register_timestamps[r as usize]
    }

    /// Writes `value` to register `r`, tagging its consistency-timestamp at `clk + position` --
    /// matches `Executor::rw_cpu`'s exact scheme for the instruction's own op_a/op_b/op_c/hi
    /// slots (see `MinimalExecutor::set_reg`'s identical doc comment).
    pub(crate) fn set_reg(&mut self, r: Register, value: u32, position: MemoryAccessPosition) {
        let value = if r == Register::ZERO { 0 } else { value };
        self.registers[r as usize] = value;
        self.register_timestamps[r as usize] = self.clk + position as u64;
    }

    /// Writes `value` to register `r` for a syscall-internal auxiliary register access -- tags
    /// its consistency-timestamp at the bare `clk` (see `MinimalExecutor::set_reg_aux`'s
    /// identical doc comment).
    pub(crate) fn set_reg_aux(&mut self, r: Register, value: u32) {
        let value = if r == Register::ZERO { 0 } else { value };
        self.registers[r as usize] = value;
        self.register_timestamps[r as usize] = self.clk;
    }

    // ---- oracle log (RAM replay) ----

    /// Pops the next oracle-log entry -- the *only* source of RAM values during replay (no
    /// backing RAM exists here at all). Used for both loads (the returned value directly) and
    /// stores (the returned value is the preimage; the new value is a pure function of it + a
    /// live register, recomputed identically to `MinimalExecutor::execute_store`, never oracled).
    pub(crate) fn next_oracle_value(&mut self) -> u32 {
        self.next_oracle_entry().value
    }

    /// Pops the next oracle-log entry in full (value *and* its `clk`) -- needed to recover a real
    /// `prev_timestamp` for a RAM `MemoryReadRecord`/`MemoryWriteRecord` (`next_oracle_value`
    /// discards the `clk`, which is fine for the many callers that only need the value to
    /// recompute a result).
    pub(crate) fn next_oracle_entry(&mut self) -> MemValue {
        self.mem_reads.next().expect(
            "oracle log exhausted before replay finished -- MinimalExecutor and CoreVM have \
             desynced on how many entries a memory access logs",
        )
    }

    /// Pops `len` consecutive oracle-log entries -- see `next_oracle_value`'s doc comment.
    pub(crate) fn next_oracle_values(&mut self, len: usize) -> Vec<u32> {
        (0..len).map(|_| self.next_oracle_value()).collect()
    }

    /// Pops `len` consecutive oracle-log entries in full -- see `next_oracle_entry`'s doc comment.
    pub(crate) fn next_oracle_entries(&mut self, len: usize) -> Vec<MemValue> {
        (0..len).map(|_| self.next_oracle_entry()).collect()
    }

    // ---- direct pc/clk/delay-slot access for TracingVM's own dispatch ----

    pub(crate) fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    pub(crate) fn next_pc(&self) -> u32 {
        self.next_pc
    }

    pub(crate) fn set_next_pc(&mut self, next_pc: u32) {
        self.next_pc = next_pc;
    }

    pub(crate) fn set_next_is_delayslot(&mut self, value: bool) {
        self.next_is_delayslot = value;
    }

    pub(crate) fn bump_clk(&mut self) {
        self.clk = bump_clk_high_if_need(self.clk, self.max_syscall_cycles);
    }

    pub(crate) fn advance_clk(&mut self) {
        self.clk += 5;
    }

    /// Bumps `clk` by a syscall's `num_extra_cycles`, on top of the base `advance_clk` every
    /// instruction gets -- mirrors `Executor::execute_operation`'s `self.state.clk +=
    /// u64::from(precompile_cycles)`, applied inside the `SYSCALL` branch itself rather than
    /// uniformly. `pub(crate)`, not private: `TracingVM` (a sibling module) has its own
    /// `execute_syscall` and must apply the same bump `CoreVM::execute_syscall` applies internally.
    pub(crate) fn advance_clk_extra(&mut self, extra_cycles: u32) {
        self.clk += u64::from(extra_cycles);
    }

    // ---- replay loop ----

    /// Replays instructions until the program halts or `clk` reaches `clk_end`.
    ///
    /// # Errors
    ///
    /// Propagates any [`ExecutionError`] from executing an instruction.
    pub(crate) fn execute(&mut self) -> Result<CoreVMStatus, ExecutionError> {
        // Do-while, matching `MinimalExecutor::try_execute_chunk`'s identical fix: `is_halted()`'s
        // `pc == 0` arm is also the *initial* `pc` for any program whose `pc_start == 0`, so it
        // must never be consulted before at least one instruction has actually retired.
        loop {
            self.execute_instruction()?;
            if self.is_halted() {
                return Ok(CoreVMStatus::Done);
            }
            if self.clk >= self.clk_end {
                return Ok(CoreVMStatus::TraceEnd);
            }
        }
    }

    /// `pub(crate)`, not private: `SplicingVM` (a sibling module) drives this directly
    /// (rather than the whole-run `execute()` loop) so it can classify each retired instruction's
    /// `MipsAirId` cost *before* deciding whether the shard-cut check should even run (mirrors
    /// `execute()`'s own do-while structure -- see its doc comment on why `is_halted()` must never
    /// be consulted before at least one instruction retires).
    pub(crate) fn execute_instruction(&mut self) -> Result<(), ExecutionError> {
        let instruction = self.program.fetch(self.pc);
        self.clk = bump_clk_high_if_need(self.clk, self.max_syscall_cycles);
        self.execute_operation(&instruction)?;
        self.clk += 5;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_operation(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let next_pc_in = self.next_pc;
        let mut next_next_pc = self.next_pc.wrapping_add(4);
        self.next_is_delayslot = false;

        if instruction.is_alu_instruction() {
            let (rd, b, c) = self.alu_operands(instruction);
            let (a, hi) = alu_compute(instruction.opcode, b, c)?;
            self.alu_write(instruction.opcode, rd, a, hi);
        } else if instruction.is_memory_load_instruction() {
            self.execute_load(instruction)?;
        } else if instruction.is_memory_store_instruction() {
            self.execute_store(instruction)?;
        } else if instruction.is_branch_instruction() {
            let rs: Register = instruction.op_a.into();
            let src1 = self.reg(rs);
            let src2 = if instruction.opcode.only_one_operand() {
                0
            } else {
                self.reg((instruction.op_b as u8).into())
            };
            let offset = instruction.op_c;
            if branch_taken(instruction.opcode, src1, src2) {
                next_next_pc = branch_target(next_pc_in, offset);
            }
            self.next_is_delayslot = true;
        } else if instruction.is_jump_instruction() {
            let link: Register = instruction.op_a.into();
            let (return_pc, target) = match instruction.opcode {
                Opcode::Jump => {
                    let target_reg: Register = (instruction.op_b as u8).into();
                    let target_pc = self.reg(target_reg);
                    jump_jr_result(next_pc_in, target_pc)
                }
                Opcode::Jumpi => jump_jumpi_result(next_pc_in, instruction.op_b),
                Opcode::JumpDirect => jump_direct_result(next_pc_in, instruction.op_b),
                _ => unreachable!("not a jump opcode: {:?}", instruction.opcode),
            };
            self.set_reg(link, return_pc, MemoryAccessPosition::A);
            next_next_pc = target;
            self.next_is_delayslot = true;
        } else if instruction.is_mov_cond_instruction() {
            let rd: Register = instruction.op_a.into();
            let rs: Register = (instruction.op_b as u8).into();
            let rt: Register = (instruction.op_c as u8).into();
            let prev_a = self.reg(rd);
            let b = self.reg(rs);
            let c = self.reg(rt);
            let a = condmov_result(instruction.opcode, prev_a, b, c);
            self.set_reg(rd, a, MemoryAccessPosition::A);
        } else if instruction.is_misc_instruction() {
            self.execute_misc(instruction)?;
        } else if instruction.is_syscall_instruction() {
            let syscall_next_pc = self.execute_syscall()?;
            next_next_pc = syscall_next_pc.wrapping_add(4);
            self.pc = syscall_next_pc;
            self.next_pc = next_next_pc;
            return Ok(());
        } else {
            return Err(ExecutionError::UnsupportedInstruction(instruction.opcode as u32));
        }

        if next_next_pc == 0 {
            return Err(ExecutionError::NullPointerReference());
        }
        self.pc = next_pc_in;
        self.next_pc = next_next_pc;
        Ok(())
    }

    /// See `MinimalExecutor::alu_operands` -- identical decode, register-only, no oracle.
    fn alu_operands(&self, instruction: &Instruction) -> (Register, u32, u32) {
        if !instruction.imm_c {
            let rd = instruction.op_a.into();
            let b = self.reg((instruction.op_b as u8).into());
            let c = self.reg((instruction.op_c as u8).into());
            (rd, b, c)
        } else if !instruction.imm_b {
            let rd = instruction.op_a.into();
            let b = self.reg((instruction.op_b as u8).into());
            (rd, b, instruction.op_c)
        } else {
            (instruction.op_a.into(), instruction.op_b, instruction.op_c)
        }
    }

    fn alu_write(&mut self, opcode: Opcode, rd: Register, a: u32, hi: u32) {
        if opcode.is_use_lo_hi_alu() {
            self.set_reg(Register::LO, a, MemoryAccessPosition::A);
            self.set_reg(Register::HI, hi, MemoryAccessPosition::HI);
        } else {
            self.set_reg(rd, a, MemoryAccessPosition::A);
        }
    }

    /// See `MinimalExecutor::execute_load`: identical decode/merge math, but `mem` comes from the
    /// oracle log instead of a live page table.
    fn execute_load(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs_raw = self.reg(rs_reg);
        let rt = self.reg(rt_reg);

        let addr = rs_raw.wrapping_add(offset);
        let mem = self.next_oracle_value();
        let rs = addr;

        let val = match instruction.opcode {
            Opcode::LH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LH, addr));
                }
                sign_extend::<16>((mem >> ((rs & 2) * 8)) & 0xffff)
            }
            Opcode::LWL => {
                let i = rs & 3;
                let val = mem << (24 - i * 8);
                let mask: u32 = 0xFFFF_FFFF_u32 << (24 - i * 8);
                (rt & (!mask)) | val
            }
            Opcode::LW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LW, addr));
                }
                mem
            }
            Opcode::LBU => (mem >> ((rs & 3) * 8)) & 0xff,
            Opcode::LHU => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LHU, addr));
                }
                (mem >> ((rs & 2) * 8)) & 0xffff
            }
            Opcode::LWR => {
                let i = rs & 3;
                let val = mem >> (i * 8);
                let mask = 0xFFFF_FFFF_u32 >> (i * 8);
                (rt & (!mask)) | val
            }
            Opcode::LL => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LL, addr));
                }
                mem
            }
            Opcode::LB => sign_extend::<8>((mem >> ((rs & 3) * 8)) & 0xff),
            _ => unreachable!("not a load opcode: {:?}", instruction.opcode),
        };
        self.set_reg(rt_reg, val, MemoryAccessPosition::A);
        Ok(())
    }

    /// See `MinimalExecutor::execute_store`: identical decode/merge math, but the current word
    /// (for byte/half merging) comes from the oracle log instead of an unlogged page-table peek
    /// -- there is nothing to peek here at all. No bounds check: `MinimalExecutor` already
    /// validated this address (or this chunk wouldn't exist), and there is no RAM to be out of
    /// bounds of during replay.
    fn execute_store(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let rt_reg: Register = instruction.op_a.into();
        let rs_reg: Register = (instruction.op_b as u8).into();
        let offset = instruction.op_c;
        let rs = self.reg(rs_reg);
        let rt = self.reg(rt_reg);

        let addr = rs.wrapping_add(offset);
        let mem = self.next_oracle_value();

        // Kept for exact parity with `MinimalExecutor::execute_store` even though a validly-
        // produced chunk can never actually trip these (the same deterministic inputs already
        // passed this same check when the chunk was produced) -- avoids CoreVM silently
        // accepting something Minimal wouldn't have.
        let val = match instruction.opcode {
            Opcode::SB => {
                let i = addr & 3;
                let val = (rt & 0xff) << (i * 8);
                let mask = 0xFFFF_FFFF_u32 ^ (0xff << (i * 8));
                (mem & mask) | val
            }
            Opcode::SH => {
                if addr & 1 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SH, addr));
                }
                let i = addr & 2;
                let val = (rt & 0xffff) << (i * 8);
                let mask = 0xFFFF_FFFF_u32 ^ (0xffff << (i * 8));
                (mem & mask) | val
            }
            Opcode::SWL => {
                let i = addr & 3;
                let val = rt >> (24 - i * 8);
                let mask = 0xFFFF_FFFF_u32 >> (24 - i * 8);
                (mem & (!mask)) | val
            }
            Opcode::SW => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SW, addr));
                }
                rt
            }
            Opcode::SWR => {
                let i = addr & 3;
                let val = rt << (i * 8);
                let mask = 0xFFFF_FFFF_u32 << (i * 8);
                (mem & (!mask)) | val
            }
            Opcode::SC => {
                if addr & 3 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SC, addr));
                }
                rt
            }
            _ => unreachable!("not a store opcode: {:?}", instruction.opcode),
        };
        let _ = val; // the recomputed value has no destination here -- there is no RAM to write
                     // to during replay; only its *derivation* matching `MinimalExecutor`'s
                     // formula (which produced this oracle entry in the first place) matters.
        if instruction.opcode == Opcode::SC {
            self.set_reg(rt_reg, 1, MemoryAccessPosition::A);
        }
        Ok(())
    }

    fn execute_misc(&mut self, instruction: &Instruction) -> Result<(), ExecutionError> {
        if instruction.opcode == Opcode::WSBH {
            let rd: Register = instruction.op_a.into();
            let rt: Register = (instruction.op_b as u8).into();
            let b = self.reg(rt);
            self.set_reg(rd, wsbh(b), MemoryAccessPosition::A);
            return Ok(());
        }

        let rd: Register = instruction.op_a.into();
        let rt: Register = (instruction.op_b as u8).into();
        let c = instruction.op_c;
        match instruction.opcode {
            Opcode::SEXT => {
                let b = self.reg(rt);
                self.set_reg(rd, sext(b, c), MemoryAccessPosition::A);
            }
            Opcode::EXT => {
                let b = self.reg(rt);
                self.set_reg(rd, ext(b, c)?, MemoryAccessPosition::A);
            }
            Opcode::INS => {
                let b = self.reg(rt);
                let a = self.reg(rd);
                self.set_reg(rd, ins(a, b, c)?, MemoryAccessPosition::A);
            }
            Opcode::TEQ => {
                let rs: Register = instruction.op_a.into();
                let rt: Register = (instruction.op_b as u8).into();
                let src2 = self.reg(rt);
                let src1 = self.reg(rs);
                teq(src1, src2)?;
            }
            Opcode::MADDU | Opcode::MSUBU | Opcode::MADD | Opcode::MSUB => {
                let lo_reg: Register = instruction.op_a.into();
                let rs: Register = (instruction.op_c as u8).into();
                let c = self.reg(rs);
                let b = self.reg(rt);
                let lo = self.reg(Register::LO);
                let hi = self.reg(Register::HI);
                let (out_lo, out_hi) = match instruction.opcode {
                    Opcode::MADDU => maddu(b, c, lo, hi),
                    Opcode::MSUBU => msubu(b, c, lo, hi),
                    Opcode::MADD => madd(b, c, lo, hi),
                    Opcode::MSUB => msub(b, c, lo, hi),
                    _ => unreachable!(),
                };
                self.set_reg(lo_reg, out_lo, MemoryAccessPosition::A);
                self.set_reg(Register::HI, out_hi, MemoryAccessPosition::HI);
            }
            _ => unreachable!("not a misc opcode: {:?}", instruction.opcode),
        }
        Ok(())
    }

    /// See `minimal::ecall`'s module doc for scope (HALT/WRITE/SYS_BRK real, everything else a
    /// documented no-op). `HALT`/`SYS_BRK` are fully register-and-`Program`-derived (no RAM
    /// involved at all) so this replays them by literally re-running the same logic; `WRITE`
    /// pops the same number of oracle entries, in the same order, that
    /// `MinimalExecutor::execute_syscall`'s `WRITE` arm logged.
    fn execute_syscall(&mut self) -> Result<u32, ExecutionError> {
        let syscall_id = self.reg(Register::V0);
        let code = SyscallCode::from_u32(syscall_id);
        let arg1 = self.reg(Register::A0);
        let arg2 = self.reg(Register::A1);

        let mut next_pc = self.pc.wrapping_add(4);
        let mut extra_cycles = 0u32;
        let a0_result: Option<u32> = match code {
            SyscallCode::HALT => {
                let exit_code = arg1;
                next_pc = 0;
                if exit_code != 0 {
                    return Err(ExecutionError::HaltWithNonZeroExitCode(exit_code));
                }
                self.exited = true;
                None
            }
            SyscallCode::WRITE => {
                let fd = arg1;
                let nbytes = self.reg(Register::A2);
                let bytes: Vec<u8> = (0..nbytes)
                    .map(|i| {
                        let word = self.next_oracle_value();
                        (word >> (((arg2 + i) % 4) * 8)) as u8
                    })
                    .collect();
                let _ = (fd, bytes); // consumed for oracle-log parity only -- `CoreVM` doesn't
                                     // reconstruct `public_values_stream` itself; `TracingVM`
                                     // does, from the same popped bytes.
                if fd == FD_PUBLIC_VALUES || fd == FD_STDOUT || fd == FD_STDERR || fd == FD_HINT {
                    // handled above by the pop loop; nothing further needed here.
                }
                None
            }
            SyscallCode::SYS_BRK => {
                let initial_brk = self
                    .program
                    .image
                    .get(&(Register::BRK as u32))
                    .copied()
                    .unwrap_or_else(|| self.reg(Register::BRK));
                let v0 = resolve_brk(initial_brk, initial_brk, arg1)?;
                self.set_reg_aux(Register::A3, 0);
                Some(v0)
            }
            SyscallCode::SHA_COMPRESS => {
                let h: [u32; 8] = std::array::from_fn(|_| self.next_oracle_value());
                let w: [u32; 64] = std::array::from_fn(|_| self.next_oracle_value());
                let out = sha256_compress(h, &w);
                for _ in out {
                    // Pops each write's logged preimage to stay in sync with
                    // `MinimalExecutor::execute_syscall`'s `mw` calls -- the popped value itself
                    // is unused since the new value is already known (`out`, a pure function of
                    // `h`/`w`).
                    self.next_oracle_value();
                }
                extra_cycles = 1;
                None
            }
            SyscallCode::SHA_EXTEND => {
                for _ in 16..64u32 {
                    let w_i_minus_15 = self.next_oracle_value();
                    let w_i_minus_2 = self.next_oracle_value();
                    let w_i_minus_16 = self.next_oracle_value();
                    let w_i_minus_7 = self.next_oracle_value();
                    let _w_i =
                        sha256_extend_word(w_i_minus_15, w_i_minus_2, w_i_minus_16, w_i_minus_7);
                    self.next_oracle_value(); // pops the write's logged preimage; see SHA_COMPRESS.
                }
                extra_cycles = 48;
                None
            }
            SyscallCode::KECCAK_SPONGE => {
                let input_len_u32s = self.next_oracle_value();
                let input_values: Vec<u32> =
                    (0..input_len_u32s).map(|_| self.next_oracle_value()).collect();
                let input_u64_values: Vec<u64> = input_values
                    .chunks_exact(2)
                    .map(|pair| pair[0] as u64 + ((pair[1] as u64) << 32))
                    .collect();

                let mut state = [0u64; KECCAK_STATE_SIZE_U64S];
                for block in input_u64_values.chunks_exact(KECCAK_GENERAL_BLOCK_SIZE_U64S) {
                    keccak_xor_block(&mut state, block);
                    keccakf(&mut state);
                }

                for _ in 0..KECCAK_GENERAL_OUTPUT_U64S {
                    self.next_oracle_value(); // least-sig write preimage; see SHA_COMPRESS.
                    self.next_oracle_value(); // most-sig write preimage.
                }
                extra_cycles = 1;
                None
            }
            SyscallCode::SECP256K1_ADD => {
                self.ec_add_replay::<Secp256k1>();
                extra_cycles = 1;
                None
            }
            SyscallCode::SECP256R1_ADD => {
                self.ec_add_replay::<Secp256r1>();
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_ADD => {
                self.ec_add_replay::<Bn254>();
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_ADD => {
                self.ec_add_replay::<Bls12381>();
                extra_cycles = 1;
                None
            }
            SyscallCode::SECP256K1_DOUBLE => {
                self.ec_double_replay::<Secp256k1>();
                None
            }
            SyscallCode::SECP256R1_DOUBLE => {
                self.ec_double_replay::<Secp256r1>();
                None
            }
            SyscallCode::BN254_DOUBLE => {
                self.ec_double_replay::<Bn254>();
                None
            }
            SyscallCode::BLS12381_DOUBLE => {
                self.ec_double_replay::<Bls12381>();
                None
            }
            SyscallCode::SECP256K1_DECOMPRESS => {
                self.ec_decompress_replay::<Secp256k1>(arg2)?;
                None
            }
            SyscallCode::SECP256R1_DECOMPRESS => {
                self.ec_decompress_replay::<Secp256r1>(arg2)?;
                None
            }
            SyscallCode::BLS12381_DECOMPRESS => {
                self.ec_decompress_replay::<Bls12381>(arg2)?;
                None
            }
            SyscallCode::ED_ADD => {
                self.ec_add_replay::<Ed25519>();
                extra_cycles = 1;
                None
            }
            SyscallCode::ED_DECOMPRESS => {
                self.ed_decompress_replay(arg2)?;
                None
            }
            SyscallCode::BN254_FP_ADD => {
                self.fp_op_replay::<Bn254BaseField>(FieldOperation::Add);
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_FP_SUB => {
                self.fp_op_replay::<Bn254BaseField>(FieldOperation::Sub);
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_FP_MUL => {
                self.fp_op_replay::<Bn254BaseField>(FieldOperation::Mul);
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP_ADD => {
                self.fp_op_replay::<Bls12381BaseField>(FieldOperation::Add);
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP_SUB => {
                self.fp_op_replay::<Bls12381BaseField>(FieldOperation::Sub);
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP_MUL => {
                self.fp_op_replay::<Bls12381BaseField>(FieldOperation::Mul);
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_FP2_ADD => {
                self.fp2_addsub_replay::<Bn254BaseField>(FieldOperation::Add);
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_FP2_SUB => {
                self.fp2_addsub_replay::<Bn254BaseField>(FieldOperation::Sub);
                extra_cycles = 1;
                None
            }
            SyscallCode::BN254_FP2_MUL => {
                self.fp2_mul_replay::<Bn254BaseField>();
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP2_ADD => {
                self.fp2_addsub_replay::<Bls12381BaseField>(FieldOperation::Add);
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP2_SUB => {
                self.fp2_addsub_replay::<Bls12381BaseField>(FieldOperation::Sub);
                extra_cycles = 1;
                None
            }
            SyscallCode::BLS12381_FP2_MUL => {
                self.fp2_mul_replay::<Bls12381BaseField>();
                extra_cycles = 1;
                None
            }
            SyscallCode::UINT256_MUL => {
                self.uint256_mul_replay();
                extra_cycles = 1;
                None
            }
            SyscallCode::U256XU2048_MUL => {
                self.u256xu2048_mul_replay();
                extra_cycles = 1;
                None
            }
            SyscallCode::POSEIDON2_PERMUTE => {
                self.poseidon2_permute_replay();
                None
            }
            SyscallCode::SYS_MMAP | SyscallCode::SYS_MMAP2 => {
                let size = align_size(arg2)?;
                let v0 = if arg1 == 0 {
                    let heap = self.reg(Register::HEAP);
                    self.set_reg_aux(Register::HEAP, heap.wrapping_add(size));
                    heap
                } else {
                    arg1
                };
                self.set_reg_aux(Register::A3, 0);
                Some(v0)
            }
            SyscallCode::SYS_CLONE => {
                self.set_reg_aux(Register::A3, 0);
                Some(1)
            }
            SyscallCode::SYS_EXT_GROUP => {
                next_pc = 0;
                self.set_reg_aux(Register::A3, 0);
                Some(0)
            }
            SyscallCode::SYS_FCNTL => {
                let (v0, a3) = fcntl_result(arg1, arg2);
                self.set_reg_aux(Register::A3, a3);
                Some(v0)
            }
            SyscallCode::SYS_READ => {
                let (v0, a3) = read_result(arg1);
                self.set_reg_aux(Register::A3, a3);
                Some(v0)
            }
            SyscallCode::SYS_WRITE => {
                let nbytes = self.reg(Register::A2);
                for _ in 0..nbytes {
                    self.next_oracle_value(); // write preimage; see SHA_COMPRESS.
                }
                self.set_reg_aux(Register::A3, 0);
                Some(nbytes)
            }
            SyscallCode::SYS_OPEN
            | SyscallCode::SYS_CLOSE
            | SyscallCode::SYS_RT_SIGACTION
            | SyscallCode::SYS_RT_SIGPROCMASK
            | SyscallCode::SYS_MADVISE
            | SyscallCode::SYS_GETTID
            | SyscallCode::SYS_SCHED_GETAFFINITY
            | SyscallCode::SYS_CLOCK_GETTIME
            | SyscallCode::SYS_NANOSLEEP
            | SyscallCode::SYS_PRLIMIT64
            | SyscallCode::SYS_SIGALTSTACK
            | SyscallCode::SYS_OPENAT
            | SyscallCode::SYS_FSTAT64
            | SyscallCode::SYS_MUNMAP => {
                self.set_reg_aux(Register::A3, 0);
                Some(0)
            }
            _ => None,
        };

        let a0 = a0_result.unwrap_or(syscall_id);
        self.set_reg(Register::V0, a0, MemoryAccessPosition::A);
        self.clk += u64::from(extra_cycles);
        Ok(next_pc)
    }

    /// Replays an `ec_add`: pops `q`'s reads, then `p`'s write-preimage (the value actually needed
    /// as an input, unlike `SHA_COMPRESS`'s discarded write pops -- see `minimal/ecall.rs`'s
    /// `ec_add_dispatch` for why the oracle order is q-then-p even though `p` is conceptually
    /// read first). The computed result has nowhere to go (`CoreVM` has no backing RAM); called
    /// only so `SplicingVM`'s replay exercises the exact same arithmetic `TracingVM` will.
    fn ec_add_replay<E: EllipticCurve>(&mut self) {
        let num_words = ec_num_words::<E>();
        let q = self.next_oracle_values(num_words);
        let p = self.next_oracle_values(num_words);
        ec_add::<E>(&p, &q);
    }

    /// Replays an `ec_double`: pops `p`'s write-preimage (the value actually needed as an input).
    fn ec_double_replay<E: EllipticCurve>(&mut self) {
        let num_words = ec_num_words::<E>();
        let p = self.next_oracle_values(num_words);
        ec_double::<E>(&p);
    }

    /// Replays an `ec_decompress`: pops `x`'s reads (used), then `y`'s write-preimage (discarded,
    /// like `SHA_COMPRESS`'s writes -- the old `y` value isn't an input to computing the new one).
    fn ec_decompress_replay<E: EllipticCurve>(&mut self, sign_bit: u32) -> Result<(), ExecutionError> {
        let num_words_field_element = ec_num_limb_words::<E>();
        let x_vec = self.next_oracle_values(num_words_field_element);
        let mut x_bytes_be = zkm_primitives::consts::words_to_bytes_le_vec(&x_vec);
        x_bytes_be.reverse();
        ec_decompress::<E>(&x_bytes_be, sign_bit).map_err(ExecutionError::CurveError)?;
        for _ in 0..num_words_field_element {
            self.next_oracle_value(); // write preimage; see SHA_COMPRESS.
        }
        Ok(())
    }

    /// Replays an `ed25519_decompress`: pops `y`'s reads (used), then `x`'s write-preimage
    /// (discarded).
    fn ed_decompress_replay(&mut self, sign: u32) -> Result<(), ExecutionError> {
        let y_vec = self.next_oracle_values(WORDS_FIELD_ELEMENT);
        let y_bytes: [u8; COMPRESSED_POINT_BYTES] =
            zkm_primitives::consts::words_to_bytes_le_vec(&y_vec).try_into().unwrap();
        ed25519_decompress(y_bytes, sign).map_err(ExecutionError::CurveError)?;
        for _ in 0..WORDS_FIELD_ELEMENT {
            self.next_oracle_value(); // write preimage; see SHA_COMPRESS.
        }
        Ok(())
    }

    /// Replays an `fp_op`: pops `y`'s reads, then `x`'s write-preimage (the value actually needed
    /// as an input -- see `ec_add_replay`'s doc comment for why the oracle order is y-then-x).
    fn fp_op_replay<P: FpOpField>(&mut self, op: FieldOperation) {
        let num_words = fp_num_words::<P>();
        let y = self.next_oracle_values(num_words);
        let x = self.next_oracle_values(num_words);
        fp_op::<P>(&x, &y, op);
    }

    /// Replays an `fp2_addsub`: pops `y`'s reads, then `x`'s write-preimage.
    fn fp2_addsub_replay<P: FpOpField>(&mut self, op: FieldOperation) {
        let num_words = fp2_num_words::<P>();
        let y = self.next_oracle_values(num_words);
        let x = self.next_oracle_values(num_words);
        fp2_addsub::<P>(&x, &y, op);
    }

    /// Replays an `fp2_mul`: pops `y`'s reads, then `x`'s write-preimage.
    fn fp2_mul_replay<P: FpOpField>(&mut self) {
        let num_words = fp2_num_words::<P>();
        let y = self.next_oracle_values(num_words);
        let x = self.next_oracle_values(num_words);
        fp2_mul::<P>(&x, &y);
    }

    /// Replays a `uint256_mul`: pops `y`'s reads, `modulus`'s reads, then `x`'s write-preimage
    /// (the value actually needed as an input).
    fn uint256_mul_replay(&mut self) {
        let y: [u32; 8] = self.next_oracle_values(WORDS_FIELD_ELEMENT).try_into().unwrap();
        let modulus: [u32; 8] = self.next_oracle_values(WORDS_FIELD_ELEMENT).try_into().unwrap();
        let x: [u32; 8] = self.next_oracle_values(WORDS_FIELD_ELEMENT).try_into().unwrap();
        uint256_mul(&x, &y, &modulus);
    }

    /// Replays a `u256xu2048_mul`: pops `a`'s reads, `b`'s reads, then `lo`'s and `hi`'s
    /// write-preimages (discarded -- unlike `ec_add`, neither write's old value is a compute
    /// input here).
    fn u256xu2048_mul_replay(&mut self) {
        let a: [u32; U256_NUM_WORDS] = self.next_oracle_values(U256_NUM_WORDS).try_into().unwrap();
        let b: [u32; U2048_NUM_WORDS] = self.next_oracle_values(U2048_NUM_WORDS).try_into().unwrap();
        let (lo, hi) = u256xu2048_mul(&a, &b);
        for _ in &lo {
            self.next_oracle_value(); // write preimage; see SHA_COMPRESS.
        }
        for _ in &hi {
            self.next_oracle_value();
        }
    }

    /// Replays a `poseidon2_permute`: pops the state's write-preimage (the value actually needed
    /// as an input).
    fn poseidon2_permute_replay(&mut self) {
        let pre_state: [u32; POSEIDON2_STATE_SIZE] =
            self.next_oracle_values(POSEIDON2_STATE_SIZE).try_into().unwrap();
        poseidon2_permute(pre_state);
    }
}

/// Mirrors `executor.rs`'s free `sign_extend` helper (also duplicated in `minimal/mod.rs` -- see
/// its own doc comment on why small, pure helpers like this are intentionally not shared via a
/// third location).
fn sign_extend<const BITS: u32>(value: u32) -> u32 {
    let shift = 32 - BITS;
    (((value << shift) as i32) >> shift) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{register::Register, Executor, Instruction, Program};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use zkm_stark::ZKMCoreOpts;

    const T0: u8 = Register::T0 as u8;
    const T1: u8 = Register::T1 as u8;
    const T2: u8 = Register::T2 as u8;
    const T3: u8 = Register::T3 as u8;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(0xC0FF_EE01)
    }

    fn load_imm(reg: u8, value: u32) -> Instruction {
        Instruction::new(Opcode::ADD, reg, 0, value, false, true)
    }

    fn run(instructions: Vec<Instruction>) -> Executor<'static> {
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().unwrap();
        runtime
    }

    /// Drives the real `Executor` through a register-register-form ALU instruction (`b` from
    /// `$t0`, `c` from `$t1`, result into `$t2`, HI/LO real registers for the dual-result ops)
    /// and returns `(a, hi)` exactly as `alu_compute` does, for direct comparison.
    fn run_alu(opcode: Opcode, b: u32, c: u32) -> Result<(u32, u32), crate::ExecutionError> {
        let instructions = vec![
            load_imm(T0, b),
            load_imm(T1, c),
            Instruction::new(opcode, T2, T0 as u32, T1 as u32, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        runtime.run().map_err(|_| crate::ExecutionError::ExceptionOrTrap())?;
        let a = if opcode.is_use_lo_hi_alu() {
            runtime.register(Register::LO)
        } else {
            runtime.register(Register::T2)
        };
        let hi = runtime.register(Register::HI);
        Ok((a, hi))
    }

    #[test]
    fn alu_matches_executor_over_random_operands() {
        let mut r = rng();
        let opcodes = [
            Opcode::ADD,
            Opcode::SUB,
            Opcode::SLL,
            Opcode::SRL,
            Opcode::SRA,
            Opcode::ROR,
            Opcode::MUL,
            Opcode::SLTU,
            Opcode::SLT,
            Opcode::MULT,
            Opcode::MULTU,
            Opcode::AND,
            Opcode::OR,
            Opcode::XOR,
            Opcode::NOR,
            Opcode::CLZ,
            Opcode::CLO,
        ];
        for opcode in opcodes {
            for _ in 0..200 {
                let b: u32 = r.gen();
                let c: u32 = r.gen();
                let (want_a, want_hi) = run_alu(opcode, b, c).unwrap();
                let (got_a, got_hi) = alu_compute(opcode, b, c).unwrap();
                assert_eq!(
                    (got_a, got_hi),
                    (want_a, want_hi),
                    "{opcode:?}(b={b:#x}, c={c:#x}): pure fn gave ({got_a:#x},{got_hi:#x}), \
                     Executor gave ({want_a:#x},{want_hi:#x})"
                );
            }
        }
    }

    #[test]
    fn alu_div_family_matches_executor_including_by_zero() {
        let mut r = rng();
        for opcode in [Opcode::DIV, Opcode::DIVU, Opcode::MOD, Opcode::MODU] {
            for _ in 0..200 {
                let b: u32 = r.gen();
                let c: u32 = r.gen();
                let want = run_alu(opcode, b, c);
                let got = alu_compute(opcode, b, c);
                match (want, got) {
                    (Ok((wa, wh)), Ok((ga, gh))) => assert_eq!(
                        (ga, gh),
                        (wa, wh),
                        "{opcode:?}(b={b:#x}, c={c:#x}) mismatch"
                    ),
                    (Err(_), Err(_)) => {}
                    (w, g) => panic!(
                        "{opcode:?}(b={b:#x}, c={c:#x}): Executor={w:?}, pure fn={g:?}"
                    ),
                }
            }
            // By-zero must trap identically.
            assert!(run_alu(opcode, 12345, 0).is_err());
            assert!(alu_compute(opcode, 12345, 0).is_err());
        }
    }

    #[test]
    fn branch_matches_executor() {
        let mut r = rng();
        let opcodes =
            [Opcode::BEQ, Opcode::BNE, Opcode::BGEZ, Opcode::BLEZ, Opcode::BGTZ, Opcode::BLTZ];
        for opcode in opcodes {
            for _ in 0..100 {
                // Bias towards equal/zero operands sometimes so BEQ/BGEZ/etc.'s edge conditions
                // actually get exercised, not just the generic "almost never equal" case.
                let src1: u32 = if r.gen_bool(0.3) { 0 } else { r.gen() };
                let src2: u32 = if r.gen_bool(0.3) { src1 } else { r.gen() };
                // Fixed, deliberately small offset: `next_pc` going into the branch is the delay
                // slot's own address, so `offset=8` lands exactly past the one-instruction marker
                // below (at "delay slot addr + 8") -- i.e. at the same resting place a
                // not-taken/no-marker-executed program would also reach, so this stays inside the
                // tiny program's bounds regardless of which way the branch actually goes and
                // never risks fetching past mapped memory.
                let offset: u32 = 8;

                // `BGEZ`/`BLEZ`/`BGTZ`/`BLTZ` are single-operand (see `only_one_operand`); encode
                // `src2` as an immediate `0` for those exactly as `branch_rr` does, and as a real
                // second register for BEQ/BNE.
                let single_operand = matches!(
                    opcode,
                    Opcode::BGEZ | Opcode::BLEZ | Opcode::BGTZ | Opcode::BLTZ
                );
                let mut instructions = vec![load_imm(T0, src1)];
                let (op_b, op_c, imm_c) = if single_operand {
                    (0u32, offset, true)
                } else {
                    instructions.push(load_imm(T1, src2));
                    (T1 as u32, offset, true)
                };
                instructions.push(Instruction::new(opcode, T0, op_b, op_c, false, imm_c));
                // Delay slot + a marker instruction so we can read back whether the branch
                // landed on the taken target (marker skipped) or fell through (marker runs).
                instructions.push(load_imm(T2, 0)); // delay slot: no-op-ish
                instructions.push(load_imm(T3, 0xDEAD_BEEF)); // marker: only runs if NOT taken

                let effective_src2 = if single_operand { 0 } else { src2 };
                let want_taken = {
                    // Reference: does the real Executor skip the marker?
                    let mut runtime = run(instructions.clone());
                    runtime.register(T3.into()) != 0xDEAD_BEEF
                };
                let got_taken = branch_taken(opcode, src1, effective_src2);
                assert_eq!(
                    got_taken, want_taken,
                    "{opcode:?}(src1={src1:#x}, src2={effective_src2:#x}) taken mismatch"
                );
            }
        }
    }

    #[test]
    fn branch_target_is_next_pc_plus_offset() {
        let mut r = rng();
        for _ in 0..200 {
            let next_pc: u32 = r.gen_range(0..0x1000) * 4;
            let offset: u32 = r.gen_range(0..0x1000) * 4;
            assert_eq!(branch_target(next_pc, offset), offset.wrapping_add(next_pc));
        }
    }

    #[test]
    fn jump_families_link_pc_is_next_pc_plus_4() {
        let mut r = rng();
        for _ in 0..100 {
            let next_pc: u32 = r.gen();
            let target: u32 = r.gen();
            assert_eq!(jump_jr_result(next_pc, target), (next_pc.wrapping_add(4), target));
            assert_eq!(jump_jumpi_result(next_pc, target), (next_pc.wrapping_add(4), target));
            let offset: u32 = r.gen();
            assert_eq!(
                jump_direct_result(next_pc, offset),
                (next_pc.wrapping_add(4), offset.wrapping_add(next_pc))
            );
        }
    }

    #[test]
    fn jr_matches_executor_lands_on_target_with_correct_link() {
        // JR $t3, $t0 where $t0 == 12, the address of the HALT sequence below: pc=0 loads the
        // target, pc=4 is the JR itself (`next_pc` going in is 4's own next_pc, i.e. the delay
        // slot's address, 8), pc=8 is the delay slot, and pc=12 is where a correctly-taken jump
        // must land. If the jump landed anywhere else, `runtime.run()` would run off the tiny
        // program and never reach the `SYSCALL` (the exact HALT-instruction sequence
        // `halt_only_program` uses), so a clean, non-panicking `.unwrap()` is itself part of the
        // assertion, not just the register checks below.
        let target = 12u32;
        let instructions = vec![
            load_imm(T0, target),
            Instruction::new(Opcode::Jump, T3, T0 as u32, 0, false, false),
            load_imm(T2, 0), // delay slot
            load_imm(Register::V0 as u8, 0), // pc=12: v0 = 0 (HALT syscall id)
            load_imm(Register::A0 as u8, 0), // pc=16: a0 = 0 (exit code)
            Instruction::new(Opcode::SYSCALL, Register::V0 as u8, Register::A0 as u8 as u32, 5, false, false),
        ];
        let mut runtime = run(instructions);

        // `next_pc` at JR-execution-time is the delay slot's own address (JR's pc + 4 == 8), not
        // JR's own pc (4) -- see the comment above.
        let (want_return, want_target) = jump_jr_result(8, target);
        assert_eq!(want_target, target);
        assert_eq!(runtime.register(T3.into()), want_return);
    }

    /// Runs `rd = prev_a; b = $t0; c = $t1; <opcode> rd, T0, T1` and returns `rd`'s final value,
    /// for `condmov_result` (MEQ/MNE)'s differential test.
    fn run_condmov(opcode: Opcode, prev_a: u32, b: u32, c: u32) -> u32 {
        const RD: u8 = Register::S0 as u8;
        let instructions = vec![
            load_imm(RD, prev_a),
            load_imm(T0, b),
            load_imm(T1, c),
            Instruction::new(opcode, RD, T0 as u32, T1 as u32, false, false),
        ];
        run(instructions).register(RD.into())
    }

    #[test]
    fn condmov_matches_executor() {
        let mut r = rng();
        for opcode in [Opcode::MEQ, Opcode::MNE] {
            for _ in 0..100 {
                let prev_a: u32 = r.gen();
                let b: u32 = r.gen();
                // Bias towards `c == 0` sometimes so both branches of the mov condition fire.
                let c: u32 = if r.gen_bool(0.3) { 0 } else { r.gen() };
                let want = run_condmov(opcode, prev_a, b, c);
                let got = condmov_result(opcode, prev_a, b, c);
                assert_eq!(got, want, "{opcode:?}(prev_a={prev_a:#x}, b={b:#x}, c={c:#x})");
            }
        }
    }

    #[test]
    fn wsbh_matches_executor() {
        let mut r = rng();
        for _ in 0..200 {
            let b: u32 = r.gen();
            const RD: u8 = Register::S0 as u8;
            let instructions = vec![
                load_imm(T0, b),
                Instruction::new(Opcode::WSBH, RD, T0 as u32, 0, false, true),
            ];
            let want = run(instructions).register(RD.into());
            assert_eq!(wsbh(b), want, "wsbh(b={b:#x})");
        }
    }

    #[test]
    fn sext_matches_executor() {
        let mut r = rng();
        for c in [0u32, 1u32] {
            for _ in 0..100 {
                let b: u32 = r.gen();
                const RD: u8 = Register::S0 as u8;
                let instructions = vec![
                    load_imm(T0, b),
                    Instruction::new(Opcode::SEXT, RD, T0 as u32, c, false, true),
                ];
                let want = run(instructions).register(RD.into());
                assert_eq!(sext(b, c), want, "sext(b={b:#x}, c={c})");
            }
        }
    }

    #[test]
    fn ext_matches_executor() {
        let mut r = rng();
        const RD: u8 = Register::S0 as u8;
        for _ in 0..300 {
            let b: u32 = r.gen();
            // Only `lsb + msbd < 32` is a legal encoding (see `ext`'s doc comment); build `c`
            // from independently-random `lsb`/`msbd` that satisfy it, rather than rejecting
            // random `c`s (which would mostly be illegal and rarely exercise real behavior).
            let lsb: u32 = r.gen_range(0..32);
            let msbd: u32 = r.gen_range(0..(32 - lsb));
            let c = (msbd << 5) | lsb;

            let instructions = vec![
                load_imm(T0, b),
                Instruction::new(Opcode::EXT, RD, T0 as u32, c, false, true),
            ];
            let want = run(instructions).register(RD.into());
            assert_eq!(ext(b, c).unwrap(), want, "ext(b={b:#x}, c={c:#x} lsb={lsb} msbd={msbd})");
        }
    }

    #[test]
    fn ins_matches_executor() {
        let mut r = rng();
        const RD: u8 = Register::S0 as u8;
        for _ in 0..300 {
            let a: u32 = r.gen(); // rd's pre-instruction value
            let b: u32 = r.gen();
            // Only `lsb <= msb` is a legal encoding (see `ins`'s doc comment).
            let lsb: u32 = r.gen_range(0..32);
            let msb: u32 = r.gen_range(lsb..32);
            let c = (msb << 5) | lsb;

            let instructions = vec![
                load_imm(RD, a),
                load_imm(T0, b),
                Instruction::new(Opcode::INS, RD, T0 as u32, c, false, true),
            ];
            let want = run(instructions).register(RD.into());
            assert_eq!(
                ins(a, b, c).unwrap(),
                want,
                "ins(a={a:#x}, b={b:#x}, c={c:#x} lsb={lsb} msb={msb})"
            );
        }
    }

    #[test]
    fn ext_rejects_illegal_encoding_like_executor() {
        // `lsb + msbd >= 32` (e.g. `lsb=31, msbd=31`) is undefined per the AIR constraint --
        // both the pure fn and a real `Executor` run must reject it identically.
        let c = (31u32 << 5) | 31u32;
        assert!(ext(0x1234, c).is_err());
        let instructions = vec![
            load_imm(T0, 0x1234),
            Instruction::new(Opcode::EXT, Register::S0 as u8, T0 as u32, c, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        assert!(runtime.run().is_err());
    }

    #[test]
    fn ins_rejects_illegal_encoding_like_executor() {
        // `msb < lsb` (e.g. `lsb=31, msb=0`) is undefined -- both must reject it identically.
        let c = (0u32 << 5) | 31u32;
        assert!(ins(0, 0x1234, c).is_err());
        let instructions = vec![
            load_imm(T0, 0x1234),
            Instruction::new(Opcode::INS, Register::S0 as u8, T0 as u32, c, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = Executor::new(program, ZKMCoreOpts::default());
        assert!(runtime.run().is_err());
    }

    #[test]
    fn teq_matches_executor() {
        let mut r = rng();
        for _ in 0..100 {
            let src1: u32 = r.gen();
            // Bias towards equal sometimes so the trap path actually fires.
            let src2: u32 = if r.gen_bool(0.3) { src1 } else { r.gen() };

            let want_err = src1 == src2;
            assert_eq!(teq(src1, src2).is_err(), want_err, "teq(src1={src1:#x}, src2={src2:#x})");

            // Cross-check against the real `Executor` too (rs=op_a, rt=op_b -- see `teq`'s doc
            // comment on `execute_teq`'s unusual operand encoding).
            let instructions = vec![
                load_imm(T0, src1),
                load_imm(T1, src2),
                Instruction::new(Opcode::TEQ, T0, T1 as u32, 0, false, false),
            ];
            let program = Program::new(instructions, 0, 0);
            let mut runtime = Executor::new(program, ZKMCoreOpts::default());
            assert_eq!(runtime.run().is_err(), want_err, "Executor teq trap mismatch");
        }
    }

    /// Runs `LO = lo; HI = hi; T0 = b; T1 = c; <opcode> LO, T0, T1` (op_a = LO by convention,
    /// matching real MADD-family encodings) and returns the resulting `(LO, HI)`, for the
    /// MADD-family differential tests. Per `execute_maddu`'s decode, `op_b` (here `T0`) supplies
    /// `rt` -> `b`, and `op_c` (here `T1`) supplies `rs` -> `c`.
    fn run_madd_family(opcode: Opcode, b: u32, c: u32, lo: u32, hi: u32) -> (u32, u32) {
        let instructions = vec![
            load_imm(Register::LO as u8, lo),
            load_imm(Register::HI as u8, hi),
            load_imm(T0, b),
            load_imm(T1, c),
            Instruction::new(opcode, Register::LO as u8, T0 as u32, T1 as u32, false, false),
        ];
        let mut runtime = run(instructions);
        (runtime.register(Register::LO), runtime.register(Register::HI))
    }

    #[test]
    fn madd_family_matches_executor() {
        let mut r = rng();
        let cases: [(Opcode, fn(u32, u32, u32, u32) -> (u32, u32)); 4] = [
            (Opcode::MADDU, maddu),
            (Opcode::MSUBU, msubu),
            (Opcode::MADD, madd),
            (Opcode::MSUB, msub),
        ];
        for (opcode, pure_fn) in cases {
            for _ in 0..150 {
                let b: u32 = r.gen();
                let c: u32 = r.gen();
                let lo: u32 = r.gen();
                let hi: u32 = r.gen();
                let want = run_madd_family(opcode, b, c, lo, hi);
                let got = pure_fn(b, c, lo, hi);
                assert_eq!(
                    got, want,
                    "{opcode:?}(b={b:#x}, c={c:#x}, lo={lo:#x}, hi={hi:#x})"
                );
            }
        }
    }

    // ---- CoreVM replay vs. the MinimalExecutor chunk it replays ----

    use crate::{
        minimal::MinimalExecutor,
        programs::tests::{fibonacci_program, halt_only_program, hello_world_program, simple_program},
    };

    /// Runs `program` on `MinimalExecutor` with a `max_trace_size` large enough that the whole
    /// run fits in a single `TraceChunk` (so `clk_end` is real program completion, not a
    /// buffer-full cutoff), then replays that one chunk on `CoreVM` and asserts they end at the
    /// identical `(pc, clk, registers)`.
    fn assert_corevm_replay_matches_minimal(program: impl Fn() -> Program, name: &str) {
        let mut minimal = MinimalExecutor::new(Arc::new(program()), u64::MAX / 2);
        let chunk = minimal
            .try_execute_chunk()
            .unwrap()
            .expect("program must produce at least one chunk");
        assert!(
            minimal.try_execute_chunk().unwrap().is_none(),
            "{name}: expected the whole run to fit in a single chunk"
        );

        let max_syscall_cycles = minimal.max_syscall_cycles();
        let mut core = CoreVM::new(&chunk, Arc::new(program()), max_syscall_cycles);
        let status = core.execute().unwrap();
        assert_eq!(status, CoreVMStatus::Done, "{name}: CoreVM should reach real completion");
        assert_eq!(core.registers(), minimal.registers(), "{name}: register mismatch");
        assert_eq!(core.pc(), minimal.pc(), "{name}: pc mismatch");
        assert_eq!(core.clk(), minimal.clk(), "{name}: clk mismatch");
    }

    #[test]
    fn corevm_replay_matches_minimal_simple_program() {
        assert_corevm_replay_matches_minimal(simple_program, "simple_program");
    }

    #[test]
    fn corevm_replay_matches_minimal_halt_only_program() {
        assert_corevm_replay_matches_minimal(halt_only_program, "halt_only_program");
    }

    #[test]
    fn corevm_replay_matches_minimal_fibonacci_real_elf() {
        assert_corevm_replay_matches_minimal(fibonacci_program, "fibonacci_program");
    }

    #[test]
    fn corevm_replay_matches_minimal_hello_world_real_elf() {
        assert_corevm_replay_matches_minimal(hello_world_program, "hello_world_program");
    }

    /// A chunk boundary mid-way through a program (`TraceEnd`, not `Done`) must also replay
    /// cleanly and consistently -- exercises `CoreVM` picking up a chunk that doesn't end in a
    /// real halt.
    #[test]
    fn corevm_replay_matches_minimal_across_a_mid_program_chunk_boundary() {
        let mut minimal = MinimalExecutor::new(Arc::new(fibonacci_program()), 4);
        let chunk = minimal.try_execute_chunk().unwrap().expect("expected at least one chunk");

        let mut core =
            CoreVM::new(&chunk, Arc::new(fibonacci_program()), minimal.max_syscall_cycles());
        let status = core.execute().unwrap();
        assert_eq!(status, CoreVMStatus::TraceEnd, "expected a mid-program cutoff, not real completion");
        assert_eq!(core.registers(), minimal.registers(), "register mismatch at chunk boundary");
        assert_eq!(core.pc(), minimal.pc(), "pc mismatch at chunk boundary");
        assert_eq!(core.clk(), minimal.clk(), "clk mismatch at chunk boundary");
    }

    /// Known-answer test for `sha256_compress`/`sha256_extend_word`: hashes the empty message
    /// (a single padded block) and checks the result against the standard SHA-256 digest of `""`.
    /// The golden-comparison tests in `tracing.rs` only compare event *counts*, not computed
    /// values, so this is the only check that the arithmetic itself is correct.
    #[test]
    fn sha256_compress_and_extend_word_match_known_answer_for_empty_message() {
        let iv: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let mut block = [0u32; 16];
        block[0] = 0x8000_0000; // padding: a single `1` bit, then zeros.
        // `block[15]` (the 64-bit message length in bits) is already `0` for the empty message.

        let mut w = [0u32; 64];
        w[..16].copy_from_slice(&block);
        for i in 16..64 {
            w[i] = sha256_extend_word(w[i - 15], w[i - 2], w[i - 16], w[i - 7]);
        }

        let digest = sha256_compress(iv, &w);
        let expected: [u32; 8] = [
            0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b,
            0x7852b855,
        ];
        assert_eq!(digest, expected, "SHA-256(\"\") mismatch");
    }
}
