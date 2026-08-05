use enum_map::EnumMap;
use hashbrown::HashMap;
use p3_koala_bear::KoalaBear;

use crate::{
    events::{PrecompileEvents, NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC},
    syscalls::SyscallCode,
    ExecutionRecord, MipsAirId, Opcode, NUM_REGISTERS,
};

const BYTE_NUM_ROWS: u64 = 1 << 16;

/// Maps each precompile [`SyscallCode`] that [`PrecompileEvents`] actually stores events under to
/// the [`MipsAirId`] chip(s) whose row count it drives one-for-one.
///
/// A handful of `SyscallCode`s are coalesced into one bucket at event-recording time (see e.g.
/// `FpOpSyscall::execute`, which files `*_FP_SUB`/`*_FP_MUL` events under the `*_FP_ADD` key, and
/// `Fp2AddSubAssignSyscall`, which files `*_FP2_SUB` under `*_FP2_ADD`), so counting the listed
/// key alone already covers every op it's coalesced with -- this table only needs the key each
/// family is actually stored under, not every `SyscallCode` variant. `SHA_EXTEND`/`SHA_COMPRESS`
/// each drive two chips (a `*Control` bracket chip, one row per event, plus the round-level
/// worker chip); every other precompile family drives exactly one chip.
///
/// `SysLinux` isn't included here: every individual Linux syscall (`SYS_BRK`, `SYS_READ`, ...)
/// already files its event under the single `SyscallCode::SYS_LINUX` key, so it's handled as a
/// regular one-chip entry via that key like everything else.
pub(crate) const PRECOMPILE_AIR_IDS: &[(SyscallCode, &[MipsAirId])] = &[
    (SyscallCode::SHA_EXTEND, &[MipsAirId::ShaExtendControl, MipsAirId::ShaExtend]),
    (SyscallCode::SHA_COMPRESS, &[MipsAirId::ShaCompressControl, MipsAirId::ShaCompress]),
    (SyscallCode::ED_ADD, &[MipsAirId::EdAddAssign]),
    (SyscallCode::ED_DECOMPRESS, &[MipsAirId::EdDecompress]),
    (SyscallCode::KECCAK_SPONGE, &[MipsAirId::KeccakSponge]),
    (SyscallCode::SECP256K1_ADD, &[MipsAirId::Secp256k1AddAssign]),
    (SyscallCode::SECP256K1_DOUBLE, &[MipsAirId::Secp256k1DoubleAssign]),
    (SyscallCode::SECP256K1_DECOMPRESS, &[MipsAirId::Secp256k1Decompress]),
    (SyscallCode::SECP256R1_ADD, &[MipsAirId::Secp256r1AddAssign]),
    (SyscallCode::SECP256R1_DOUBLE, &[MipsAirId::Secp256r1DoubleAssign]),
    (SyscallCode::SECP256R1_DECOMPRESS, &[MipsAirId::Secp256r1Decompress]),
    (SyscallCode::BN254_ADD, &[MipsAirId::Bn254AddAssign]),
    (SyscallCode::BN254_DOUBLE, &[MipsAirId::Bn254DoubleAssign]),
    (SyscallCode::BLS12381_ADD, &[MipsAirId::Bls12381AddAssign]),
    (SyscallCode::BLS12381_DOUBLE, &[MipsAirId::Bls12381DoubleAssign]),
    (SyscallCode::BLS12381_DECOMPRESS, &[MipsAirId::Bls12381Decompress]),
    (SyscallCode::UINT256_MUL, &[MipsAirId::Uint256MulMod]),
    (SyscallCode::U256XU2048_MUL, &[MipsAirId::U256XU2048Mul]),
    (SyscallCode::POSEIDON2_PERMUTE, &[MipsAirId::Poseidon2Permute]),
    (SyscallCode::BLS12381_FP_ADD, &[MipsAirId::Bls12381FpOpAssign]),
    (SyscallCode::BLS12381_FP2_ADD, &[MipsAirId::Bls12831Fp2AddSubAssign]),
    (SyscallCode::BLS12381_FP2_MUL, &[MipsAirId::Bls12831Fp2MulAssign]),
    (SyscallCode::BN254_FP_ADD, &[MipsAirId::Bn254FpOpAssign]),
    (SyscallCode::BN254_FP2_ADD, &[MipsAirId::Bn254Fp2AddSubAssign]),
    (SyscallCode::BN254_FP2_MUL, &[MipsAirId::Bn254Fp2MulAssign]),
    (SyscallCode::SYS_LINUX, &[MipsAirId::SysLinux]),
];

/// Adds every precompile chip's contribution to `cells`, in the same `next_power_of_two(event
/// count) * cost_per_event` unit system the core-chip terms above/below use.
///
/// Unlike the core-chip terms (which project a worst case from `num_events_per_air`, a count
/// derived from opcode counters *before* the next `shape_check_frequency`-cycle window has
/// executed), precompile counts are read directly off the live shard's `ExecutionRecord` --
/// exact, not projected. That's fine here: `estimate_mips_lde_size` is re-run every
/// `shape_check_frequency` cycles regardless, so the (at most one, since a syscall retires at
/// most once per cycle) precompile event that could fire in between two checks is already well
/// within the slack every other chip's periodic-recheck cadence tolerates.
///
/// `O(number of precompile families)` (~28 fixed entries), not `O(number of precompile events)`:
/// each family costs one `HashMap` lookup into `PrecompileEvents` plus a `Vec::len()`, so this
/// stays cheap enough for the hot `inc_shard_if_need` path that calls it every
/// `shape_check_frequency` cycles.
fn add_precompile_cells(
    cells: &mut u64,
    precompile_events: &PrecompileEvents,
    costs_per_air: &HashMap<MipsAirId, u64>,
) {
    let mut total_precompile_events: u64 = 0;
    for (syscall_code, air_ids) in PRECOMPILE_AIR_IDS {
        let Some(events) = precompile_events.get_events(*syscall_code) else {
            continue;
        };
        let count = events.len() as u64;
        if count == 0 {
            continue;
        }
        total_precompile_events += count;
        let padded = count.next_power_of_two();
        for air_id in *air_ids {
            *cells += padded * costs_per_air[air_id];
        }
    }
    // The syscall-precompile dispatch chip: one row per precompile event, across every family
    // (see `SyscallChip::generate_trace`'s `SyscallShardKind::Precompile` arm, which iterates
    // `precompile_events.all_events()`).
    if total_precompile_events > 0 {
        *cells += total_precompile_events.next_power_of_two()
            * costs_per_air[&MipsAirId::SyscallPrecompile];
    }
}

/// Returns `true` for the `MipsAirId` variants covered exactly (not just conservatively) by
/// [`estimate_record_trace_bytes`]'s per-chip event counting.
const fn is_core_air(id: MipsAirId) -> bool {
    matches!(
        id,
        MipsAirId::Program
            | MipsAirId::DivRem
            | MipsAirId::Add
            | MipsAirId::Addi
            | MipsAirId::AddNoop
            | MipsAirId::AluX0
            | MipsAirId::Sub
            | MipsAirId::Bitwise
            | MipsAirId::Mul
            | MipsAirId::ShiftRight
            | MipsAirId::ShiftLeft
            | MipsAirId::Lui
            | MipsAirId::Lt
            | MipsAirId::Slti
            | MipsAirId::CloClz
            | MipsAirId::Branch
            | MipsAirId::Jump
            | MipsAirId::Jumpi
            | MipsAirId::JumpDirect
            | MipsAirId::SyscallInstrs
            | MipsAirId::SyscallCore
            | MipsAirId::LoadWord
            | MipsAirId::LoadX0
            | MipsAirId::StoreWord
            | MipsAirId::LoadByte
            | MipsAirId::LoadHalf
            | MipsAirId::LoadWordUnaligned
            | MipsAirId::StoreByte
            | MipsAirId::StoreHalf
            | MipsAirId::StoreWordUnaligned
            | MipsAirId::StoreConditional
            | MipsAirId::Sext
            | MipsAirId::Ins
            | MipsAirId::Ext
            | MipsAirId::Maddsub
            | MipsAirId::Teq
            | MipsAirId::MemoryGlobalInit
            | MipsAirId::MemoryGlobalFinalize
            | MipsAirId::MemoryLocal
            | MipsAirId::Global
            | MipsAirId::Byte
            | MipsAirId::MovCond
            | MipsAirId::StateBump
            | MipsAirId::MemoryBump
    )
}

/// Estimates a shard's real, materialized main-trace commitment size in bytes, from a
/// completed [`ExecutionRecord`]'s exact per-chip event counts -- unlike
/// [`estimate_mips_event_counts`]/[`pad_mips_event_counts`], which project a worst case
/// *before* a shard has finished executing, this sums real `Vec::len()`s after the fact, so it
/// carries no shape-check-frequency slack for the core chips.
///
/// Precompile chips (SHA/Keccak/EC/BN254/BLS12381/...) are not covered by an exact
/// per-syscall row mapping -- Ziren has no `SyscallCode -> MipsAirId` table to reuse for that --
/// so they're covered by one conservative term instead: total precompile event count times the
/// single most expensive precompile chip's per-row cost, then a flat safety margin over the
/// whole total. This keeps the estimate from ever *undercounting* real committed size (the
/// property an admission-control budget needs), at the cost of being loose for precompile-heavy
/// shards.
#[must_use]
pub fn estimate_record_trace_bytes(
    record: &ExecutionRecord,
    costs_per_air: &HashMap<MipsAirId, u64>,
) -> u64 {
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&MipsAirId::Byte];
    // The Program chip's preprocessed trace is padded to the program's real instruction count
    // (see `ProgramChip::generate_preprocessed_trace`), not some worst-case ceiling -- and
    // unlike the event-count estimators below, this function already has the real program in
    // hand via `record.program`, so there's no need to guess.
    cells += (record.program.instructions.len() as u64).next_power_of_two()
        * costs_per_air[&MipsAirId::Program];

    let mut add_chip_cells = |air: MipsAirId, count: usize| {
        cells += (count as u64).next_power_of_two() * costs_per_air[&air];
    };
    add_chip_cells(MipsAirId::Add, record.add_events.len());
    add_chip_cells(MipsAirId::Addi, record.addi_events.len());
    add_chip_cells(MipsAirId::AddNoop, record.add_noop_events.len());
    add_chip_cells(MipsAirId::AluX0, record.alu_x0_events.len());
    add_chip_cells(MipsAirId::Sub, record.sub_events.len());
    add_chip_cells(MipsAirId::Mul, record.mul_events.len());
    add_chip_cells(MipsAirId::Bitwise, record.bitwise_events.len());
    add_chip_cells(MipsAirId::ShiftLeft, record.shift_left_events.len());
    add_chip_cells(MipsAirId::Lui, record.lui_events.len());
    add_chip_cells(MipsAirId::ShiftRight, record.shift_right_events.len());
    add_chip_cells(MipsAirId::DivRem, record.divrem_events.len());
    add_chip_cells(MipsAirId::Lt, record.lt_events.len());
    add_chip_cells(MipsAirId::Slti, record.slti_events.len());
    add_chip_cells(MipsAirId::CloClz, record.cloclz_events.len());
    add_chip_cells(MipsAirId::LoadWord, record.load_word_events.len());
    add_chip_cells(MipsAirId::LoadX0, record.load_x0_events.len());
    add_chip_cells(MipsAirId::StoreWord, record.store_word_events.len());
    add_chip_cells(MipsAirId::LoadByte, record.load_byte_events.len());
    add_chip_cells(MipsAirId::LoadHalf, record.load_half_events.len());
    add_chip_cells(MipsAirId::LoadWordUnaligned, record.load_word_unaligned_events.len());
    add_chip_cells(MipsAirId::StoreByte, record.store_byte_events.len());
    add_chip_cells(MipsAirId::StoreHalf, record.store_half_events.len());
    add_chip_cells(MipsAirId::StoreWordUnaligned, record.store_word_unaligned_events.len());
    add_chip_cells(MipsAirId::StoreConditional, record.store_conditional_events.len());
    add_chip_cells(MipsAirId::Branch, record.branch_events.len());
    add_chip_cells(MipsAirId::Jump, record.jump_events.len());
    add_chip_cells(MipsAirId::Jumpi, record.jumpi_events.len());
    add_chip_cells(MipsAirId::JumpDirect, record.jumpdirect_events.len());
    add_chip_cells(MipsAirId::MovCond, record.movcond_events.len());
    add_chip_cells(MipsAirId::Sext, record.sext_events.len());
    add_chip_cells(MipsAirId::Ins, record.ins_events.len());
    add_chip_cells(MipsAirId::Ext, record.ext_events.len());
    add_chip_cells(MipsAirId::Maddsub, record.maddsub_events.len());
    add_chip_cells(MipsAirId::Teq, record.teq_events.len());
    add_chip_cells(MipsAirId::MemoryGlobalInit, record.global_memory_initialize_events.len());
    add_chip_cells(MipsAirId::MemoryGlobalFinalize, record.global_memory_finalize_events.len());
    add_chip_cells(MipsAirId::MemoryLocal, record.cpu_local_memory_access.len());
    add_chip_cells(MipsAirId::SyscallInstrs, record.syscall_events.len());
    add_chip_cells(MipsAirId::SyscallCore, record.syscall_events.len());
    add_chip_cells(MipsAirId::Global, record.global_lookup_events.len());
    add_chip_cells(MipsAirId::StateBump, record.bump_clk_high_events.len());
    add_chip_cells(MipsAirId::MemoryBump, record.bump_memory_events.len());

    let precompile_event_count: usize =
        record.precompile_events.iter().map(|(_, events)| events.len()).sum();
    if precompile_event_count > 0 {
        let max_precompile_cost = costs_per_air
            .iter()
            .filter(|(air, _)| !is_core_air(**air))
            .map(|(_, cost)| *cost)
            .max()
            .unwrap_or(0);
        cells += (precompile_event_count as u64).next_power_of_two() * max_precompile_cost;
    }

    // Safety multiplier: the budget this feeds needs to bound *peak resident memory while a
    // shard is actively being committed/proved*, not just the padded main traces' own storage
    // size. `commit_traces` keeps its main traces, an interleaved-MLE copy, and a 4x-blown-up
    // (`log_blowup`) FRI codeword over the extension field all resident *for the rest of the
    // shard's proof* (the codeword isn't freed until FRI's query-opening phase, at the very end
    // of `prove_evaluation_claims`); LogUp-GKR and zerocheck each stack a further transient
    // extension-field-width peak on top of that already-resident baseline before their own
    // buffers free. 10x approximates that combined peak; it must never undercount, since the
    // budget this feeds has no other signal for how large a shard actually gets while active.
    cells *= 10;

    cells * ((core::mem::size_of::<KoalaBear>() << 1) as u64)
}

/// Estimates the LDE area.
///
/// `program_size` is the calling program's real, padded (next-power-of-two) instruction count --
/// see [`estimate_record_trace_bytes`]'s doc comment on the Program chip contribution for why
/// this must be the real size rather than a worst-case ceiling.
///
/// `precompile_events` is the live shard-in-progress's exact precompile events (see
/// [`add_precompile_cells`]'s doc comment for why these are read directly rather than projected
/// like `num_events_per_air`). Every precompile/syscall-family `MipsAirId` (SHA/Keccak/
/// Poseidon2/Edwards/Weierstrass/BN254/BLS12-381/Uint256/SysLinux/...) is accounted for here --
/// previously this function had no precompile terms at all, so a shard whose *CPU-chip* cost
/// looked small could still carry an unbounded amount of un-budgeted precompile row weight.
#[must_use]
pub fn estimate_mips_lde_size(
    num_events_per_air: EnumMap<MipsAirId, u64>,
    costs_per_air: &HashMap<MipsAirId, u64>,
    program_size: u64,
    precompile_events: &PrecompileEvents,
) -> u64 {
    // Compute the byte chip contribution.
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&MipsAirId::Byte];

    // Compute the program chip contribution.
    cells += program_size * costs_per_air[&MipsAirId::Program];

    // Compute the add chip contribution.
    cells += (num_events_per_air[MipsAirId::Add]).next_power_of_two()
        * costs_per_air[&MipsAirId::Add];

    // Compute the addi chip contribution.
    cells += (num_events_per_air[MipsAirId::Addi]).next_power_of_two()
        * costs_per_air[&MipsAirId::Addi];

    // Compute the add-noop (SYNC/Pref) chip contribution.
    cells += (num_events_per_air[MipsAirId::AddNoop]).next_power_of_two()
        * costs_per_air[&MipsAirId::AddNoop];

    // Compute the shared add/sub-to-register-0 chip contribution.
    cells += (num_events_per_air[MipsAirId::AluX0]).next_power_of_two()
        * costs_per_air[&MipsAirId::AluX0];

    // Compute the sub chip contribution.
    cells += (num_events_per_air[MipsAirId::Sub]).next_power_of_two()
        * costs_per_air[&MipsAirId::Sub];

    // Compute the mul chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Mul]).next_power_of_two() * costs_per_air[&MipsAirId::Mul];

    // Compute the bitwise chip contribution.
    cells += (num_events_per_air[MipsAirId::Bitwise]).next_power_of_two()
        * costs_per_air[&MipsAirId::Bitwise];

    // Compute the shift left chip contribution.
    cells += (num_events_per_air[MipsAirId::ShiftLeft]).next_power_of_two()
        * costs_per_air[&MipsAirId::ShiftLeft];

    // Compute the LUI chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Lui]).next_power_of_two() * costs_per_air[&MipsAirId::Lui];

    // Compute the shift right chip contribution.
    cells += (num_events_per_air[MipsAirId::ShiftRight]).next_power_of_two()
        * costs_per_air[&MipsAirId::ShiftRight];

    // Compute the divrem chip contribution.
    cells += (num_events_per_air[MipsAirId::DivRem]).next_power_of_two()
        * costs_per_air[&MipsAirId::DivRem];

    // Compute the lt chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Lt]).next_power_of_two() * costs_per_air[&MipsAirId::Lt];

    // Compute the slti (immediate-form SLT/SLTU) chip contribution.
    cells += (num_events_per_air[MipsAirId::Slti]).next_power_of_two()
        * costs_per_air[&MipsAirId::Slti];

    // Compute the memory local chip contribution.
    cells += (num_events_per_air[MipsAirId::MemoryLocal]).next_power_of_two()
        * costs_per_air[&MipsAirId::MemoryLocal];

    // Compute the branch chip contribution.
    cells += (num_events_per_air[MipsAirId::Branch]).next_power_of_two()
        * costs_per_air[&MipsAirId::Branch];

    // Compute the jump chips' contribution.
    cells +=
        (num_events_per_air[MipsAirId::Jump]).next_power_of_two() * costs_per_air[&MipsAirId::Jump];
    cells += (num_events_per_air[MipsAirId::Jumpi]).next_power_of_two()
        * costs_per_air[&MipsAirId::Jumpi];
    cells += (num_events_per_air[MipsAirId::JumpDirect]).next_power_of_two()
        * costs_per_air[&MipsAirId::JumpDirect];

    // Compute the SyscallInstruction chip contribution.
    cells += (num_events_per_air[MipsAirId::SyscallInstrs]).next_power_of_two()
        * costs_per_air[&MipsAirId::SyscallInstrs];

    // Compute the LoadWord chip contribution.
    cells += (num_events_per_air[MipsAirId::LoadWord]).next_power_of_two()
        * costs_per_air[&MipsAirId::LoadWord];

    // Compute the LoadX0 (real `lw $zero, ...`) chip contribution.
    cells += (num_events_per_air[MipsAirId::LoadX0]).next_power_of_two()
        * costs_per_air[&MipsAirId::LoadX0];

    // Compute the StoreWord chip contribution.
    cells += (num_events_per_air[MipsAirId::StoreWord]).next_power_of_two()
        * costs_per_air[&MipsAirId::StoreWord];

    // Compute the LoadByte chip contribution.
    cells += (num_events_per_air[MipsAirId::LoadByte]).next_power_of_two()
        * costs_per_air[&MipsAirId::LoadByte];

    // Compute the LoadHalf chip contribution.
    cells += (num_events_per_air[MipsAirId::LoadHalf]).next_power_of_two()
        * costs_per_air[&MipsAirId::LoadHalf];

    // Compute the LoadWordUnaligned chip contribution.
    cells += (num_events_per_air[MipsAirId::LoadWordUnaligned]).next_power_of_two()
        * costs_per_air[&MipsAirId::LoadWordUnaligned];

    // Compute the StoreByte chip contribution.
    cells += (num_events_per_air[MipsAirId::StoreByte]).next_power_of_two()
        * costs_per_air[&MipsAirId::StoreByte];

    // Compute the StoreHalf chip contribution.
    cells += (num_events_per_air[MipsAirId::StoreHalf]).next_power_of_two()
        * costs_per_air[&MipsAirId::StoreHalf];

    // Compute the StoreWordUnaligned chip contribution.
    cells += (num_events_per_air[MipsAirId::StoreWordUnaligned]).next_power_of_two()
        * costs_per_air[&MipsAirId::StoreWordUnaligned];

    // Compute the StoreConditional chip contribution.
    cells += (num_events_per_air[MipsAirId::StoreConditional]).next_power_of_two()
        * costs_per_air[&MipsAirId::StoreConditional];

    // Compute the Sext chip contribution.
    cells += (num_events_per_air[MipsAirId::Sext]).next_power_of_two()
        * costs_per_air[&MipsAirId::Sext];

    // Compute the Ins chip contribution.
    cells += (num_events_per_air[MipsAirId::Ins]).next_power_of_two()
        * costs_per_air[&MipsAirId::Ins];

    // Compute the Ext chip contribution.
    cells += (num_events_per_air[MipsAirId::Ext]).next_power_of_two()
        * costs_per_air[&MipsAirId::Ext];

    // Compute the Maddsub chip contribution.
    cells += (num_events_per_air[MipsAirId::Maddsub]).next_power_of_two()
        * costs_per_air[&MipsAirId::Maddsub];

    // Compute the Teq chip contribution.
    cells += (num_events_per_air[MipsAirId::Teq]).next_power_of_two()
        * costs_per_air[&MipsAirId::Teq];

    // Compute the cloclz chip contribution.
    cells += (num_events_per_air[MipsAirId::CloClz]).next_power_of_two()
        * costs_per_air[&MipsAirId::CloClz];

    // Compute the syscall core chip contribution.
    cells += (num_events_per_air[MipsAirId::SyscallCore]).next_power_of_two()
        * costs_per_air[&MipsAirId::SyscallCore];

    // Compute the movcond chip contribution.
    cells += (num_events_per_air[MipsAirId::MovCond]).next_power_of_two()
        * costs_per_air[&MipsAirId::MovCond];

    // Compute the global chip contribution.
    cells += (num_events_per_air[MipsAirId::Global]).next_power_of_two()
        * costs_per_air[&MipsAirId::Global];

    // Compute the state bump chip contribution.
    cells += (num_events_per_air[MipsAirId::StateBump]).next_power_of_two()
        * costs_per_air[&MipsAirId::StateBump];

    // Compute the memory bump chip contribution.
    cells += (num_events_per_air[MipsAirId::MemoryBump]).next_power_of_two()
        * costs_per_air[&MipsAirId::MemoryBump];

    // Compute every precompile/syscall-family chip's contribution (SHA/Keccak/Poseidon2/
    // Edwards/Weierstrass/BN254/BLS12-381/Uint256/SysLinux/...) -- see `add_precompile_cells`'s
    // doc comment.
    add_precompile_cells(&mut cells, precompile_events, costs_per_air);

    cells * ((core::mem::size_of::<KoalaBear>() << 1) as u64)
}

/// Estimate
/// Maps the opcode counts to the number of events in each air.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn estimate_mips_event_counts(
    touched_addresses: u64,
    syscalls_sent: u64,
    addi_events: u64,
    add_noop_events: u64,
    add_x0_events: u64,
    sub_x0_events: u64,
    slt_i_events: u64,
    sltu_i_events: u64,
    slt_x0_events: u64,
    sltu_x0_events: u64,
    bitwise_x0_events: u64,
    shift_right_x0_events: u64,
    lui_events: u64,
    shift_left_x0_events: u64,
    mul_x0_events: u64,
    movcond_x0_events: u64,
    divrem_x0_events: u64,
    cloclz_x0_events: u64,
    ins_x0_events: u64,
    ext_x0_events: u64,
    sext_x0_events: u64,
    load_x0_events: u64,
    opcode_counts: EnumMap<Opcode, u64>,
) -> EnumMap<MipsAirId, u64> {
    let mut events_counts: EnumMap<MipsAirId, u64> = EnumMap::default();
    // Compute the number of events in the add chip. `opcode_counts[Opcode::ADD]` mixes
    // register-form ADD, immediate-form ADDI/ADDIU, fully-immediate SYNC/Pref, register-form
    // ADD with `op_a==0`, and register-form internal dependency rows from other chips --
    // `addi_events`/`add_noop_events`/`add_x0_events` isolate the immediate-form/fully-immediate/
    // zero-destination counts (see `AddiChip`/`AddNoopChip`/`AluX0Chip`'s doc comments), so
    // they're subtracted out here and counted on their own lines below.
    events_counts[MipsAirId::Add] =
        opcode_counts[Opcode::ADD] - addi_events - add_noop_events - add_x0_events;

    // Compute the number of events in the addi chip.
    events_counts[MipsAirId::Addi] = addi_events;

    // Compute the number of events in the add-noop (SYNC/Pref) chip.
    events_counts[MipsAirId::AddNoop] = add_noop_events;

    // Compute the number of events in the sub chip. MIPS has no SUBI, so every SUB opcode
    // occurrence belongs here, except a real SUB with `op_a==0` (isolated via `sub_x0_events`,
    // see `AluX0Chip`'s doc comment).
    events_counts[MipsAirId::Sub] = opcode_counts[Opcode::SUB] - sub_x0_events;

    // Compute the number of events in the shared
    // add/sub/lt/bitwise/shift-right/shift-left/mul-to-register-0 chip.
    events_counts[MipsAirId::AluX0] = add_x0_events
        + sub_x0_events
        + slt_x0_events
        + sltu_x0_events
        + bitwise_x0_events
        + shift_right_x0_events
        + shift_left_x0_events
        + mul_x0_events
        + movcond_x0_events
        + divrem_x0_events
        + cloclz_x0_events
        + ins_x0_events
        + ext_x0_events
        + sext_x0_events;

    // Compute the number of events in the mul chip. `opcode_counts[Opcode::MUL]` includes real,
    // retired `op_a==0` rows too (isolated via `mul_x0_events`, see `AluX0Chip`'s doc comment);
    // MULT/MULTU always decode with `op_a=32`, so they never need this split.
    events_counts[MipsAirId::Mul] = opcode_counts[Opcode::MUL] + opcode_counts[Opcode::MULT]
        + opcode_counts[Opcode::MULTU]
        - mul_x0_events;

    // Compute the number of events in the bitwise chip. `opcode_counts[Opcode::XOR]`/`[OR]`/
    // `[AND]`/`[NOR]` include real, retired `op_a==0` rows too (isolated via `bitwise_x0_events`,
    // see `AluX0Chip`'s doc comment), so that's subtracted out here.
    events_counts[MipsAirId::Bitwise] = opcode_counts[Opcode::XOR]
        + opcode_counts[Opcode::OR]
        + opcode_counts[Opcode::AND]
        + opcode_counts[Opcode::NOR]
        - bitwise_x0_events;

    // Compute the number of events in the shift left chip. `opcode_counts[Opcode::SLL]` mixes
    // register-form SLL/SLLV, LUI (isolated via `lui_events`, see `LuiChip`'s doc comment), and
    // zero-destination SLL/SLLV (isolated via `shift_left_x0_events`, see `AluX0Chip`'s doc
    // comment), so those are subtracted out here and counted on their own lines.
    events_counts[MipsAirId::ShiftLeft] =
        opcode_counts[Opcode::SLL] - lui_events - shift_left_x0_events;

    // Compute the number of events in the LUI chip.
    events_counts[MipsAirId::Lui] = lui_events;

    // Compute the number of events in the shift right chip. `opcode_counts[Opcode::SRL]`/`[SRA]`/
    // `[ROR]` include real, retired `op_a==0` rows too (isolated via `shift_right_x0_events`, see
    // `AluX0Chip`'s doc comment), so that's subtracted out here.
    events_counts[MipsAirId::ShiftRight] = opcode_counts[Opcode::SRL]
        + opcode_counts[Opcode::SRA]
        + opcode_counts[Opcode::ROR]
        - shift_right_x0_events;

    // Compute the number of events in the divrem chip. `opcode_counts[Opcode::MOD]`/`[MODU]`
    // include real, retired `op_a==0` rows too (isolated via `divrem_x0_events`, see
    // `AluX0Chip`'s doc comment); DIV/DIVU always decode with `op_a=32`, so they never need this
    // split.
    events_counts[MipsAirId::DivRem] = opcode_counts[Opcode::DIV]
        + opcode_counts[Opcode::DIVU]
        + opcode_counts[Opcode::MOD]
        + opcode_counts[Opcode::MODU]
        - divrem_x0_events;

    // Compute the number of events in the lt chip. `opcode_counts[Opcode::SLT]`/`[Opcode::SLTU]`
    // mix register-form SLT/SLTU, immediate-form SLTI/SLTIU, register-form SLT/SLTU with
    // `op_a==0`, and register-form internal dependency rows from other chips --
    // `slt_i_events`/`sltu_i_events`/`slt_x0_events`/`sltu_x0_events` isolate the immediate-form/
    // zero-destination counts (see `SltiChip`/`AluX0Chip`'s doc comments), so they're subtracted
    // out here and counted on their own lines below.
    events_counts[MipsAirId::Lt] = opcode_counts[Opcode::SLT] + opcode_counts[Opcode::SLTU]
        - slt_i_events
        - sltu_i_events
        - slt_x0_events
        - sltu_x0_events;

    // Compute the number of events in the slti (immediate-form SLT/SLTU) chip.
    events_counts[MipsAirId::Slti] = slt_i_events + sltu_i_events;

    // Compute the number of events in the memory local chip.
    events_counts[MipsAirId::MemoryLocal] =
        touched_addresses.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64);

    // Compute the number of events in the branch chip.
    events_counts[MipsAirId::Branch] = opcode_counts[Opcode::BEQ]
        + opcode_counts[Opcode::BNE]
        + opcode_counts[Opcode::BGTZ]
        + opcode_counts[Opcode::BGEZ]
        + opcode_counts[Opcode::BLTZ]
        + opcode_counts[Opcode::BLEZ];

    // Compute the number of events in the jump chips.
    events_counts[MipsAirId::Jump] = opcode_counts[Opcode::Jump];
    events_counts[MipsAirId::Jumpi] = opcode_counts[Opcode::Jumpi];
    events_counts[MipsAirId::JumpDirect] = opcode_counts[Opcode::JumpDirect];

    // Compute the number of events in the LoadWord/LoadX0/StoreWord chips.
    // `opcode_counts[Opcode::LW] + opcode_counts[Opcode::LL]` mixes real non-zero-destination
    // LW/LL with real `lw $zero, ...`/`ll $zero, ...` -- `load_x0_events` isolates the latter
    // (see `LoadX0Chip`'s doc comment).
    events_counts[MipsAirId::LoadWord] =
        opcode_counts[Opcode::LW] + opcode_counts[Opcode::LL] - load_x0_events;
    events_counts[MipsAirId::LoadX0] = load_x0_events;
    events_counts[MipsAirId::StoreWord] = opcode_counts[Opcode::SW];

    // Compute the number of events in the LoadByte chip.
    events_counts[MipsAirId::LoadByte] = opcode_counts[Opcode::LB] + opcode_counts[Opcode::LBU];

    // Compute the number of events in the LoadHalf chip.
    events_counts[MipsAirId::LoadHalf] = opcode_counts[Opcode::LH] + opcode_counts[Opcode::LHU];

    // Compute the number of events in the LoadWordUnaligned chip.
    events_counts[MipsAirId::LoadWordUnaligned] =
        opcode_counts[Opcode::LWL] + opcode_counts[Opcode::LWR];

    // Compute the number of events in the StoreByte chip.
    events_counts[MipsAirId::StoreByte] = opcode_counts[Opcode::SB];

    // Compute the number of events in the StoreHalf chip.
    events_counts[MipsAirId::StoreHalf] = opcode_counts[Opcode::SH];

    // Compute the number of events in the StoreWordUnaligned chip.
    events_counts[MipsAirId::StoreWordUnaligned] =
        opcode_counts[Opcode::SWL] + opcode_counts[Opcode::SWR];

    // Compute the number of events in the StoreConditional chip.
    events_counts[MipsAirId::StoreConditional] = opcode_counts[Opcode::SC];

    // Compute the number of events in the Sext/Ins/Ext/Maddsub/Teq chips.
    // `opcode_counts[Opcode::SEXT]` includes real, retired `op_a==0` rows too (isolated via
    // `sext_x0_events`, see `AluX0Chip`'s doc comment), so that's subtracted out here.
    events_counts[MipsAirId::Sext] = opcode_counts[Opcode::SEXT] - sext_x0_events;
    // `opcode_counts[Opcode::INS]` includes real, retired `op_a==0` rows too (isolated via
    // `ins_x0_events`, see `AluX0Chip`'s doc comment), so that's subtracted out here.
    events_counts[MipsAirId::Ins] = opcode_counts[Opcode::INS] - ins_x0_events;
    // `opcode_counts[Opcode::EXT]` includes real, retired `op_a==0` rows too (isolated via
    // `ext_x0_events`, see `AluX0Chip`'s doc comment), so that's subtracted out here.
    events_counts[MipsAirId::Ext] = opcode_counts[Opcode::EXT] - ext_x0_events;
    events_counts[MipsAirId::Maddsub] = opcode_counts[Opcode::MADDU]
        + opcode_counts[Opcode::MSUBU]
        + opcode_counts[Opcode::MADD]
        + opcode_counts[Opcode::MSUB];
    events_counts[MipsAirId::Teq] = opcode_counts[Opcode::TEQ];

    // `opcode_counts[Opcode::WSBH]`/`[Opcode::MNE]`/`[Opcode::MEQ]` include real, retired
    // `op_a==0` rows too (isolated via `movcond_x0_events`, see `AluX0Chip`'s doc comment), so
    // that's subtracted out here.
    events_counts[MipsAirId::MovCond] = opcode_counts[Opcode::WSBH]
        + opcode_counts[Opcode::MNE]
        + opcode_counts[Opcode::MEQ]
        - movcond_x0_events;

    // Compute the number of events in the CloClz chip. `opcode_counts[Opcode::CLO]`/`[CLZ]`
    // include real, retired `op_a==0` rows too (isolated via `cloclz_x0_events`, see
    // `AluX0Chip`'s doc comment), so that's subtracted out here.
    events_counts[MipsAirId::CloClz] =
        opcode_counts[Opcode::CLO] + opcode_counts[Opcode::CLZ] - cloclz_x0_events;

    // Compute the number of events in the syscall core chip.
    events_counts[MipsAirId::SyscallCore] = syscalls_sent;

    // Compute the number of events in the SyscallInstrs chip.
    events_counts[MipsAirId::SyscallInstrs] = opcode_counts[Opcode::SYSCALL];

    // Compute the number of events in the global chip.
    events_counts[MipsAirId::Global] = 2 * touched_addresses + syscalls_sent;

    // Unlike before, DivRem's `c * quotient`/`abs`/`abs(remainder) < max(abs(c), 1)` checks are
    // all verified locally via embedded `MulOperation`/`AddOperation`/`LtOperation` copies now (no
    // cross-chip `send_alu` lookups into `MulChip`/`AddChip`/`LtChip` at all -- see `DivRemChip`'s
    // doc comment), so no dependency adjustment is needed here for Mul/Lt. Same for `BranchChip`'s
    // former SLT/SLT dependency into `LtChip` (see `BranchChip`'s doc comment).
    //
    // Note: we ignore the additional dependencies for add/sub, since they are accounted for in
    // the maximal shapes.

    events_counts
}

/// Pads the event counts to account for the worst case jump in events across N cycles.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn pad_mips_event_counts(
    mut event_counts: EnumMap<MipsAirId, u64>,
    num_cycles: u64,
) -> EnumMap<MipsAirId, u64> {
    event_counts.iter_mut().for_each(|(k, v)| match k {
        // No dependency-row producer ever targets this shape anymore (`AddChip` has no
        // synthetic-dependency role -- see its doc comment), so a real instruction's worst-case
        // growth is 1 per cycle.
        MipsAirId::Add => *v += num_cycles,
        // At most one real instruction retires per cycle, so a real ADDI's worst-case growth
        // is 1 per cycle (unlike Add/Sub, which also absorb dependency rows injected by other
        // instructions -- see the multipliers above/below for those).
        MipsAirId::Addi => *v += num_cycles,
        // Same reasoning as Addi: no dependency-row producer ever targets this shape (see
        // `AddNoopChip`'s doc comment), so a real instruction's worst-case growth is 1 per cycle.
        MipsAirId::AddNoop => *v += num_cycles,
        // Same reasoning as Addi/AddNoop: no dependency-row producer ever targets this shape (see
        // `AluX0Chip`'s doc comment), so a real instruction's worst-case growth is 1 per cycle.
        MipsAirId::AluX0 => *v += num_cycles,
        // Same reasoning as Addi/AddNoop/AluX0: no dependency-row producer ever targets this
        // shape anymore (`LoadByteChip`/`LoadHalfChip`'s former LB/LH sign-extension sends are
        // now a direct byte assertion local to those chips -- see `SubChip`'s doc comment), so a
        // real instruction's worst-case growth is 1 per cycle.
        MipsAirId::Sub => *v += num_cycles,
        MipsAirId::Mul => *v += 4 * num_cycles,
        MipsAirId::Bitwise => *v += 3 * num_cycles,
        MipsAirId::ShiftLeft => *v += num_cycles,
        // Same reasoning as ShiftLeft: no dependency-row producer ever targets this shape (see
        // `LuiChip`'s doc comment), so a real instruction's worst-case growth is 1 per cycle.
        MipsAirId::Lui => *v += num_cycles,
        MipsAirId::ShiftRight => *v += num_cycles,
        MipsAirId::DivRem => *v += 4 * num_cycles,
        // Same reasoning as Addi/AddNoop/AluX0/Slti: no dependency-row producer ever targets
        // this shape anymore (`Branch`'s former SLT sends and `DivRem`'s former SLTU send are
        // both verified locally now -- see `BranchChip`/`DivRemChip`'s doc comments), so a real
        // instruction's worst-case growth is 1 per cycle.
        MipsAirId::Lt => *v += num_cycles,
        // Same reasoning as Addi/AddNoop/AluX0: no dependency-row producer ever targets this
        // shape (see `SltiChip`'s doc comment), so a real instruction's worst-case growth is 1
        // per cycle.
        MipsAirId::Slti => *v += num_cycles,
        MipsAirId::MemoryLocal => *v += 64 * num_cycles,
        MipsAirId::Branch => *v += 8 * num_cycles,
        MipsAirId::Jump => *v += 2 * num_cycles,
        MipsAirId::Jumpi => *v += 2 * num_cycles,
        MipsAirId::JumpDirect => *v += 2 * num_cycles,
        MipsAirId::SyscallInstrs => *v += num_cycles,
        // Memory instructions are never synthetic-dependency-row producers or targets, so at
        // most one real instruction of each specific opcode class retires per cycle -- `+1`
        // margin over that derived worst case of 1 for each memory chip below.
        MipsAirId::LoadWord => *v += 2 * num_cycles,
        // Same reasoning as Addi/AddNoop/AluX0/Slti: no dependency-row producer ever targets
        // this shape (see `LoadX0Chip`'s doc comment), so a real instruction's worst-case growth
        // is 1 per cycle.
        MipsAirId::LoadX0 => *v += num_cycles,
        MipsAirId::StoreWord => *v += 2 * num_cycles,
        MipsAirId::LoadByte => *v += 2 * num_cycles,
        MipsAirId::LoadHalf => *v += 2 * num_cycles,
        MipsAirId::LoadWordUnaligned => *v += 2 * num_cycles,
        MipsAirId::StoreByte => *v += 2 * num_cycles,
        MipsAirId::StoreHalf => *v += 2 * num_cycles,
        MipsAirId::StoreWordUnaligned => *v += 2 * num_cycles,
        MipsAirId::StoreConditional => *v += 2 * num_cycles,
        // Each opcode variant contributes its own 1x-per-cycle worst case; `Maddsub` covers 4
        // opcodes (MADD/MADDU/MSUB/MSUBU), matching the combined `8 * num_cycles` the
        // pre-split `MiscInstrs` used (1 each for Sext/Ins/Ext/Teq + 4 for Maddsub).
        // TODO: Check this value.
        MipsAirId::Sext => *v += num_cycles,
        MipsAirId::Ins => *v += num_cycles,
        MipsAirId::Ext => *v += num_cycles,
        MipsAirId::Maddsub => *v += 4 * num_cycles,
        MipsAirId::Teq => *v += num_cycles,
        MipsAirId::CloClz => *v += 3 * num_cycles,     // TODO: Check this value.
        MipsAirId::SyscallCore => *v += 2 * num_cycles,
        MipsAirId::MovCond => *v += 2 * num_cycles,
        MipsAirId::Global => *v += 64 * num_cycles,
        // A `clk_high` crossing happens roughly once every `1 << 24` clk ticks (clk advances by
        // 5 or more per cycle); dividing by `1 << 20` instead of the real `1 << 24`-ish rate
        // keeps this a safe overestimate.
        MipsAirId::StateBump => *v += num_cycles.div_ceil(1 << 20),
        // A shard boundary refreshes up to `NUM_REGISTERS` registers once; reactive re-stamps
        // (currently only possible for SUB's register accesses -- see `SubChip`'s doc comment)
        // are conservatively estimated at the same rate as a `clk_high` crossing.
        MipsAirId::MemoryBump => *v += NUM_REGISTERS as u64 + num_cycles.div_ceil(1 << 20),
        _ => (),
    });
    event_counts
}

#[cfg(test)]
mod tests {
    use super::{estimate_mips_lde_size, MipsAirId, PrecompileEvents};
    use crate::events::{KeccakSpongeEvent, MemoryWriteRecord, PrecompileEvent, SyscallEvent};
    use crate::syscalls::SyscallCode;
    use enum_map::EnumMap;
    use hashbrown::HashMap;
    use strum::IntoEnumIterator;

    /// A `costs_per_air` covering every `MipsAirId`, so `estimate_mips_lde_size`'s unconditional
    /// `costs_per_air[&id]` lookups never panic on a missing entry -- every AIR gets a small
    /// distinct-ish placeholder cost except the two under test below, which get real,
    /// deliberately memorable values so the assertions can compute an exact expected delta.
    fn test_costs() -> HashMap<MipsAirId, u64> {
        let mut costs: HashMap<MipsAirId, u64> = MipsAirId::iter().map(|id| (id, 1)).collect();
        costs.insert(MipsAirId::KeccakSponge, 1000);
        costs.insert(MipsAirId::SyscallPrecompile, 7);
        costs
    }

    fn dummy_syscall_event() -> SyscallEvent {
        SyscallEvent {
            pc: 0,
            next_pc: 0,
            clk: 0,
            a_record: MemoryWriteRecord::default(),
            a_record_is_real: false,
            b_record: None,
            c_record: None,
            syscall_id: 0,
            arg1: 0,
            arg2: 0,
        }
    }

    /// A record with zero precompile events must produce exactly the same estimate as the
    /// pre-precompile-awareness version did (i.e. `add_precompile_cells` is a strict no-op),
    /// so this fix can't regress the common non-precompile-heavy shard.
    #[test]
    fn precompile_free_shard_is_unaffected() {
        let costs = test_costs();
        let counts: EnumMap<MipsAirId, u64> = EnumMap::default();
        let empty = PrecompileEvents::default();

        let baseline = estimate_mips_lde_size(counts, &costs, 1, &empty);

        // Sanity: this isn't just trivially zero.
        assert!(baseline > 0);
        // Calling again with the same (still-empty) events must be perfectly stable.
        assert_eq!(baseline, estimate_mips_lde_size(counts, &costs, 1, &empty));
    }

    /// The previously-missing gap this fix closes: a shard whose only real weight is a heavy
    /// precompile (here, `KeccakSponge`, the widest chip in `mips_costs.json`) must have that
    /// weight reflected in the estimate, in the same `next_power_of_two(events) * cost` unit
    /// system as every core-chip term, not silently ignored.
    #[test]
    fn precompile_heavy_shard_is_priced_in() {
        let costs = test_costs();
        let counts: EnumMap<MipsAirId, u64> = EnumMap::default();

        let empty = PrecompileEvents::default();
        let baseline = estimate_mips_lde_size(counts, &costs, 1, &empty);

        let mut heavy = PrecompileEvents::default();
        let num_events = 10_000_u64;
        for _ in 0..num_events {
            heavy.add_event(
                SyscallCode::KECCAK_SPONGE,
                dummy_syscall_event(),
                PrecompileEvent::KeccakSponge(KeccakSpongeEvent::default()),
            );
        }

        let with_keccak = estimate_mips_lde_size(counts, &costs, 1, &heavy);
        assert!(
            with_keccak > baseline,
            "a shard with {num_events} KeccakSponge events must be priced higher than an \
             otherwise-identical shard with none"
        );

        // The delta must match the documented unit system exactly: `next_power_of_two(event
        // count) * cost_per_event`, folded through the same `size_of::<KoalaBear>() << 1`
        // bytes-per-cell factor as every other term -- for both the KeccakSponge chip itself and
        // the one-row-per-event SyscallPrecompile dispatch chip that every precompile family also
        // feeds.
        let padded = num_events.next_power_of_two();
        let expected_cells = padded * costs[&MipsAirId::KeccakSponge]
            + padded * costs[&MipsAirId::SyscallPrecompile];
        let bytes_per_cell = (core::mem::size_of::<p3_koala_bear::KoalaBear>() << 1) as u64;
        assert_eq!(with_keccak - baseline, expected_cells * bytes_per_cell);
    }

    /// Two precompile families present at once must both be priced in, independently -- this
    /// guards against a regression where only the *first* matching entry in
    /// `PRECOMPILE_AIR_IDS` (or only the widest family, mirroring the old
    /// `estimate_record_trace_bytes`-style single-max-cost shortcut) gets counted.
    #[test]
    fn multiple_precompile_families_all_counted() {
        let costs = test_costs();
        let counts: EnumMap<MipsAirId, u64> = EnumMap::default();

        let mut events = PrecompileEvents::default();
        for _ in 0..4 {
            events.add_event(
                SyscallCode::KECCAK_SPONGE,
                dummy_syscall_event(),
                PrecompileEvent::KeccakSponge(KeccakSpongeEvent::default()),
            );
        }
        for _ in 0..4 {
            events.add_event(
                SyscallCode::POSEIDON2_PERMUTE,
                dummy_syscall_event(),
                PrecompileEvent::Poseidon2Permute(Default::default()),
            );
        }

        let combined = estimate_mips_lde_size(counts, &costs, 1, &events);

        let mut keccak_only = PrecompileEvents::default();
        for _ in 0..4 {
            keccak_only.add_event(
                SyscallCode::KECCAK_SPONGE,
                dummy_syscall_event(),
                PrecompileEvent::KeccakSponge(KeccakSpongeEvent::default()),
            );
        }
        let keccak_only_estimate = estimate_mips_lde_size(counts, &costs, 1, &keccak_only);

        // Adding the second family on top of the first must add strictly more area -- if only
        // one family were being counted, these would be equal.
        assert!(combined > keccak_only_estimate);
    }
}
