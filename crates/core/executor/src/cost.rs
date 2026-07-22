use enum_map::EnumMap;
use hashbrown::HashMap;
use p3_koala_bear::KoalaBear;

use crate::{events::NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC, MipsAirId, Opcode};

const BYTE_NUM_ROWS: u64 = 1 << 16;

/// Estimates the LDE area.
///
/// `program_size` is the calling program's real, padded (next-power-of-two) instruction count.
#[must_use]
pub fn estimate_mips_lde_size(
    num_events_per_air: EnumMap<MipsAirId, u64>,
    costs_per_air: &HashMap<MipsAirId, u64>,
    program_size: u64,
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

    // Compute the shift right chip contribution.
    cells += (num_events_per_air[MipsAirId::ShiftRight]).next_power_of_two()
        * costs_per_air[&MipsAirId::ShiftRight];

    // Compute the divrem chip contribution.
    cells += (num_events_per_air[MipsAirId::DivRem]).next_power_of_two()
        * costs_per_air[&MipsAirId::DivRem];

    // Compute the lt chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Lt]).next_power_of_two() * costs_per_air[&MipsAirId::Lt];

    // Compute the memory local chip contribution.
    cells += (num_events_per_air[MipsAirId::MemoryLocal]).next_power_of_two()
        * costs_per_air[&MipsAirId::MemoryLocal];

    // Compute the branch chip contribution.
    cells += (num_events_per_air[MipsAirId::Branch]).next_power_of_two()
        * costs_per_air[&MipsAirId::Branch];

    // Compute the jump chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Jump]).next_power_of_two() * costs_per_air[&MipsAirId::Jump];

    // Compute the SyscallInstruction chip contribution.
    cells += (num_events_per_air[MipsAirId::SyscallInstrs]).next_power_of_two()
        * costs_per_air[&MipsAirId::SyscallInstrs];

    // Compute the MemoryInstruction chip contribution.
    cells += (num_events_per_air[MipsAirId::MemoryInstrs]).next_power_of_two()
        * costs_per_air[&MipsAirId::MemoryInstrs];

    // Compute the LoadWord chip contribution.
    cells += (num_events_per_air[MipsAirId::LoadWord]).next_power_of_two()
        * costs_per_air[&MipsAirId::LoadWord];

    // Compute the StoreWord chip contribution.
    cells += (num_events_per_air[MipsAirId::StoreWord]).next_power_of_two()
        * costs_per_air[&MipsAirId::StoreWord];

    // Compute the MiscInstruction chip contribution.
    cells += (num_events_per_air[MipsAirId::MiscInstrs]).next_power_of_two()
        * costs_per_air[&MipsAirId::MiscInstrs];

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

    cells * ((core::mem::size_of::<KoalaBear>() << 1) as u64)
}

/// Estimate
/// Maps the opcode counts to the number of events in each air.
#[must_use]
pub fn estimate_mips_event_counts(
    touched_addresses: u64,
    syscalls_sent: u64,
    addi_events: u64,
    opcode_counts: EnumMap<Opcode, u64>,
) -> EnumMap<MipsAirId, u64> {
    let mut events_counts: EnumMap<MipsAirId, u64> = EnumMap::default();
    // Compute the number of events in the add chip. `opcode_counts[Opcode::ADD]` mixes
    // register-form ADD, immediate-form ADDI/ADDIU, and register-form internal dependency rows
    // from other chips -- `addi_events` isolates the immediate-form count (see `AddiChip`'s doc
    // comment), so it's subtracted out here and counted on its own line below.
    events_counts[MipsAirId::Add] = opcode_counts[Opcode::ADD] - addi_events;

    // Compute the number of events in the addi chip.
    events_counts[MipsAirId::Addi] = addi_events;

    // Compute the number of events in the sub chip. MIPS has no SUBI, so every SUB opcode
    // occurrence (real or a dependency row from another chip) belongs here.
    events_counts[MipsAirId::Sub] = opcode_counts[Opcode::SUB];

    // Compute the number of events in the mul chip.
    events_counts[MipsAirId::Mul] =
        opcode_counts[Opcode::MUL] + opcode_counts[Opcode::MULT] + opcode_counts[Opcode::MULTU];

    // Compute the number of events in the bitwise chip.
    events_counts[MipsAirId::Bitwise] = opcode_counts[Opcode::XOR]
        + opcode_counts[Opcode::OR]
        + opcode_counts[Opcode::AND]
        + opcode_counts[Opcode::NOR];

    // Compute the number of events in the shift left chip.
    events_counts[MipsAirId::ShiftLeft] = opcode_counts[Opcode::SLL];

    // Compute the number of events in the shift right chip.
    events_counts[MipsAirId::ShiftRight] =
        opcode_counts[Opcode::SRL] + opcode_counts[Opcode::SRA] + opcode_counts[Opcode::ROR];

    // Compute the number of events in the divrem chip.
    events_counts[MipsAirId::DivRem] = opcode_counts[Opcode::DIV] + opcode_counts[Opcode::DIVU];

    // Compute the number of events in the lt chip.
    events_counts[MipsAirId::Lt] = opcode_counts[Opcode::SLT] + opcode_counts[Opcode::SLTU];

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

    // Compute the number of events in the jump chip.
    events_counts[MipsAirId::Jump] = opcode_counts[Opcode::Jump]
        + opcode_counts[Opcode::Jumpi]
        + opcode_counts[Opcode::JumpDirect];

    // Compute the number of events in the MemoryInstrs chip (every memory opcode except the
    // word-aligned LW/SW, which get their own narrower chips below).
    events_counts[MipsAirId::MemoryInstrs] = opcode_counts[Opcode::LB]
        + opcode_counts[Opcode::LH]
        + opcode_counts[Opcode::LBU]
        + opcode_counts[Opcode::LHU]
        + opcode_counts[Opcode::SB]
        + opcode_counts[Opcode::SH]
        + opcode_counts[Opcode::LWL]
        + opcode_counts[Opcode::LWR]
        + opcode_counts[Opcode::LL]
        + opcode_counts[Opcode::SWL]
        + opcode_counts[Opcode::SWR]
        + opcode_counts[Opcode::SC];

    // Compute the number of events in the LoadWord/StoreWord chips.
    events_counts[MipsAirId::LoadWord] = opcode_counts[Opcode::LW];
    events_counts[MipsAirId::StoreWord] = opcode_counts[Opcode::SW];

    // Compute the number of events in the MiscInstrs chip.
    events_counts[MipsAirId::MiscInstrs] = opcode_counts[Opcode::INS]
        + opcode_counts[Opcode::EXT]
        + opcode_counts[Opcode::SEXT]
        + opcode_counts[Opcode::MADDU]
        + opcode_counts[Opcode::MSUBU]
        + opcode_counts[Opcode::MADD]
        + opcode_counts[Opcode::MSUB]
        + opcode_counts[Opcode::TEQ];

    events_counts[MipsAirId::MovCond] =
        opcode_counts[Opcode::WSBH] + opcode_counts[Opcode::MNE] + opcode_counts[Opcode::MEQ];

    // Compute the number of events in the auipc chip.
    events_counts[MipsAirId::CloClz] = opcode_counts[Opcode::CLO] + opcode_counts[Opcode::CLZ];

    // Compute the number of events in the syscall core chip.
    events_counts[MipsAirId::SyscallCore] = syscalls_sent;

    // Compute the number of events in the SyscallInstrs chip.
    events_counts[MipsAirId::SyscallInstrs] = opcode_counts[Opcode::SYSCALL];

    // Compute the number of events in the global chip.
    events_counts[MipsAirId::Global] = 2 * touched_addresses + syscalls_sent;

    // Adjust for divrem dependencies.
    events_counts[MipsAirId::Mul] += events_counts[MipsAirId::DivRem];
    events_counts[MipsAirId::Lt] += events_counts[MipsAirId::DivRem];

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
        // At most one instruction retires per cycle, so only one of the mutually-exclusive
        // dependency-row producers below can fire per cycle. Add's worst case is a DIVREM
        // retiring with both its `c`/remainder sign-correction checks active (2 ADD dependency
        // rows, `emit_divrem_dependencies`); Branch/Jump/`EXT`-flavored MiscInstrs each emit at
        // most 1 ADD dependency row, and a real retired ADD is also just 1 -- all below the
        // DIVREM worst case. `+1` margin over the derived worst case of 2.
        MipsAirId::Add => *v += 3 * num_cycles,
        // At most one real instruction retires per cycle, so a real ADDI's worst-case growth
        // is 1 per cycle (unlike Add/Sub, which also absorb dependency rows injected by other
        // instructions -- see the multipliers above/below for those).
        MipsAirId::Addi => *v += num_cycles,
        // MIPS has no SUBI, so Sub's only dependency-row producer is `emit_memory_dependencies`'
        // LB/LH sign-extension check (at most 1 per retiring memory instruction), same order as
        // a real retired SUB. `+1` margin over the derived worst case of 1.
        MipsAirId::Sub => *v += 2 * num_cycles,
        MipsAirId::Mul => *v += 4 * num_cycles,
        MipsAirId::Bitwise => *v += 3 * num_cycles,
        MipsAirId::ShiftLeft => *v += num_cycles,
        MipsAirId::ShiftRight => *v += num_cycles,
        MipsAirId::DivRem => *v += 4 * num_cycles,
        MipsAirId::Lt => *v += 2 * num_cycles,
        MipsAirId::MemoryLocal => *v += 64 * num_cycles,
        MipsAirId::Branch => *v += 8 * num_cycles,
        MipsAirId::Jump => *v += 2 * num_cycles,
        MipsAirId::SyscallInstrs => *v += num_cycles,
        // Memory instructions are never synthetic-dependency-row producers or targets, so at
        // most one real instruction of each specific opcode class retires per cycle -- `+1`
        // margin over that derived worst case of 1 for each of the three memory chips below.
        MipsAirId::MemoryInstrs => *v += 2 * num_cycles,
        MipsAirId::LoadWord => *v += 2 * num_cycles,
        MipsAirId::StoreWord => *v += 2 * num_cycles,
        MipsAirId::MiscInstrs => *v += 8 * num_cycles, // TODO: Check this value.
        MipsAirId::CloClz => *v += 3 * num_cycles,     // TODO: Check this value.
        MipsAirId::SyscallCore => *v += 2 * num_cycles,
        MipsAirId::MovCond => *v += 2 * num_cycles,
        MipsAirId::Global => *v += 64 * num_cycles,
        _ => (),
    });
    event_counts
}
