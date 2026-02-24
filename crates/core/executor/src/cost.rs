use enum_map::EnumMap;
use hashbrown::HashMap;
use p3_koala_bear::KoalaBear;

use crate::{events::NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC, MipsAirId, Opcode};

const BYTE_NUM_ROWS: u64 = 1 << 16;
const MAX_PROGRAM_SIZE: u64 = 1 << 22;

/// Estimates the LDE area.
#[must_use]
pub fn estimate_mips_lde_size(
    num_events_per_air: EnumMap<MipsAirId, u64>,
    costs_per_air: &HashMap<MipsAirId, u64>,
) -> u64 {
    // Compute the byte chip contribution.
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&MipsAirId::Byte];

    // Compute the program chip contribution.
    cells += MAX_PROGRAM_SIZE * costs_per_air[&MipsAirId::Program];

    // Compute the cpu chip contribution.
    cells +=
        (num_events_per_air[MipsAirId::Cpu]).next_power_of_two() * costs_per_air[&MipsAirId::Cpu];

    // Compute the addsub chip contribution.
    cells += (num_events_per_air[MipsAirId::AddSub]).next_power_of_two()
        * costs_per_air[&MipsAirId::AddSub];

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

    // Compute the MiscInstruction chip contribution.
    cells += (num_events_per_air[MipsAirId::MiscInstrs]).next_power_of_two()
        * costs_per_air[&MipsAirId::MiscInstrs];

    // Compute the cloclz chip contribution.
    cells += (num_events_per_air[MipsAirId::CloClz]).next_power_of_two()
        * costs_per_air[&MipsAirId::CloClz];

    // Compute the syscall core chip contribution.
    cells += (num_events_per_air[MipsAirId::SyscallCore]).next_power_of_two()
        * costs_per_air[&MipsAirId::SyscallCore];

    // Compute the global chip contribution.
    cells += (num_events_per_air[MipsAirId::Global]).next_power_of_two()
        * costs_per_air[&MipsAirId::Global];

    cells * ((core::mem::size_of::<KoalaBear>() << 1) as u64)
}

/// Estimate
/// Maps the opcode counts to the number of events in each air.
#[must_use]
pub fn estimate_mips_event_counts(
    cpu_cycles: u64,
    touched_addresses: u64,
    syscalls_sent: u64,
    opcode_counts: &[u64],
) -> EnumMap<MipsAirId, u64> {
    let mut events_counts: EnumMap<MipsAirId, u64> = EnumMap::default();
    // Compute the number of events in the cpu chip.
    events_counts[MipsAirId::Cpu] = cpu_cycles;

    // Compute the number of events in the add sub chip.
    events_counts[MipsAirId::AddSub] =
        opcode_counts[Opcode::ADD as usize] + opcode_counts[Opcode::SUB as usize];

    // Compute the number of events in the mul chip.
    events_counts[MipsAirId::Mul] = opcode_counts[Opcode::MUL as usize]
        + opcode_counts[Opcode::MULT as usize]
        + opcode_counts[Opcode::MULTU as usize];

    // Compute the number of events in the bitwise chip.
    events_counts[MipsAirId::Bitwise] = opcode_counts[Opcode::XOR as usize]
        + opcode_counts[Opcode::OR as usize]
        + opcode_counts[Opcode::AND as usize]
        + opcode_counts[Opcode::NOR as usize];

    // Compute the number of events in the shift left chip.
    events_counts[MipsAirId::ShiftLeft] = opcode_counts[Opcode::SLL as usize];

    // Compute the number of events in the shift right chip.
    events_counts[MipsAirId::ShiftRight] = opcode_counts[Opcode::SRL as usize]
        + opcode_counts[Opcode::SRA as usize]
        + opcode_counts[Opcode::ROR as usize];

    // Compute the number of events in the divrem chip.
    events_counts[MipsAirId::DivRem] =
        opcode_counts[Opcode::DIV as usize] + opcode_counts[Opcode::DIVU as usize];

    // Compute the number of events in the lt chip.
    events_counts[MipsAirId::Lt] =
        opcode_counts[Opcode::SLT as usize] + opcode_counts[Opcode::SLTU as usize];

    // Compute the number of events in the memory local chip.
    events_counts[MipsAirId::MemoryLocal] =
        touched_addresses.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64);

    // Compute the number of events in the branch chip.
    events_counts[MipsAirId::Branch] = opcode_counts[Opcode::BEQ as usize]
        + opcode_counts[Opcode::BNE as usize]
        + opcode_counts[Opcode::BGTZ as usize]
        + opcode_counts[Opcode::BGEZ as usize]
        + opcode_counts[Opcode::BLTZ as usize]
        + opcode_counts[Opcode::BLEZ as usize];

    // Compute the number of events in the jump chip.
    events_counts[MipsAirId::Jump] = opcode_counts[Opcode::Jump as usize]
        + opcode_counts[Opcode::Jumpi as usize]
        + opcode_counts[Opcode::JumpDirect as usize];

    // Compute the number of events in the MemoryInstrs chip.
    events_counts[MipsAirId::MemoryInstrs] = opcode_counts[Opcode::LB as usize]
        + opcode_counts[Opcode::LH as usize]
        + opcode_counts[Opcode::LW as usize]
        + opcode_counts[Opcode::LBU as usize]
        + opcode_counts[Opcode::LHU as usize]
        + opcode_counts[Opcode::SB as usize]
        + opcode_counts[Opcode::SH as usize]
        + opcode_counts[Opcode::SW as usize]
        + opcode_counts[Opcode::LWL as usize]
        + opcode_counts[Opcode::LWR as usize]
        + opcode_counts[Opcode::LL as usize]
        + opcode_counts[Opcode::SWL as usize]
        + opcode_counts[Opcode::SWR as usize]
        + opcode_counts[Opcode::SC as usize];

    // Compute the number of events in the MiscInstrs chip.
    events_counts[MipsAirId::MiscInstrs] = opcode_counts[Opcode::INS as usize]
        + opcode_counts[Opcode::EXT as usize]
        + opcode_counts[Opcode::SEXT as usize]
        + opcode_counts[Opcode::MADDU as usize]
        + opcode_counts[Opcode::MSUBU as usize]
        + opcode_counts[Opcode::MADD as usize]
        + opcode_counts[Opcode::MSUB as usize]
        + opcode_counts[Opcode::TEQ as usize];

    events_counts[MipsAirId::MovCond] = opcode_counts[Opcode::WSBH as usize]
        + opcode_counts[Opcode::MNE as usize]
        + opcode_counts[Opcode::MEQ as usize];

    // Compute the number of events in the auipc chip.
    events_counts[MipsAirId::CloClz] =
        opcode_counts[Opcode::CLO as usize] + opcode_counts[Opcode::CLZ as usize];

    // Compute the number of events in the syscall core chip.
    events_counts[MipsAirId::SyscallCore] = syscalls_sent;

    // Compute the number of events in the global chip.
    events_counts[MipsAirId::Global] = 2 * touched_addresses + syscalls_sent;

    // Adjust for divrem dependencies.
    events_counts[MipsAirId::Mul] += events_counts[MipsAirId::DivRem];
    events_counts[MipsAirId::Lt] += events_counts[MipsAirId::DivRem];

    // Note: we ignore the additional dependencies for addsub, since they are accounted for in
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
        MipsAirId::Cpu => *v += num_cycles,
        MipsAirId::AddSub => *v += 5 * num_cycles,
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
        MipsAirId::MemoryInstrs => *v += 8 * num_cycles,
        MipsAirId::MiscInstrs => *v += 8 * num_cycles, // TODO: Check this value.
        MipsAirId::CloClz => *v += 3 * num_cycles,     // TODO: Check this value.
        MipsAirId::SyscallCore => *v += 2 * num_cycles,
        MipsAirId::Global => *v += 64 * num_cycles,
        _ => (),
    });
    event_counts
}
