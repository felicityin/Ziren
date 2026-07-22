use enum_map::EnumMap;
use hashbrown::HashMap;
use p3_koala_bear::KoalaBear;

use crate::{events::NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC, MipsAirId, Opcode};

const BYTE_NUM_ROWS: u64 = 1 << 16;

/// The `MipsAirId`s whose contribution to a shard's estimated trace area is
/// `count.next_power_of_two() * cost` -- every chip `estimate_mips_lde_size` sums this way,
/// excluding `Byte`/`Program` (fixed-size, not event-count-derived). `ShapeChecker::new` seeds
/// each of these with its own `1 * cost` baseline up front, matching what `estimate_mips_lde_size`
/// computes for an all-zero input (`0u64.next_power_of_two() == 1`) -- so the incremental
/// per-event bumps only ever need to account for the *change* in padded contribution, never
/// re-derive this floor.
const PADDED_COST_AIRS: [MipsAirId; 21] = [
    MipsAirId::Add,
    MipsAirId::Addi,
    MipsAirId::Sub,
    MipsAirId::Mul,
    MipsAirId::Bitwise,
    MipsAirId::ShiftLeft,
    MipsAirId::ShiftRight,
    MipsAirId::DivRem,
    MipsAirId::Lt,
    MipsAirId::MemoryLocal,
    MipsAirId::Branch,
    MipsAirId::Jump,
    MipsAirId::SyscallInstrs,
    MipsAirId::MemoryInstrs,
    MipsAirId::LoadWord,
    MipsAirId::StoreWord,
    MipsAirId::MiscInstrs,
    MipsAirId::CloClz,
    MipsAirId::SyscallCore,
    MipsAirId::MovCond,
    MipsAirId::Global,
];

/// Incrementally tracks a shard-in-progress's estimated padded trace area and max chip height, so
/// `SplicingVM::should_cut_shard` can check the shard-cut condition in O(1) on every cycle instead
/// of periodically re-deriving it from scratch via `estimate_mips_event_counts`/
/// `pad_mips_event_counts`/`estimate_mips_lde_size` (still used, unchanged, by `Executor::
/// inc_shard_if_need`'s own separate copy of this cost model) -- eliminating both the staleness
/// between periodic checks and the worst-case padding margin that covered it. Mirrors SP1's real
/// `ShapeChecker` architecture, adapted to Ziren's own `MipsAirId` set and (already more unified)
/// accessor structure; SP1-specific machinery Ziren has no equivalent of (page-protection
/// tracking, untrusted-instruction-fetch verification -- both part of SP1's "untrusted guest
/// programs" security feature) is intentionally not ported.
pub struct ShapeChecker {
    pub trace_area: u64,
    pub max_height: u64,
    heights: EnumMap<MipsAirId, u64>,
    costs: HashMap<MipsAirId, u64>,
    /// Raw count backing `MipsAirId::MemoryLocal`'s height, which packs
    /// `NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC` touches per row rather than being 1:1 with touch
    /// count -- see `handle_local_mem_event`.
    touched_addresses: u64,
    /// Blocks any further shard cut once a `COMMIT`/`COMMIT_DEFERRED_PROOFS` syscall has fired
    /// within the current shard-in-progress, mirroring SP1's `ShapeChecker::handle_commit`/
    /// `check_shard_limit`'s `!is_commit_on` guard. Ziren's own `should_cut_shard` has no
    /// equivalent guard today; included here as part of faithfully porting `ShapeChecker`'s real,
    /// shipped behavior even though SP1's own source doesn't explain the underlying reason.
    pub is_commit_on: bool,
}

impl ShapeChecker {
    #[must_use]
    pub fn new(costs: HashMap<MipsAirId, u64>, program_size: u64) -> Self {
        let mut trace_area = BYTE_NUM_ROWS * costs.get(&MipsAirId::Byte).copied().unwrap_or(0)
            + program_size * costs.get(&MipsAirId::Program).copied().unwrap_or(0);
        for air in PADDED_COST_AIRS {
            trace_area += costs.get(&air).copied().unwrap_or(0);
        }
        Self {
            trace_area,
            max_height: 0,
            heights: EnumMap::default(),
            costs,
            touched_addresses: 0,
            is_commit_on: false,
        }
    }

    /// Bump `air`'s height by `delta`, adjusting the running padded `trace_area`/`max_height` for
    /// whatever power-of-two boundary this crosses. A no-op for `delta == 0`. Missing cost data
    /// (`Default`'s empty placeholder, which `TracingVM`'s own `CoreVM` never overwrites since it
    /// ignores this tracking entirely -- see `shape_checker`'s field doc comment -- but still
    /// executes the same `step()`/`mr`/`mw` this is hooked into) contributes 0 rather than
    /// panicking, matching `LocalCounts`'s old "harmlessly maintained but unused" behavior.
    fn bump(&mut self, air: MipsAirId, delta: u64) {
        if delta == 0 {
            return;
        }
        let old_height = self.heights[air];
        let new_height = old_height + delta;
        self.heights[air] = new_height;
        let delta_padded = new_height.next_power_of_two() - old_height.next_power_of_two();
        if delta_padded != 0 {
            self.trace_area += delta_padded * self.costs.get(&air).copied().unwrap_or(0);
        }
        self.max_height = self.max_height.max(new_height);
    }

    /// A general-memory or register access touched an address for the first time this shard (see
    /// `CoreVM::mr`/`mw`'s own "first touch" condition) -- bumps `MemoryLocal`/`Global`.
    /// `MemoryLocal`'s height packs `NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC` touches per row
    /// (mirrors `estimate_mips_event_counts`'s `touched_addresses.div_ceil(..)`), so it only
    /// actually grows once every `NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC`th touch.
    pub fn handle_local_mem_event(&mut self) {
        let old_rows =
            self.touched_addresses.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64);
        self.touched_addresses += 1;
        let new_rows =
            self.touched_addresses.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64);
        self.bump(MipsAirId::MemoryLocal, new_rows - old_rows);
        self.bump(MipsAirId::Global, 2);
    }

    /// A real syscall was dispatched -- bumps `SyscallCore`/`Global`. `SyscallInstrs` (the
    /// `SYSCALL` opcode's own CPU-level retirement) is handled by `handle_opcode` instead, exactly
    /// like every other opcode.
    pub fn handle_syscall_dispatched(&mut self) {
        self.bump(MipsAirId::SyscallCore, 1);
        self.bump(MipsAirId::Global, 1);
    }

    /// A `COMMIT`/`COMMIT_DEFERRED_PROOFS` syscall fired -- see `is_commit_on`'s doc comment.
    pub fn handle_commit(&mut self) {
        self.is_commit_on = true;
    }

    /// Bump the `MipsAirId` a real, retired instruction of this opcode belongs to (its "base"
    /// contribution -- see `estimate_mips_event_counts`'s per-opcode sums, which this mirrors in
    /// incremental form). `is_addi` disambiguates `Opcode::ADD`'s two possible chips (MIPS decodes
    /// ADDI as an ADD-opcode instruction with immediate operand flags -- see `CoreVM::step`'s own
    /// `instruction.imm_c && !instruction.imm_b` check).
    pub fn handle_opcode(&mut self, opcode: Opcode, is_addi: bool) {
        if let Some(air) = base_air_id_for_opcode(opcode, is_addi) {
            self.bump(air, 1);
        }
    }

    /// Bump a dependency-row producer's target chip directly, by `delta` -- mirrors one of
    /// `CoreVM::step`'s existing per-instruction dependency-row bumps (see that function's
    /// dispatch block), just landing in the equivalent `MipsAirId` instead of an intermediate
    /// `Opcode` count.
    pub fn handle_dependency(&mut self, opcode: Opcode, delta: u64) {
        if let Some(air) = base_air_id_for_opcode(opcode, false) {
            self.bump(air, delta);
        }
    }
}

/// The `MipsAirId` a real, retired instruction of `opcode` belongs to, for the "base"
/// (non-dependency-row) contribution `estimate_mips_event_counts`'s per-opcode sums encode.
/// `is_addi` only matters for `Opcode::ADD` (see `ShapeChecker::handle_opcode`'s doc comment).
/// Returns `None` for opcodes this cost model doesn't track (mirrors `estimate_mips_event_counts`
/// silently not reading some `opcode_counts` entries).
fn base_air_id_for_opcode(opcode: Opcode, is_addi: bool) -> Option<MipsAirId> {
    Some(match opcode {
        Opcode::ADD if is_addi => MipsAirId::Addi,
        Opcode::ADD => MipsAirId::Add,
        Opcode::SUB => MipsAirId::Sub,
        Opcode::MUL | Opcode::MULT | Opcode::MULTU => MipsAirId::Mul,
        Opcode::XOR | Opcode::OR | Opcode::AND | Opcode::NOR => MipsAirId::Bitwise,
        Opcode::SLL => MipsAirId::ShiftLeft,
        Opcode::SRL | Opcode::SRA | Opcode::ROR => MipsAirId::ShiftRight,
        Opcode::DIV | Opcode::DIVU => MipsAirId::DivRem,
        Opcode::SLT | Opcode::SLTU => MipsAirId::Lt,
        Opcode::BEQ
        | Opcode::BNE
        | Opcode::BGTZ
        | Opcode::BGEZ
        | Opcode::BLTZ
        | Opcode::BLEZ => MipsAirId::Branch,
        Opcode::Jump | Opcode::Jumpi | Opcode::JumpDirect => MipsAirId::Jump,
        Opcode::LB
        | Opcode::LH
        | Opcode::LBU
        | Opcode::LHU
        | Opcode::SB
        | Opcode::SH
        | Opcode::LWL
        | Opcode::LWR
        | Opcode::LL
        | Opcode::SWL
        | Opcode::SWR
        | Opcode::SC => MipsAirId::MemoryInstrs,
        Opcode::LW => MipsAirId::LoadWord,
        Opcode::SW => MipsAirId::StoreWord,
        Opcode::INS
        | Opcode::EXT
        | Opcode::SEXT
        | Opcode::MADDU
        | Opcode::MSUBU
        | Opcode::MADD
        | Opcode::MSUB
        | Opcode::TEQ => MipsAirId::MiscInstrs,
        Opcode::WSBH | Opcode::MNE | Opcode::MEQ => MipsAirId::MovCond,
        Opcode::CLO | Opcode::CLZ => MipsAirId::CloClz,
        Opcode::SYSCALL => MipsAirId::SyscallInstrs,
        _ => return None,
    })
}

impl Default for ShapeChecker {
    /// A placeholder with no real cost data -- `CoreVM::new()` uses this so its own signature
    /// doesn't need `costs`/`program_size` params that `TracingVM` (which never reads this at
    /// all) has no natural reason to supply. `SplicingVM::splice_chunk` overwrites it with a real
    /// `ShapeChecker::new(..)` right after constructing `CoreVM`, mirroring the same
    /// construct-then-overwrite pattern already used for `CoreVM::registers`/`pc`/`clk`.
    fn default() -> Self {
        Self {
            trace_area: 0,
            max_height: 0,
            heights: EnumMap::default(),
            costs: HashMap::new(),
            touched_addresses: 0,
            is_commit_on: false,
        }
    }
}

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
