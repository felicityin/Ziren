use zkm_stark::CORE_MAX_LOG_ROW_COUNT;

/// The maximum number of instructions in a program.
pub const MAX_PROGRAM_SIZE: usize = 1 << 22;

/// The default increment for the program counter.  Is used for all instructions except
/// for branches and jumps.
pub const DEFAULT_PC_INC: u32 = 4;
/// This is used in the `InstrEvent` to indicate that the instruction is not from the CPU.
/// A valid pc should be divisible by 4, so we use 1 to indicate that the pc is not used.
pub const UNUSED_PC: u32 = 1;

/// Boundary width, in bits, splitting the global `clk` into `clk_low = clk &
/// (CORE_SHARD_CLK_LIMIT - 1)` and `clk_high = clk >> 24`. Every migrated opcode chip's shared
/// `CpuState` range-checks `clk_low` into a 16+8-bit limb pair
/// (`crates/core/machine/src/adapter/state.rs`'s `eval_cpu_state`, via
/// `eval_range_check_24bits`); `clk_high` may change mid-shard, proven in-circuit by the
/// `clk_high`-transition AIR chip fed by [`crate::events::BumpClkHighEvent`]. Shard cuts are
/// decided by `SplicingVM`'s `ShapeChecker` (`crate::splicing`), independent of this constant.
pub const CORE_SHARD_CLK_LIMIT: u64 = 1 << 24;

/// Safety margin subtracted from `1 << CORE_MAX_LOG_ROW_COUNT` to get
/// [`CORE_SHARD_HEIGHT_THRESHOLD`], covering the slop between `ShapeChecker`'s per-instruction
/// height estimate and a chip's real padded row count.
pub const CORE_SHARD_HEIGHT_HEADROOM: u64 = 1 << 16;

/// Hard per-chip real-row-count ceiling: no single chip may reach this many real rows within
/// one shard, since the jagged PCS requires every chip's committed trace to have exactly
/// `1 << CORE_MAX_LOG_ROW_COUNT` rows after padding (`slop/crates/jagged/src/prover.rs`'s
/// `assert_eq!(padded_mle.num_variables(), self.max_log_row_count as u32)`, mirrored by a
/// verifier-side rejection, `slop/crates/jagged/src/verifier.rs`'s `IncorrectShape` check).
/// Unlike `element_threshold` (a trace-area/LDE-size ceiling), that alone does not bound any
/// *individual* chip's row count -- most per-opcode chips get exactly one event per matching
/// cycle, so an opcode-dominated tight loop can otherwise drive a single chip's row count
/// arbitrarily close to the shard's overall size. `SplicingVM`'s `ShapeChecker` (`crate::splicing`)
/// checks every instruction's running per-chip height against this threshold directly.
pub const CORE_SHARD_HEIGHT_THRESHOLD: u64 =
    (1 << CORE_MAX_LOG_ROW_COUNT) - CORE_SHARD_HEIGHT_HEADROOM;

const _: () = assert!(CORE_SHARD_HEIGHT_THRESHOLD <= 1 << CORE_MAX_LOG_ROW_COUNT);
