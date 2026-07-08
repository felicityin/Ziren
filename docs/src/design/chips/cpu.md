# CPU 

The CPU chip handles the core logic for processing MIPS instructions. Each program cycle corresponds to a table row accessed via the pc column in the preprocessed Program table. Constraints on pc transitions, clock cycles, and operand handling are enforced through column-based verification.

The CPU architecture employs a structured column-based design to manage instruction execution, branching/jump logic, memory operations, and system calls. Key components are organized into specialized modules (represented as specific columns in the CPU table) with clearly defined constraints and interactions. The CPU table uses selector columns to distinguish instruction types and perform corresponding constraint validation.

## Column Classification 

The CPU columns in Ziren encapsulate the core execution context of MIPS instructions within the zkVM. Key components include:
- ​Shard Management​​: Tracks execution shards for cross-shard operations like syscalls and memory access.
- Clock System​​: Tracks the global clock cycles.
- ​Program Counter​​: Sequential validation via pc, next_pc, and next_next_pc for instruction flow correctness.
- Instruction Decoding​​: Stores opcode, operands, and immediate flags.
- ​Memory Access​​: Validates read/write operations through memory corresponding columns.
- ​Control Flags​​: Identifies instruction types and operand constraints.
- ​Operand Validation​​: Enforces register/immediate selection and zero-register checks.

## ​Constraint Categories​​

Ziren's CPU constraints ensure instruction integrity across four key dimensions:

- Flow Constraints​​

  - Program counter continuity: Consecutive real rows satisfy `local.next_pc = next.pc`.
  - Delay-slot carry: For non-halting successor rows, `local.next_next_pc = next.next_pc`, so branch/jump targets survive across the delay slot.
  - Sequential rows: When `is_sequential = 1`, the CPU enforces `next_next_pc = next_pc + 4`.
  - Shard boundary safety: The last real row of a shard must be sequential or halt. Branch/jump rows cannot end a shard because shard public values export only `next_pc`, not `next_next_pc`.
  - Clock synchronization: Synchronizes timing mechanisms for system operations.

- ​​Operand Constraints​​
  
  Validates operand sources (register/immediate distinction) and enforces zero-value rules under specific conditions.

- Memory Consistency Constraints
  
  Address validity: Verify memory address validity.
  Value consistency: Verify memory consistency checking.

- ​Execution Context Constraints​​

  Instruction exclusivity: Maintains instruction type exclusivity.
  Real-row validation: Enforces operational validity flags for non-padded execution.

These constraints are implemented through AIR polynomial identities, cross-table lookup arguments, boolean assertions, and multi-set hashing ensuring verifiable MIPS execution within Ziren's zkVM framework.

## PC Boundary Rules

The CPU chip carries both `next_pc` and `next_next_pc` in each row.

- `next_pc` is the address of the delay-slot instruction and is always the next row's `pc` for real transitions.
- `next_next_pc` is the PC after the delay slot. For sequential rows it equals `next_pc + 4`; for branch and jump rows it carries the taken or fall-through post-delay-slot target.
- Shard public values expose only `start_pc` and `next_pc`. Because of that, a branch or jump row cannot be the last real row of a shard; otherwise its `next_next_pc` target would be lost at the boundary.
