# Flow Control

Ziren enforces MIPS32r2 control flow verification via dedicated Branch and Jump chips, ensuring precise execution of program control instructions.

 ## Branch Chip

MIPS branch instructions execute conditional jumps through register comparisons (BEQ/BNE for equality, BGTZ/BLEZ etc. for sign checks). They calculate targets using 16-bit offsets shifted left twice (enabling ±128KB jumps) and feature a mandatory branch delay slot that always executes the next instruction—simplifying pipelining by allowing compiler-controlled optimizations.

### Structure Description
Branch chip uses columns to record the following information.
- ​Control Flow Management​​
  - Tracks current and future program counter states across sequential and branching execution paths (`pc, next_pc,target_pc,next_next_pc`).
  - Implements 32-bit address validation through dedicated range-checking components(`next_pc_range_checker, target_pc_range_checker, next_next_pc_range_checker`).
- ​Operand Handling System​​
  - Stores three register/immediate values following MIPS three-operand convention (`op_a_value, op_b_value, op_c_value`).
- ​​Instruction Semantics Encoding
  - Embeds five mutually exclusive flags corresponding to MIPS branch opcodes (`is_beq, is_bltz, is_blez, is_bgtz, is_bgez`).
- ​Execution State Tracking​​
  - Maintains dual execution path indicators for taken/not-taken branch conditions(`is_branching, not_branching`).
- ​Comparison Logic Core​​
  - Evaluates signed integer relationships between primary operands, generating equality, greater-than, and less-than condition flags (`a_eq_b, a_gt_b, a_lt_b`).

### Major Constraints

We use the following key constraints to validate the branch chip:

- Program Counter Validation

  - `next_pc` is the delay-slot PC and must match the next CPU row's `pc`.
  - `next_next_pc` is the post-delay-slot PC carried by the current row.
  - Taken branch case: `next_next_pc = target_pc`.
  - Not-taken branch case: `next_next_pc = next_pc + 4`.
  - `is_branching` and `not_branching` are mutually exclusive and exhaustive for real instructions.
  - Branch rows cannot terminate a shard, because shard public values export only `next_pc` and not the post-delay-slot target in `next_next_pc`.

- Instruction Validity
  - Exactly one branch instruction flag must be active per row (`1 = is_beq + ... + is_bgtz`).
  - Instruction flags are strictly boolean values (0/1).
  - Opcode validity is enforced through linear combination verification.

- Branch Condition Logic
  `is_branching` and `not_branching` consistent with condition flags.

## Jump Chip

MIPS jump instructions force unconditional PC changes via absolute or register-based targets.They calculate 256MB-range addresses by combining PC's upper bits with 26-bit immediates or use full 32-bit register values. All jumps enforce a ​mandatory delay slot executing the next instruction—enabling compiler-driven pipeline optimizations without speculative execution.

### Structure Description
Jump chip uses columns to record the following information:

- ​Control Flow Management​​

  - Tracks current program counter and jump targets (`pc, next_pc, target_pc`).
  - Implements 32-bit address validation via dedicated range checkers (`next_pc_range_checker, target_pc_range_checker, op_a_range_checker`).
- ​​Operand System​​
  - Stores three operands for jump address calculation (`op_a_value, op_b_value, op_c_value`).
- ​​Instruction Semantics​​
  - Embeds three mutually exclusive jump-type flags (`is_jump, is_jumpi, is_jumpdirect`).

### Major Constraints

We use the following key constraints to validate the jump chip:

- Instruction Validity
  - Exactly one jump instruction flag must be active per row:

    ```rust
    1 = is_jump + is_jumpi + is_jumpdirect
    ```
  - Instruction flags are strictly boolean (0/1).
  - Opcode validity enforced through linear combination verification:
    ```rust
    opcode = is_jump*JUMP + is_jumpi*JUMPI + is_jumpdirect*JUMPDIRECT
    ```
- Return Address Handling
  - Return address is saved in op_a_value:
    ```rust
    op_a_value = next_pc + 4
    ```
    op_a_value is saved into op_a register only when 'op_a_0 = 0'(checked in CpuChip)
- Delay-slot / target handling
  - `next_pc` is still the delay-slot PC.
  - The actual jump destination is carried in `next_next_pc` and becomes the successor row's `next_pc` after the delay slot executes.
  - Jump rows cannot be the last real row of a shard for the same reason as branches: only `next_pc` is exported through shard public values.
- Range Checking
  - All critical values (`op_a_value, next_pc, target_pc`) are range-checked, ensuring values are valid 32-bit words.
- PC Transition Logic
  - Target_pc calculation via ALU operation:
    ```rust
    send_alu(
      Opcode::ADD,
      target_pc = next_pc + op_b_value, 
      is_jumpdirect
    )
    ```
  - Direct jumps (`is_jumpdirect`) use immediate operand addition.
