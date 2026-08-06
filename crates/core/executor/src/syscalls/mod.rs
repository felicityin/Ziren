//! Syscall code definitions, shared by `MinimalExecutor`/`CoreVM`/`TracingVM` (dispatched
//! directly by opcode in `minimal/syscall.rs`/`vm.rs`/`tracing.rs`) and by the machine crate's AIR
//! chips (which decode a `SyscallCode`'s packed byte layout, e.g. `syscall_id()`).

mod code;

pub use code::*;
