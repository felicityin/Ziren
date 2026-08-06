use serde::{Deserialize, Serialize};
use thiserror::Error;
use zkm_curves::CurveError;

use crate::Opcode;

/// Errors that execution (`MinimalExecutor`/`CoreVM`/`TracingVM`) can throw.
#[derive(Error, Debug, Serialize, Deserialize)]
pub enum ExecutionError {
    /// The execution failed with a non-zero exit code.
    #[error("execution failed with exit code {0}")]
    HaltWithNonZeroExitCode(u32),

    /// The execution failed with an invalid memory access.
    #[error("invalid memory access for opcode {0} and address {1}")]
    InvalidMemoryAccess(Opcode, u32),

    /// The execution failed with an unimplemented syscall.
    #[error("unimplemented syscall {0}")]
    UnsupportedSyscall(u32),

    /// The execution failed with an unimplemented instruction.
    #[error("unimplemented instruction {0}")]
    UnsupportedInstruction(u32),

    /// The execution failed with a breakpoint.
    #[error("breakpoint encountered")]
    Breakpoint(),

    /// The execution failed with an exceeded cycle limit.
    #[error("exceeded cycle limit of {0}")]
    ExceededCycleLimit(u64),

    /// The execution failed because the syscall was called in unconstrained mode.
    #[error("syscall called in unconstrained mode")]
    InvalidSyscallUsage(u64),

    /// The execution failed with exception or trap.
    #[error("exception/trap encountered")]
    ExceptionOrTrap(),

    /// The execution failed with an exceeded cycle limit.
    #[error("exceeded memory access bound of {0}")]
    MemoryOutOfBoundsAccess(u64),

    /// The execution failed with invalid syscall args.
    #[error("invalid syscall args encountered")]
    InvalidSyscallArgs(),

    /// The execution failed with an unimplemented feature.
    #[error("got unimplemented as opcode")]
    Unimplemented(),

    /// The program ended in unconstrained mode.
    #[error("program ended in unconstrained mode")]
    EndInUnconstrained(),

    #[error("Null Pointer Reference")]
    NullPointerReference(),

    /// The execution failed because a buffer length did not match the expected size.
    #[error("invalid buffer length: expected {0}, got {1}")]
    InvalidBufferLength(usize, usize),

    /// The execution failed because a buffer length was smaller than the minimum required.
    #[error("buffer length {1} must be greater than or equal to {0}")]
    BufferLengthTooSmall(usize, usize),

    /// The execution failed because a hook received an unsupported elliptic curve identifier.
    #[error("unsupported ecrecover curve id: {0}")]
    UnsupportedEcrecoverCurveId(u8),

    /// The execution failed while converting a slice to an array due to size mismatch.
    #[error("failed to convert slice {0} to array")]
    IntoArrayError(String),

    /// The execution failed because a finite field element was not in canonical form
    /// (i.e., not properly reduced modulo the field's modulus).
    #[error("element {0} must be less than modulus {1}")]
    ElementNotCanonical(String, String),

    /// The execution failed because a finite field element was zero where a non-zero
    /// value was required.
    #[error("element {0} must be non-zero")]
    ElementZero(String),

    /// The execution failed because a quadratic non-residue (NQR) was not in the
    /// valid range (non-zero and less than the modulus).
    #[error("NQR {0} must be non-zero and less then modulus {1}")]
    NqrNotCanonical(String, String),

    /// The execution failed because a value did not satisfy the quadratic residue
    /// property: (root * root) % modulus != qr.
    #[error("{0} * {0}) % {1} != {2}")]
    NqrNotQuadratic(String, String, String),

    /// The execution failed due to an error in the underlying elliptic curve operation.
    #[error("curve error: {0}")]
    CurveError(CurveError),
}
