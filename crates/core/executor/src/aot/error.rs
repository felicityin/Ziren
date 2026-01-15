use thiserror::Error;

use crate::Opcode;

#[derive(Error, Debug)]
pub enum AotError {
    #[error("AOT compilation not supported for this opcode")]
    NotSupported,

    #[error("Invalid instruction format")]
    InvalidInstruction,

    #[error("static program error: {0}")]
    Static(#[from] StaticProgramError),

    #[error("Other AOT error: {0}")]
    Other(String),
}

/// Errors in the program that can be statically analyzed before runtime.
#[derive(Error, Debug)]
pub enum StaticProgramError {
    #[error("invalid instruction at pc {0}")]
    InvalidInstruction(u32),
    #[error("Too many executors")]
    TooManyExecutors,
    #[error("at pc {pc}, opcode {opcode} was not enabled")]
    DisabledOperation { pc: u32, opcode: Opcode },
    #[error("Executor not found for opcode {opcode}")]
    ExecutorNotFound { opcode: Opcode },
    #[error("Failed to create temporary file: {err}")]
    FailToCreateTemporaryFile { err: String },
    #[error("Failed to write into temporary file: {err}")]
    FailToWriteTemporaryFile { err: String },
    #[error("Failed to generate dynamic library: {err}")]
    FailToGenerateDynamicLibrary { err: String },
}
