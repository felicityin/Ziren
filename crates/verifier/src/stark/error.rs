use thiserror::Error;

#[derive(Error, Debug)]
pub enum StarkError {
    #[error("Invalid public values")]
    InvalidPublicValues,
    #[error("Version mismatch")]
    VersionMismatch(String),
    #[error("Invalid verification key")]
    InvalidVerificationKey,
    #[error("Core machine verification error: {0}")]
    Core(String),
    #[error("Recursion verification error: {0}")]
    Recursion(String),
}
