mod air;
mod columns;
mod constants;
mod trace;
mod utils;

pub const KECCAK_GENERAL_RATE_U32S: usize = 36;
pub const KECCAK_STATE_U32S: usize = 50;
pub const KECCAK_GENERAL_OUTPUT_U32S: usize = 16;
pub const BITS_PER_LIMB: usize = 64 / p3_keccak_air::U64_LIMBS;

#[derive(Default)]
pub struct KeccakSpongeChip;

impl KeccakSpongeChip {
    pub const fn new() -> Self {
        Self {}
    }
}
#[cfg(test)]
pub mod sponge_tests {
    use crate::utils::{run_test, setup_logger};
    use test_artifacts::KECCAK_SPONGE_ELF;
    use zkm_core_executor::Program;
    #[test]
    fn test_keccak_sponge_program_prove() {
        setup_logger();
        let program = Program::from(KECCAK_SPONGE_ELF).unwrap();
        run_test(program).unwrap();
    }
}
