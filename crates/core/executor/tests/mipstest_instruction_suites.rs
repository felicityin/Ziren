use std::sync::Arc;

use zkm_core_executor::execute_fast_registers;
use zkm_instruction_test_defs::{for_each_instruction_suite, InstructionTestSuite};

macro_rules! define_executor_suite_test {
    ($name:ident, $suite:expr) => {
        #[test]
        fn $name() {
            let suite = &$suite;
            for i in 0..suite.len() {
                eprintln!("running executor suite={} case={}", suite.name(), suite.case_name(i));
                let registers = execute_fast_registers(Arc::new(suite.program(i))).unwrap();
                let mut read_reg = |reg| registers[reg as usize];
                suite.assert_executor(i, &mut read_reg);
            }
        }
    };
}

for_each_instruction_suite!(define_executor_suite_test);
