pub mod common;
pub mod error;
pub mod pure;

use std::{ffi::c_void, io::Write, process::Command, sync::Arc};

use libloading::Library;

use crate::aot::common::*;
use crate::Register;
use crate::{
    aot::error::{AotError, StaticProgramError},
    ExecutionError, ExecutionState, ExecutorMode, Program,
};

type AsmRunFn =
    unsafe extern "C" fn(vm_state_ptr: *mut c_void, instructions_count: u32, pc: u32, next_pc: u32);

/// An aot executor for the MIPS zkVM.
///
/// The executor is responsible for executing a user program and tracing important events which
/// occur during execution (i.e., memory reads, alu operations, etc).
pub struct AotExecutor {
    /// The mode the executor is running in.
    pub executor_mode: ExecutorMode,

    /// The state of the execution.
    pub state: Box<ExecutionState>,

    /// Guest code
    pub lib: Library,

    pub instructions_count: usize,
}

pub struct AotCompiler {
    /// The program.
    pub program: Arc<Program>,
}

impl AotExecutor {
    /// Compile the AOT assembly into a dynamic library and create an AotExecutor.
    pub fn new(program: Program) -> Result<Self, AotError> {
        let executor_mode = ExecutorMode::Trace;
        let state = Box::new(ExecutionState::new(program.pc_start, program.next_pc));
        let instructions_count = program.instructions.len();

        let aot = AotCompiler::new(program);
        let asm_code = aot.create_pure_asm()?;
        let lib = asm_to_lib(&asm_code)?;

        Ok(Self { lib, executor_mode, state, instructions_count })
    }

    /// Executes the program.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn run(&mut self) -> Result<(), ExecutionError> {
        let vm_state_ptr = self.state.as_mut() as *mut ExecutionState;

        tracing::info_span!("execute").in_scope(|| unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> =
                self.lib.get(b"asm_run").expect("Failed to get asm_run symbol");

            asm_run(
                vm_state_ptr.cast(),
                self.instructions_count as u32,
                self.state.pc,
                self.state.next_pc,
            );
        });

        Ok(())
    }

    /// Get the current value of a register, but doesn't use a memory record.
    /// Careful call it directly.
    #[must_use]
    pub fn register(&mut self, register: Register) -> u32 {
        self.state.read_register(register as u32)
    }
}

impl AotCompiler {
    pub fn push_external_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    push rbp\n";
        asm_str += "    push rbx\n";
        asm_str += "    push r12\n";
        asm_str += "    push r13\n";
        asm_str += "    push r14\n";
        // A dummy push to ensure the stack is 16 bytes aligned
        asm_str += "    push r15\n";

        asm_str
    }

    fn pop_external_registers() -> String {
        let mut asm_str = String::new();
        // There was a dummy push to ensure the stack is 16 bytes aligned
        asm_str += "    pop r15\n";
        asm_str += "    pop r14\n";
        asm_str += "    pop r13\n";
        asm_str += "    pop r12\n";
        asm_str += "    pop rbx\n";
        asm_str += "    pop rbp\n";

        asm_str
    }

    fn push_internal_registers() -> String {
        let mut asm_str = String::new();

        asm_str += "    push rcx\n";
        asm_str += "    push rdx\n";
        asm_str += "    push rsi\n";
        asm_str += "    push rdi\n";
        asm_str += "    push r8\n";
        asm_str += "    push r9\n";
        asm_str += "    push r10\n";
        asm_str += "    push r11\n";
        asm_str += "    push rax\n";

        asm_str
    }

    fn pop_internal_registers() -> String {
        let mut asm_str = String::new();

        asm_str += "    pop rax\n";
        asm_str += "    pop r11\n";
        asm_str += "    pop r10\n";
        asm_str += "    pop r9\n";
        asm_str += "    pop r8\n";
        asm_str += "    pop rdi\n";
        asm_str += "    pop rsi\n";
        asm_str += "    pop rdx\n";
        asm_str += "    pop rcx\n";

        asm_str
    }

    // r15 stores vm_register_address
    fn mips_regs_to_xmm() -> String {
        let mut asm_str = String::new();

        asm_str += &format!("    push {REG_AS2_PTR}\n");
        asm_str += &format!("    pextrq {REG_AS2_PTR}, xmm0, 1\n");

        for r in 0..16 {
            asm_str += &format!("   mov rdi, [{REG_AS2_PTR} + 8*{r}]\n");
            asm_str += &format!("   pinsrq xmm{r}, rdi, 0\n");
        }

        for r in 16..17 {
            asm_str += &format!("   mov rdi, [{REG_AS2_PTR} + 8*{r}]\n");
            asm_str += &format!("   pinsrq xmm{}, rdi, 1\n", r - 3);
        }

        asm_str += &format!("    pop {REG_AS2_PTR}\n");

        asm_str += &sync_xmm_to_gpr();

        asm_str
    }

    fn xmm_to_mips_regs() -> String {
        let mut asm_str = String::new();

        asm_str += &sync_gpr_to_xmm();

        asm_str += &format!("    push {REG_AS2_PTR}\n");
        asm_str += &format!("    pextrq {REG_AS2_PTR}, xmm0, 1\n");

        for r in 0..16 {
            // at each iteration we save register 2r and 2r+1 of the guest mem to xmm
            asm_str += &format!("   movq [{REG_AS2_PTR} + 8*{r}], xmm{r}\n");
        }

        for r in 16..17 {
            // at each iteration we save register 2r and 2r+1 of the guest mem to xmm
            asm_str += &format!("   pextrq [{REG_AS2_PTR} + 8*{r}], xmm{}, 1\n", r - 3);
        }

        asm_str += &format!("    pop {REG_AS2_PTR}\n");

        asm_str
    }
}

pub(crate) fn asm_to_lib(asm_source: &str) -> Result<Library, StaticProgramError> {
    let start = std::time::Instant::now();
    // Create a temporary file for the .s file.
    let src_file = tempfile::Builder::new()
        .prefix("asm_x86_run")
        .suffix(".s")
        .tempfile()
        .expect("Failed to create temporary file for asm_x86_run .s file");
    src_file
        .as_file()
        .write(asm_source.as_bytes())
        .map_err(|e| StaticProgramError::FailToWriteTemporaryFile { err: e.to_string() })?;
    let src_path = src_file.into_temp_path();

    // Create a temporary file for the .so file.
    let lib_path = tempfile::Builder::new()
        .prefix("asm_x86_run")
        .suffix(".so")
        .tempfile()
        .map_err(|e| StaticProgramError::FailToCreateTemporaryFile { err: e.to_string() })?
        .into_temp_path();

    // gcc -fPIC -Wl,-z,noexecstack -shared asm_x86_run.s -o asm_x86_run.so
    let status = Command::new("gcc")
        .arg("-fPIC")
        .arg("-Wl,-z,noexecstack")
        .arg("-shared")
        .arg(&src_path)
        .arg("-o")
        .arg(&lib_path)
        .status()
        .map_err(|e| StaticProgramError::FailToGenerateDynamicLibrary { err: e.to_string() })?;
    if !status.success() {
        return Err(StaticProgramError::FailToGenerateDynamicLibrary { err: status.to_string() });
    }

    let lib = unsafe { Library::new(&lib_path).expect("Failed to load library") };
    tracing::trace!(
        "Time taken to build and load .so for AotInstance metered execution: {}ms",
        start.elapsed().as_millis()
    );
    Ok(lib)
}

unsafe extern "C" fn set_pc(state_ptr: *mut c_void, next_pc: u32) {
    let state = &mut *(state_ptr as *mut ExecutionState);
    state.pc = next_pc;
}

extern "C" fn get_pc(state_ptr: *mut c_void) -> *mut u64 {
    let state = unsafe { &mut *(state_ptr as *mut ExecutionState) };

    // since pc is the first element of the vm_state field and we use `repr(C)`
    // hence `ptr` will be equal to the address of pc in vm_state
    let ptr = state.pc as *mut u32;
    ptr as *mut u64
}

extern "C" fn get_address_space(state_ptr: *mut c_void, address_space: u32) -> *mut u64 {
    let state = unsafe { &mut *(state_ptr as *mut ExecutionState) };

    let ptr = &state.memory.memory.mem[address_space as usize];
    ptr.as_ptr() as *mut u64 // mut u64 because we want to write 8 bytes at a time
}

#[cfg(test)]
mod tests {
    use super::AotExecutor;
    use crate::{Instruction, Opcode, Program, Register};

    #[test]
    fn test_aot_add() {
        // add
        simple_op_code_test(Opcode::ADD, 37 + 5, 37, 5);
        // addi
        simple_op_code_i_test(Opcode::ADD, 37 + 5 + 42, 37, 5, 42);
        // addi negative
        simple_op_code_i_test(Opcode::ADD, 5 - 1 + 4, 5, 0xFFFF_FFFF, 4);
    }

    #[test]
    fn test_aot_sub() {
        // sub
        simple_op_code_test(Opcode::SUB, 37 - 5, 37, 5);
        // subi
        simple_op_code_i_test(Opcode::SUB, 37 - 5 - 2, 37, 5, 2);
        // subi negative
        simple_op_code_i_test(Opcode::SUB, 5 + 1 - 4, 5, 0xFFFF_FFFF, 4);
    }

    #[test]
    fn test_aot_and() {
        // and
        simple_op_code_test(Opcode::AND, 37 & 5, 37, 5);
        // andi
        simple_op_code_i_test(Opcode::AND, 37 & 5 & 42, 37, 5, 42);
    }

    #[test]
    fn test_aot_or() {
        // or
        simple_op_code_test(Opcode::OR, 37 | 5, 37, 5);
        // ori
        simple_op_code_i_test(Opcode::OR, 37 | 5 | 42, 37, 5, 42);
    }

    #[test]
    fn test_aot_xor() {
        // xor
        simple_op_code_test(Opcode::XOR, 37 ^ 5, 37, 5);
        // xori
        simple_op_code_i_test(Opcode::XOR, 37 ^ 5 ^ 42, 37, 5, 42);
    }

    #[test]
    fn test_aot_mul() {
        simple_op_code_test(Opcode::MUL, 0x00001200, 0x00007e00, 0xb6db6db7);
        simple_op_code_test(Opcode::MUL, 0x00001240, 0x00007fc0, 0xb6db6db7);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x00000000, 0x00000000);
        simple_op_code_test(Opcode::MUL, 0x00000001, 0x00000001, 0x00000001);
        simple_op_code_test(Opcode::MUL, 0x00000015, 0x00000003, 0x00000007);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x00000000, 0xffff8000);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x80000000, 0x00000000);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x80000000, 0xffff8000);
        simple_op_code_test(Opcode::MUL, 0x0000ff7f, 0xaaaaaaab, 0x0002fe7d);
        simple_op_code_test(Opcode::MUL, 0x0000ff7f, 0x0002fe7d, 0xaaaaaaab);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0xff000000, 0xff000000);
        simple_op_code_test(Opcode::MUL, 0x00000001, 0xffffffff, 0xffffffff);
        simple_op_code_test(Opcode::MUL, 0xffffffff, 0xffffffff, 0x00000001);
        simple_op_code_test(Opcode::MUL, 0xffffffff, 0x00000001, 0xffffffff);
    }

    #[test]
    fn test_aot_shift() {
        // sllv
        simple_op_code_test(Opcode::SLL, 1 << 2, 1, 2);
        // srlv
        simple_op_code_test(Opcode::SRL, 8 >> 1, 8, 1);
        // srav
        simple_op_code_test(Opcode::SRA, 37 >> 4, 37, 4);
        // rotrv
        let c = (((0x12345678 as u64) + ((0x12345678 as u64) << 32)) >> 4) as u32;
        simple_op_code_test(Opcode::ROR, c, 0x12345678, 4);

        // sll
        simple_op_code_i_test(Opcode::SLL, 1 << 2 << 3, 1, 2, 3);
        // srl
        simple_op_code_i_test(Opcode::SRL, 8 >> 1 >> 1, 8, 1, 1);
        // sra
        simple_op_code_i_test(Opcode::SRA, 37 >> 4 >> 1, 37, 4, 1);
        // rotr
        let c = ((c as u64) + ((c as u64) << 32)) >> 4;
        simple_op_code_i_test(Opcode::ROR, c as u32, 0x12345678, 4, 4);
    }

    #[test]
    fn test_aot_shifts() {
        simple_op_code_test(Opcode::SLL, 0x00000001, 0x00000001, 0);
        simple_op_code_test(Opcode::SLL, 0x00000002, 0x00000001, 1);
        simple_op_code_test(Opcode::SLL, 0x00000080, 0x00000001, 7);
        simple_op_code_test(Opcode::SLL, 0x00004000, 0x00000001, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0x00000001, 31);
        simple_op_code_test(Opcode::SLL, 0xffffffff, 0xffffffff, 0);
        simple_op_code_test(Opcode::SLL, 0xfffffffe, 0xffffffff, 1);
        simple_op_code_test(Opcode::SLL, 0xffffff80, 0xffffffff, 7);
        simple_op_code_test(Opcode::SLL, 0xffffc000, 0xffffffff, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0xffffffff, 31);
        simple_op_code_test(Opcode::SLL, 0x21212121, 0x21212121, 0);
        simple_op_code_test(Opcode::SLL, 0x42424242, 0x21212121, 1);
        simple_op_code_test(Opcode::SLL, 0x90909080, 0x21212121, 7);
        simple_op_code_test(Opcode::SLL, 0x48484000, 0x21212121, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0x21212121, 31);
        simple_op_code_test(Opcode::SLL, 0x21212121, 0x21212121, 0xffffffe0);
        simple_op_code_test(Opcode::SLL, 0x42424242, 0x21212121, 0xffffffe1);
        simple_op_code_test(Opcode::SLL, 0x90909080, 0x21212121, 0xffffffe7);
        simple_op_code_test(Opcode::SLL, 0x48484000, 0x21212121, 0xffffffee);
        simple_op_code_test(Opcode::SLL, 0x00000000, 0x21212120, 0xffffffff);

        simple_op_code_test(Opcode::SRL, 0xffff8000, 0xffff8000, 0);
        simple_op_code_test(Opcode::SRL, 0x7fffc000, 0xffff8000, 1);
        simple_op_code_test(Opcode::SRL, 0x01ffff00, 0xffff8000, 7);
        simple_op_code_test(Opcode::SRL, 0x0003fffe, 0xffff8000, 14);
        simple_op_code_test(Opcode::SRL, 0x0001ffff, 0xffff8001, 15);
        simple_op_code_test(Opcode::SRL, 0xffffffff, 0xffffffff, 0);
        simple_op_code_test(Opcode::SRL, 0x7fffffff, 0xffffffff, 1);
        simple_op_code_test(Opcode::SRL, 0x01ffffff, 0xffffffff, 7);
        simple_op_code_test(Opcode::SRL, 0x0003ffff, 0xffffffff, 14);
        simple_op_code_test(Opcode::SRL, 0x00000001, 0xffffffff, 31);
        simple_op_code_test(Opcode::SRL, 0x21212121, 0x21212121, 0);
        simple_op_code_test(Opcode::SRL, 0x10909090, 0x21212121, 1);
        simple_op_code_test(Opcode::SRL, 0x00424242, 0x21212121, 7);
        simple_op_code_test(Opcode::SRL, 0x00008484, 0x21212121, 14);
        simple_op_code_test(Opcode::SRL, 0x00000000, 0x21212121, 31);
        simple_op_code_test(Opcode::SRL, 0x21212121, 0x21212121, 0xffffffe0);
        simple_op_code_test(Opcode::SRL, 0x10909090, 0x21212121, 0xffffffe1);
        simple_op_code_test(Opcode::SRL, 0x00424242, 0x21212121, 0xffffffe7);
        simple_op_code_test(Opcode::SRL, 0x00008484, 0x21212121, 0xffffffee);
        simple_op_code_test(Opcode::SRL, 0x00000000, 0x21212121, 0xffffffff);

        simple_op_code_test(Opcode::SRA, 0x00000000, 0x00000000, 0);
        simple_op_code_test(Opcode::SRA, 0xc0000000, 0x80000000, 1);
        simple_op_code_test(Opcode::SRA, 0xff000000, 0x80000000, 7);
        simple_op_code_test(Opcode::SRA, 0xfffe0000, 0x80000000, 14);
        simple_op_code_test(Opcode::SRA, 0xffffffff, 0x80000001, 31);
        simple_op_code_test(Opcode::SRA, 0x7fffffff, 0x7fffffff, 0);
        simple_op_code_test(Opcode::SRA, 0x3fffffff, 0x7fffffff, 1);
        simple_op_code_test(Opcode::SRA, 0x00ffffff, 0x7fffffff, 7);
        simple_op_code_test(Opcode::SRA, 0x0001ffff, 0x7fffffff, 14);
        simple_op_code_test(Opcode::SRA, 0x00000000, 0x7fffffff, 31);
        simple_op_code_test(Opcode::SRA, 0x81818181, 0x81818181, 0);
        simple_op_code_test(Opcode::SRA, 0xc0c0c0c0, 0x81818181, 1);
        simple_op_code_test(Opcode::SRA, 0xff030303, 0x81818181, 7);
        simple_op_code_test(Opcode::SRA, 0xfffe0606, 0x81818181, 14);
        simple_op_code_test(Opcode::SRA, 0xffffffff, 0x81818181, 31);
    }

    #[test]
    fn test_aot_mult() {
        let mult = |b: u32, c: u32| -> (u32, u32) {
            let out = (((b as i32) as i64) * ((c as i32) as i64)) as u64;
            (out as u32, (out >> 32) as u32) // lo,hi
        };
        let multu = |b: u32, c: u32| -> (u32, u32) {
            let out = b as u64 * c as u64;
            (out as u32, (out >> 32) as u32) //lo,hi
        };

        let tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in tests {
            let (lo, hi) = mult(b, c);
            lo_hi_op_code_test(Opcode::MULT, hi, lo, b, c);

            let (lo, hi) = multu(b, c);
            lo_hi_op_code_test(Opcode::MULTU, hi, lo, b, c);
        }
    }

    #[test]
    fn test_aot_div() {
        let div = |b: u32, c: u32| -> (u32, u32) {
            (
                ((b as i32) / (c as i32)) as u32, // lo
                ((b as i32) % (c as i32)) as u32, // hi
            )
        };
        let divu = |b: u32, c: u32| -> (u32, u32) {
            (b / c, b % c) // lo,hi
        };

        let tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in tests {
            let (lo, hi) = div(b, c);
            lo_hi_op_code_test(Opcode::DIV, hi, lo, b, c);

            let (lo, hi) = divu(b, c);
            lo_hi_op_code_test(Opcode::DIVU, hi, lo, b, c);
        }
    }

    #[test]
    fn test_aot_mod() {
        let modu = |b: u32, c: u32| -> u32 { b % c };
        let modu_tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in modu_tests {
            let expected = modu(b, c);
            simple_op_code_test(Opcode::MODU, expected, b, c);
        }

        let mod_signed = |b: u32, c: u32| -> u32 { ((b as i32) % (c as i32)) as u32 };
        let mod_tests = vec![
            (10, 3),
            (100, 7),
            (1234, 56),
            (0xffff, 0xff),
            (u32::MAX - 1, u32::MAX - 2),
            (0xffff_ffff, 3),
            (0xffff_fffe, 7),
        ];
        for (b, c) in mod_tests {
            let expected = mod_signed(b, c);
            simple_op_code_test(Opcode::MOD, expected, b, c);
        }
    }

    #[test]
    fn test_aot_slt() {
        // slt
        simple_op_code_test(Opcode::SLT, 1, 5, 10);
        simple_op_code_test(Opcode::SLT, 0, 10, 5);
        simple_op_code_test(Opcode::SLT, 0, 10, 10);
        // slti
        slt_i_test(Opcode::SLT, 1, 5, 10);
        slt_i_test(Opcode::SLT, 0, 10, 5);
        slt_i_test(Opcode::SLT, 0, 10, 10);
        // sltu
        simple_op_code_test(Opcode::SLTU, 1, 5, 10);
        simple_op_code_test(Opcode::SLTU, 0, 10, 5);
        simple_op_code_test(Opcode::SLTU, 0, 10, 10);
        // sltiu
        slt_i_test(Opcode::SLTU, 1, 5, 10);
        slt_i_test(Opcode::SLTU, 0, 10, 5);
        slt_i_test(Opcode::SLTU, 0, 10, 10);
    }

    #[test]
    fn test_aot_nor() {
        let nor = |b: u32, c: u32| -> u32 { !(b | c) };
        let mod_tests =
            vec![(10, 3), (100, 7), (1234, 56), (0xffff, 0xff), (u32::MAX - 1, u32::MAX - 2)];
        for (b, c) in mod_tests {
            let expected = nor(b, c);
            simple_op_code_test(Opcode::NOR, expected, b, c);
        }
    }

    #[test]
    fn test_aot_cloz() {
        let clz = |b: u32| -> u32 { b.leading_zeros() };
        let clo = |b: u32| -> u32 { b.leading_ones() };
        let cloz_tests = vec![10, 100, 1234, 0xffff, u32::MAX - 1];
        for b in cloz_tests {
            let expected = clz(b);
            op_code_one_test(Opcode::CLZ, expected, b);
            let expected = clo(b);
            op_code_one_test(Opcode::CLO, expected, b);
        }
    }

    #[test]
    fn test_aot_beq_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 1, false, true),
            Instruction::new(Opcode::BEQ, 29, 30, 4, false, false),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 24);
    }

    #[test]
    fn test_aot_beq_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 2, false, true),
            Instruction::new(Opcode::BEQ, 29, 30, 100, false, false),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 16);
    }

    #[test]
    fn test_aot_bne_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BNE, Register::A0 as u8, 0, 8, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 12);
    }

    #[test]
    fn test_aot_bne_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BNE, Register::A0 as u8, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 8);
    }

    #[test]
    fn test_aot_bltz_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xFFFF_FFFF, false, true),
            Instruction::new(Opcode::BLTZ, 29, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 12);
    }

    #[test]
    fn test_aot_bltz_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BLTZ, Register::A0 as u8, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 8);
    }

    #[test]
    fn test_aot_blez_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BLEZ, Register::A0 as u8, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 8);
    }

    #[test]
    fn test_aot_blez_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::BLEZ, 29, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 12);
    }

    #[test]
    fn test_aot_bgtz_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 1, false, true),
            Instruction::new(Opcode::BGTZ, 29, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 12);
    }

    #[test]
    fn test_aot_bgtz_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BGTZ, Register::A0 as u8, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 8);
    }

    #[test]
    fn test_aot_bgez_jump() {
        let instructions = vec![
            Instruction::new(Opcode::BGEZ, Register::A0 as u8, 0, 4, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 8);
    }

    #[test]
    fn test_aot_bgez_not_jump() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 0xFFFF_FFFF, false, true),
            Instruction::new(Opcode::BGEZ, 29, 0, 100, true, true),
            Instruction::new(Opcode::ADD, 31, 0, 1, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.state.pc, 12);
    }

    fn simple_op_code_test(opcode: Opcode, expected: u32, a: u32, b: u32) {
        // addi x29, x0, a
        // addi x30, x0, b
        // <opcode> RA, x29, x30
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, a, false, true),
            Instruction::new(Opcode::ADD, 30, 0, b, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, 30, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);
        assert_eq!(runtime.state.pc, 12);
    }

    fn simple_op_code_i_test(opcode: Opcode, expected: u32, a: u32, b: u32, c: u32) {
        // addi x29, x0, a
        // <opcode i> x30, x29, b
        // <opcode i> RA, x30, c
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, a, false, true),
            Instruction::new(opcode, 30, 29, b, false, true),
            Instruction::new(opcode, Register::RA as u8, 30, c, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);
        assert_eq!(runtime.state.pc, 12);
    }

    fn lo_hi_op_code_test(opcode: Opcode, expected_hi: u32, expected_lo: u32, a: u32, b: u32) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, a, false, true),
            Instruction::new(Opcode::ADD, 30, 0, b, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, 30, false, false),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.register(Register::LO), expected_lo);
        assert_eq!(runtime.register(Register::HI), expected_hi);
    }

    fn slt_i_test(opcode: Opcode, expected: u32, a: u32, b: u32) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, a, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, b, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);
    }

    fn op_code_one_test(opcode: Opcode, expected: u32, b: u32) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, b, false, true),
            Instruction::new(opcode, Register::RA as u8, 29, b, false, true),
        ];
        let program = Program::new(instructions, 0, 0);
        let mut runtime = AotExecutor::new(program).unwrap();
        runtime.run().unwrap();
        assert_eq!(runtime.register(Register::RA), expected);
    }
}
