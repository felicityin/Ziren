pub mod asm;
pub mod checkpoint;
pub mod common;
pub mod error;
pub mod pure;

use std::{ffi::c_void, io::Write, process::Command, sync::Arc};

use libloading::Library;

use crate::aot::common::*;
use crate::{
    aot::error::{AotError, StaticProgramError},
    ExecutionError, Program,
};
use crate::{Executor, ExecutorMode};

type PureAsmRunFn = unsafe extern "C" fn(executor_ptr: *mut c_void);
type MeteredAsmRunFn = unsafe extern "C" fn(executor_ptr: *mut c_void);

pub struct AotCompiler {
    /// The program.
    pub program: Arc<Program>,

    /// The mode the executor is running in.
    pub executor_mode: ExecutorMode,

    /// The maximum size of each shard.
    pub shard_size: u32,

    /// The maximum number of cycles for a syscall.
    pub max_syscall_cycles: u32,

    /// The frequency to check the stopping condition.
    pub shape_check_frequency: u64,
}

impl AotCompiler {
    /// Create a new AOT instance for the given program.
    pub fn new(
        program: Arc<Program>,
        executor_mode: ExecutorMode,
        shard_size: u32,
        max_syscall_cycles: u32,
        shape_check_frequency: u64,
    ) -> Self {
        Self { program, executor_mode, shard_size, max_syscall_cycles, shape_check_frequency }
    }
}

impl<'a> Executor<'a> {
    pub fn aot_compile_pure_lib(&mut self) {
        let aot = AotCompiler::new(
            self.program.clone(),
            ExecutorMode::Simple,
            self.shard_size,
            self.max_syscall_cycles,
            self.shape_check_frequency,
        );
        let asm_code = aot.create_pure_asm().unwrap();
        let pure_lib = asm_to_lib(&asm_code).unwrap();
        self.pure_lib = Some(pure_lib);
    }

    pub fn aot_compile_metered_lib(&mut self) {
        let aot = AotCompiler::new(
            self.program.clone(),
            ExecutorMode::Checkpoint,
            self.shard_size,
            self.max_syscall_cycles,
            self.shape_check_frequency,
        );
        let asm_code = aot.create_metered_asm().unwrap();
        let metered_lib = asm_to_lib(&asm_code).unwrap();
        self.metered_lib = Some(metered_lib);
    }

    /// Executes the program.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    ///
    pub fn aot_pure_run(&mut self) -> Result<(), ExecutionError> {
        self.print_report = false;
        self.executor_mode = ExecutorMode::Simple;
        tracing::info_span!("initialize").in_scope(|| {
            self.initialize();
        });

        let executor_ptr = self as *mut Executor;

        tracing::info_span!("[aot] pure execute").in_scope(|| unsafe {
            let asm_run: libloading::Symbol<PureAsmRunFn> = self
                .pure_lib
                .as_ref()
                .expect("Please complete AOT first")
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            asm_run(executor_ptr.cast());
        });

        self.postprocess();

        Ok(())
    }

    pub fn aot_metered_run(&mut self) -> Result<(), ExecutionError> {
        self.print_report = false;
        self.executor_mode = ExecutorMode::Checkpoint;
        while !self.aot_metered_execute()? {}
        Ok(())
    }

    /// Executes the program.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    ///
    pub fn aot_metered_execute(&mut self) -> Result<bool, ExecutionError> {
        // If it's the first cycle, initialize the program.
        if self.state.global_clk == 0 {
            tracing::debug_span!("initialize").in_scope(|| {
                self.initialize();
            });
        }

        // Loop until we've executed `self.shard_batch_size` shards if `self.shard_batch_size` is
        // set.
        let mut done = false;
        let mut num_shards_executed = 0;
        loop {
            if self.execute_metered_shard()? {
                done = true;
                self.postprocess();
                break;
            }

            num_shards_executed += 1;
            if num_shards_executed >= self.shard_batch_size {
                break;
            }
        }

        Ok(done)
    }

    fn execute_metered_shard(&mut self) -> Result<bool, ExecutionError> {
        let executor_ptr = self as *mut Executor;

        tracing::debug_span!("[aot] metered execute one checkpoint").in_scope(|| unsafe {
            let asm_run: libloading::Symbol<MeteredAsmRunFn> = self
                .metered_lib
                .as_ref()
                .expect("Please complete AOT first")
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            asm_run(executor_ptr.cast());
        });

        let done = self.state.pc == 0
            || self.state.exited
            || self.state.pc.wrapping_sub(self.program.pc_base)
                >= (self.program.instructions.len() * 4) as u32;
        if done && self.unconstrained {
            log::error!("program ended in unconstrained mode at clk {}", self.state.global_clk);
            return Err(ExecutionError::EndInUnconstrained());
        }
        Ok(done)
    }
}

impl AotCompiler {
    pub fn before_call() -> String {
        let mut asm = String::new();
        asm += &Self::save_xmm_regs();
        asm += &Self::push_address_space_start();
        asm += &Self::push_internal_registers();
        asm
    }

    pub fn after_call() -> String {
        let mut asm = String::new();
        asm += &Self::pop_internal_registers(); // pop the internal registers from the stack
        asm += &Self::pop_address_space_start();
        // read the memory from the memory location of the MIPS registers in `GuestMemory`
        // registers, to the appropriate XMM registers
        asm += &Self::load_xmm_regs();
        asm
    }

    pub fn push_external_registers() -> String {
        let mut asm = String::new();
        asm += "    push rbp\n";
        asm += "    push rbx\n";
        asm += "    push r12\n";
        asm += "    push r13\n";
        asm += "    push r14\n";
        // A dummy push to ensure the stack is 16 bytes aligned
        asm += "    push r15\n";

        asm
    }

    fn pop_external_registers() -> String {
        let mut asm = String::new();
        // There was a dummy push to ensure the stack is 16 bytes aligned
        asm += "    pop r15\n";
        asm += "    pop r14\n";
        asm += "    pop r13\n";
        asm += "    pop r12\n";
        asm += "    pop rbx\n";
        asm += "    pop rbp\n";

        asm
    }

    fn push_internal_registers() -> String {
        let mut asm = String::new();

        asm += "    push rcx\n";
        asm += "    push rdx\n";
        asm += "    push rsi\n";
        asm += "    push rdi\n";
        asm += "    push r8\n";
        asm += "    push r9\n";
        asm += "    push r10\n";
        asm += "    push r11\n";
        asm += "    push rax\n";

        asm
    }

    fn pop_internal_registers() -> String {
        let mut asm = String::new();

        asm += "    pop rax\n";
        asm += "    pop r11\n";
        asm += "    pop r10\n";
        asm += "    pop r9\n";
        asm += "    pop r8\n";
        asm += "    pop rdi\n";
        asm += "    pop rsi\n";
        asm += "    pop rdx\n";
        asm += "    pop rcx\n";

        asm
    }

    // r15 stores vm_register_address
    fn load_xmm_regs() -> String {
        let mut asm = String::new();

        asm += "    push r14\n";
        asm += &format!("    pextrq r14, xmm{REG_ADDR_SPACE}, 1\n");

        for r in 0..16 {
            asm += &format!("   mov rdi, [r14 + 8*{r}]\n");
            asm += &format!("   pinsrq xmm{r}, rdi, 0\n");
        }

        for r in 16..17 {
            asm += &format!("   mov rdi, [r14 + 8*{r}]\n");
            asm += &format!("   pinsrq xmm{}, rdi, 1\n", r - 3);
        }

        asm += "    pop r14\n";

        asm += &sync_xmm_to_gpr();

        asm
    }

    fn save_xmm_regs() -> String {
        let mut asm = String::new();

        asm += &sync_gpr_to_xmm();

        asm += "    push r14\n";
        asm += &format!("    pextrq r14, xmm{REG_ADDR_SPACE}, 1\n");

        for r in 0..16 {
            // at each iteration we save register 2r and 2r+1 of the guest mem to xmm
            asm += &format!("   movq [r14 + 8*{r}], xmm{r}\n");
        }

        for r in 16..17 {
            // at each iteration we save register 2r and 2r+1 of the guest mem to xmm
            asm += &format!("   pextrq [r14 + 8*{r}], xmm{}, 1\n", r - 3);
        }

        asm += "    pop r14\n";

        asm
    }

    fn push_address_space_start() -> String {
        let mut asm = String::new();
        // SAFETY: pay attention to byte alignment.
        asm += "   pextrq rdi, xmm0, 1\n";
        asm += "   push rdi\n";
        asm += "   pextrq rdi, xmm1, 1\n";
        asm += "   push rdi\n";
        asm += "   pextrq rdi, xmm2, 1\n";
        asm += "   push rdi\n";
        asm += "   pextrq rdi, xmm3, 1\n";
        asm += "   push rdi\n";
        asm += "   pextrq rdi, xmm4, 1\n";
        asm += "   push rdi\n";
        asm += "   pextrq rdi, xmm5, 1\n";
        asm += "   push rdi\n";
        asm
    }

    fn pop_address_space_start() -> String {
        let mut asm = String::new();
        // SAFETY: pay attention to byte alignment.
        asm += "   pop rdi\n";
        asm += "   pinsrq xmm5, rdi, 1\n";
        asm += "   pop rdi\n";
        asm += "   pinsrq xmm4, rdi, 1\n";
        asm += "   pop rdi\n";
        asm += "   pinsrq xmm3, rdi, 1\n";
        asm += "   pop rdi\n";
        asm += "   pinsrq xmm2, rdi, 1\n";
        asm += "   pop rdi\n";
        asm += "   pinsrq xmm1, rdi, 1\n";
        asm += "   pop rdi\n";
        asm += "   pinsrq xmm0, rdi, 1\n";
        asm
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

unsafe extern "C" fn set_pc(executor_ptr: *mut c_void, next_pc: u32) {
    let executor = unsafe { &mut *(executor_ptr as *mut Executor) };
    executor.state.pc = next_pc;
}

extern "C" fn get_pc(executor_ptr: *mut c_void) -> *mut u64 {
    let executor = unsafe { &mut *(executor_ptr as *mut Executor) };

    // since pc is the first element of the state field and we use `repr(C)`
    // hence `ptr` will be equal to the address of pc in state
    let ptr = executor.state.pc as *mut u32;
    ptr as *mut u64
}

extern "C" fn get_address_space(executor_ptr: *mut c_void, address_space: u32) -> *mut u64 {
    let executor = unsafe { &mut *(executor_ptr as *mut Executor) };

    let ptr = &executor.state.memory.memory.mem[address_space as usize];
    ptr.as_ptr() as *mut u64 // mut u64 because we want to write 8 bytes at a time
}

extern "C" fn get_access_shard_space(executor_ptr: *mut c_void, address_space: u32) -> *mut u64 {
    let executor = unsafe { &mut *(executor_ptr as *mut Executor) };

    let ptr = &executor.state.access_shard.memory.mem[address_space as usize];
    ptr.as_ptr() as *mut u64 // mut u64 because we want to write 8 bytes at a time
}

extern "C" fn get_access_clk_space(executor_ptr: *mut c_void, address_space: u32) -> *mut u64 {
    let executor = unsafe { &mut *(executor_ptr as *mut Executor) };

    let ptr = &executor.state.access_clk.memory.mem[address_space as usize];
    ptr.as_ptr() as *mut u64 // mut u64 because we want to write 8 bytes at a time
}
