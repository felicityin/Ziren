use hashbrown::HashMap;

#[cfg(feature = "aot-access")]
use crate::memory::GuestMemory;
use crate::{state::ForkState, ExecutionError, ExecutorMode};
use crate::NUM_REGISTERS;

use super::{Syscall, SyscallCode, SyscallContext};

pub(crate) struct EnterUnconstrainedSyscall;

impl Syscall for EnterUnconstrainedSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext,
        _: SyscallCode,
        _: u32,
        _: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        if ctx.rt.unconstrained {
            panic!("Unconstrained block is already active.");
        }
        ctx.rt.unconstrained = true;

        #[cfg(not(feature = "aot-access"))]
        let (access_shard, access_clk, accessed) = (
            // std::mem::take(&mut ctx.rt.state.memory),
            std::mem::take(&mut ctx.rt.state.access_shard),
            std::mem::take(&mut ctx.rt.state.access_clk),
            std::mem::take(&mut ctx.rt.state.accessed),
        );

        #[cfg(feature = "aot-access")]
        let (memory, access_shard, access_clk, accessed) =
            tracing::info_span!("Unconstrained mode: swap accessed meta").in_scope(|| {
                let memory = ctx.rt.state.memory.clone();
                let mut access_shard = GuestMemory::new_u16();
                let mut access_clk = GuestMemory::default();
                let mut accessed = GuestMemory::new_u8();
                std::mem::swap(&mut ctx.rt.state.access_shard, &mut access_shard);
                std::mem::swap(&mut ctx.rt.state.access_clk, &mut access_clk);
                std::mem::swap(&mut ctx.rt.state.accessed, &mut accessed);
                (memory, access_shard, access_clk, accessed)
            });

        ctx.rt.unconstrained_state = ForkState {
            global_clk: ctx.rt.state.global_clk,
            clk: ctx.rt.state.clk,
            pc: ctx.rt.state.pc,
            // memory,
            memory_diff: HashMap::default(),
            access_shard,
            access_clk,
            accessed,
            record: std::mem::take(&mut ctx.rt.record),
            op_record: std::mem::take(&mut ctx.rt.memory_accesses),
            executor_mode: ctx.rt.executor_mode,
        };
        ctx.rt.executor_mode = ExecutorMode::Simple;
        Ok(Some(1))
    }
}

pub(crate) struct ExitUnconstrainedSyscall;

impl Syscall for ExitUnconstrainedSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext,
        _: SyscallCode,
        _: u32,
        _: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        // Reset the state of the runtime.
        if ctx.rt.unconstrained {
            ctx.rt.state.global_clk = ctx.rt.unconstrained_state.global_clk;
            ctx.rt.state.clk = ctx.rt.unconstrained_state.clk;
            ctx.rt.state.pc = ctx.rt.unconstrained_state.pc;
            ctx.next_pc = ctx.rt.state.pc.wrapping_add(4);
            // ctx.rt.state.memory = std::mem::take(&mut ctx.rt.unconstrained_state.memory);
            #[cfg(not(feature = "aot-access"))]
            {
                for (addr, value) in ctx.rt.unconstrained_state.memory_diff.drain() {
                    if addr < NUM_REGISTERS as u32 {
                        ctx.rt.state.write_register(addr, value);
                    } else {
                        ctx.rt.state.write_memory(addr, value);
                    }
                }

                ctx.rt.state.access_shard =
                    std::mem::take(&mut ctx.rt.unconstrained_state.access_shard); // It does not work for AOT
                ctx.rt.state.access_clk =
                    std::mem::take(&mut ctx.rt.unconstrained_state.access_clk);
                ctx.rt.state.accessed = std::mem::take(&mut ctx.rt.unconstrained_state.accessed);
            }
            #[cfg(feature = "aot-access")]
            {
                std::mem::swap(
                    &mut ctx.rt.state.access_shard,
                    &mut ctx.rt.unconstrained_state.access_shard,
                );
                std::mem::swap(
                    &mut ctx.rt.state.access_clk,
                    &mut ctx.rt.unconstrained_state.access_clk,
                );
                std::mem::swap(
                    &mut ctx.rt.state.accessed,
                    &mut ctx.rt.unconstrained_state.accessed,
                );
            }
            ctx.rt.record = std::mem::take(&mut ctx.rt.unconstrained_state.record);
            ctx.rt.memory_accesses = std::mem::take(&mut ctx.rt.unconstrained_state.op_record);
            ctx.rt.executor_mode = ctx.rt.unconstrained_state.executor_mode;
            ctx.rt.unconstrained = false;
        }
        ctx.rt.unconstrained_state = ForkState::default();
        Ok(Some(0))
    }
}
