use crate::{state::ForkState, ExecutionError, ExecutorMode};

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
        let memory = tracing::info_span!("Unconstrained mode: copy memory")
            .in_scope(|| ctx.rt.state.memory.clone());
        ctx.rt.unconstrained_state = ForkState {
            global_clk: ctx.rt.state.global_clk,
            clk: ctx.rt.state.clk,
            pc: ctx.rt.state.pc,
            memory,
            access_shard: std::mem::take(&mut ctx.rt.state.access_shard),
            access_clk: std::mem::take(&mut ctx.rt.state.access_clk),
            accessed: std::mem::take(&mut ctx.rt.state.accessed),
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
            ctx.rt.state.memory = std::mem::take(&mut ctx.rt.unconstrained_state.memory);
            #[cfg(not(feature = "aot-access"))]
            {
                ctx.rt.state.access_shard =
                    std::mem::take(&mut ctx.rt.unconstrained_state.access_shard); // It does not work for AOT
                ctx.rt.state.access_clk =
                    std::mem::take(&mut ctx.rt.unconstrained_state.access_clk);
                ctx.rt.state.accessed = std::mem::take(&mut ctx.rt.unconstrained_state.accessed);
            }
            #[cfg(feature = "aot-access")]
            {
                tracing::info_span!("Unconstrained mode: copy accessing meta").in_scope(|| {
                    ctx.rt.state.access_shard = ctx.rt.unconstrained_state.access_shard.clone();
                    ctx.rt.state.access_clk = ctx.rt.unconstrained_state.access_clk.clone();
                    ctx.rt.state.accessed = ctx.rt.unconstrained_state.accessed.clone();
                });
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
