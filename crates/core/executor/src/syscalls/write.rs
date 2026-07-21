use crate::{ExecutionError, Register};

use super::{Syscall, SyscallCode, SyscallContext, SyscallRuntime};

pub use zkm_primitives::consts::fd::*;

pub(crate) struct WriteSyscall;

impl<R: SyscallRuntime> Syscall<R> for WriteSyscall {
    fn execute(
        &self,
        ctx: &mut SyscallContext<R>,
        _: SyscallCode,
        arg1: u32,
        arg2: u32,
    ) -> Result<Option<u32>, ExecutionError> {
        let fd = arg1;
        let write_buf = arg2;
        let nbytes = ctx.register_unsafe(Register::A2);
        // Read nbytes from memory starting at write_buf.
        let bytes = (0..nbytes).map(|i| ctx.byte_unsafe(write_buf + i)).collect::<Vec<u8>>();
        let slice = bytes.as_slice();
        write_fd(ctx, fd, slice)?;
        Ok(None)
    }
}

pub fn write_fd<R: SyscallRuntime>(
    ctx: &mut SyscallContext<R>,
    fd: u32,
    slice: &[u8],
) -> Result<(), ExecutionError> {
    if fd == FD_STDOUT {
        if let Ok(s) = core::str::from_utf8(slice) {
            match parse_cycle_tracker_command(s) {
                Some(command) => handle_cycle_tracker_command(ctx.rt, command),
                None => {
                    let flush_s = ctx.rt.io_buf_push(fd, s);
                    flush_s.into_iter().for_each(|line| ctx.rt.stdout_line(&line));
                }
            }
        } else {
            eprintln!("Warning: Stdout Received invalid UTF-8 data in slice: {slice:?}");
        }
    } else if fd == FD_STDERR {
        if let Ok(s) = core::str::from_utf8(slice) {
            let flush_s = ctx.rt.io_buf_push(fd, s);
            flush_s.into_iter().for_each(|line| ctx.rt.stderr_line(&line));
        } else {
            eprintln!("Warning: Stderr Received invalid UTF-8 data in slice: {slice:?}");
        }
    } else if fd == FD_PUBLIC_VALUES {
        ctx.rt.write_public_values(slice);
    } else if fd == FD_HINT {
        ctx.rt.push_hint_input(slice.to_vec());
    } else {
        match ctx.rt.invoke_hook(fd, slice)? {
            Some(res) => {
                // Add result vectors to the beginning of the stream, preserving their relative
                // order, matching `Executor::state.input_stream.splice(ptr..ptr, res)`.
                for bytes in res.into_iter().rev() {
                    ctx.rt.push_hint_input(bytes);
                }
            }
            None => tracing::warn!("tried to write to unknown file descriptor {fd}"),
        }
    }
    Ok(())
}

/// An enum representing the different cycle tracker commands.
#[derive(Clone)]
enum CycleTrackerCommand {
    Start(String),
    End(String),
    ReportStart(String),
    ReportEnd(String),
}

/// Parse a cycle tracker command from a string. If the string does not match any known command,
/// returns None.
fn parse_cycle_tracker_command(s: &str) -> Option<CycleTrackerCommand> {
    let (command, fn_name) = s.split_once(':')?;
    let trimmed_name = fn_name.trim().to_string();

    match command {
        "cycle-tracker-start" => Some(CycleTrackerCommand::Start(trimmed_name)),
        "cycle-tracker-end" => Some(CycleTrackerCommand::End(trimmed_name)),
        "cycle-tracker-report-start" => Some(CycleTrackerCommand::ReportStart(trimmed_name)),
        "cycle-tracker-report-end" => Some(CycleTrackerCommand::ReportEnd(trimmed_name)),
        _ => None,
    }
}

/// Handle a cycle tracker command.
fn handle_cycle_tracker_command<R: SyscallRuntime>(rt: &mut R, command: CycleTrackerCommand) {
    match command {
        CycleTrackerCommand::Start(name) | CycleTrackerCommand::ReportStart(name) => {
            rt.cycle_tracker_start(&name);
        }
        CycleTrackerCommand::End(name) => {
            rt.cycle_tracker_end(&name);
        }
        CycleTrackerCommand::ReportEnd(name) => {
            // Attempt to end the cycle tracker and accumulate the total cycles in the fn_name's
            // entry in the ExecutionReport.
            if let Some(total_cycles) = rt.cycle_tracker_end(&name) {
                rt.cycle_tracker_report(&name, total_cycles);
            }
        }
    }
}
