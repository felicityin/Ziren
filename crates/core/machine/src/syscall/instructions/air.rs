use std::borrow::Borrow;

use p3_air::{Air, AirBuilder};
use p3_field::FieldAlgebra;
use p3_matrix::Matrix;
use zkm_core_executor::{events::MemoryAccessPosition, syscalls::SyscallCode, Opcode};
use zkm_hypercube::{
    air::{
        BaseAirBuilder, LookupScope, PublicValues, ZKMAirBuilder, POSEIDON_NUM_WORDS,
        PV_DIGEST_NUM_WORDS, ZKM_PROOF_NUM_PV_ELTS,
    },
    word::Word,
};

use crate::{
    adapter::{clk_low_expr, eval_cpu_state, eval_state_chain, InstructionCols},
    air::{MemoryAirBuilder, ProgramAirBuilder, WordAirBuilder},
    operations::{IsZeroOperation, KoalaBearWordRangeChecker},
};

use super::{columns::SyscallInstrColumns, SyscallInstrsChip};

impl<AB> Air<AB> for SyscallInstrsChip
where
    AB: ZKMAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SyscallInstrColumns<AB::Var> = (*local).borrow();
        let is_real: AB::Expr = local.is_real.into();

        let public_values_slice: [AB::PublicVar; ZKM_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar> =
            public_values_slice.as_slice().borrow();

        // SAFETY: `is_real` is checked to be boolean, and `instruction.opcode` matches
        // `Opcode::SYSCALL` on every real row (asserted below).
        builder.assert_bool(local.is_real);

        // Verify that local.is_halt is correct.
        self.eval_is_halt_syscall(builder, local);

        let clk_low = clk_low_expr::<AB>(&local.state);
        let clk_high: AB::Expr = local.state.clk_high.into();

        // The instruction word is reconstructed here rather than stored: SYSCALL always
        // hardcodes its operand registers (`op_a`=`V0`=2, `op_b`=`A0`=4, `op_c`=`A1`=5 -- see
        // this chip's doc comment), so `opcode`/`op_a`/`op_b`/`op_c`/`op_a_0`/`imm_b`/`imm_c` are
        // all compile-time constants, never witnessed columns.
        let instruction: InstructionCols<AB::Expr> = InstructionCols {
            opcode: Opcode::SYSCALL.as_field::<AB::F>().into(),
            op_a: AB::Expr::from_canonical_u32(2),
            op_b: Word([
                AB::Expr::from_canonical_u32(4),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            op_c: Word([
                AB::Expr::from_canonical_u32(5),
                AB::Expr::zero(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]),
            op_a_0: AB::Expr::zero(),
            imm_b: AB::Expr::zero(),
            imm_c: AB::Expr::zero(),
        };
        builder.send_program(local.pc, instruction, is_real.clone());

        // SYSCALL is a read-modify-write of `op_a`: its witnessed `value` is masked/muxed
        // depending on the syscall kind (see `eval_syscall`), so it needs
        // `RegisterWriteAccessCols`'s own witnessed `value` rather than a directly-fed lookup
        // value (see `RegisterWriteAccessCols`'s doc comment). Register positions must be
        // read/written in the order C, B, A (see `MemoryAccessPosition`'s doc comment); each gets
        // its own `clk_low` offset, matching the executor's own `rr_traced`/`rw_traced`
        // timestamps for these accesses (`c = rr_cpu(A1, C)` then `b = rr_cpu(A0, B)` then
        // `rw_cpu(V0, a, A)`).
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::C as u32),
            AB::Expr::from_canonical_u32(5),
            &local.op_c_access,
            is_real.clone(),
        );
        builder.eval_register_access_read(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::B as u32),
            AB::Expr::from_canonical_u32(4),
            &local.op_b_access,
            is_real.clone(),
        );
        builder.eval_register_access_write(
            clk_high.clone(),
            clk_low.clone() + AB::Expr::from_canonical_u32(MemoryAccessPosition::A as u32),
            AB::Expr::from_canonical_u32(2),
            &local.op_a_access,
            is_real.clone(),
        );

        eval_cpu_state(builder, &local.state, clk_low.clone(), is_real.clone());

        // `next_pc` is overwritten to the public sentinel `0` when halting (see
        // `eval_halt_unimpl`), so the state chain's incoming value must substitute `pc + 4`
        // there instead -- witnessed separately since `LookupKind::State` values must stay
        // affine in the trace columns (mirrors `CpuChip::eval_state_chain`'s identical
        // halt-sentinel mechanism).
        builder.assert_eq(
            local.state_chain_next_pc,
            local.next_pc.into()
                + local.is_halt.into() * (local.pc + AB::Expr::from_canonical_u32(4)),
        );

        // We already assert that `local.state.clk < 2^24`. `num_extra_cycles` is an entry of a
        // word and therefore less than `2^8`, this means that the sum cannot overflow in a 31
        // bit field.
        let clk_increment = AB::Expr::from_canonical_u32(5) + local.num_extra_cycles;
        let next_next_pc = local.next_pc + AB::Expr::from_canonical_u32(4);
        eval_state_chain(
            builder,
            clk_high,
            clk_low,
            local.pc.into(),
            local.state_chain_next_pc.into(),
            local.next_pc.into(),
            next_next_pc,
            clk_increment,
            is_real.clone(),
        );

        // `num_extra_cycles` is checked to be equal to the return value of `get_num_extra_syscall_cycles`
        builder.assert_eq::<AB::Var, AB::Expr>(
            local.num_extra_cycles,
            self.get_num_extra_syscall_cycles::<AB>(local),
        );

        // SYSCALL instruction.
        self.eval_syscall(builder, local);

        // COMMIT/COMMIT_DEFERRED_PROOFS syscall instruction.
        self.eval_commit(
            builder,
            local,
            public_values.committed_value_digest,
            public_values.deferred_proofs_digest,
        );

        // HALT syscall and UNIMPL instruction.
        self.eval_halt_unimpl(builder, local, public_values);
    }
}

// The syscall code is the read-in value of op_a at the start of the instruction.
// We interpret the syscall_code as little-endian bytes and interpret each byte as a u8

#[inline(always)]
fn get_syscall_id<AB: ZKMAirBuilder>(local: &SyscallInstrColumns<AB::Var>) -> AB::Expr {
    // syscall id is stored in byte 0, 1.
    let syscall_code = local.op_a_access.prev_value;
    syscall_code[0] + syscall_code[1] * AB::Expr::from_canonical_u32(256)
}

#[inline(always)]
fn get_send_table<AB: ZKMAirBuilder>(local: &SyscallInstrColumns<AB::Var>) -> AB::Var {
    // send_to_table is stored in byte 2
    let syscall_code = local.op_a_access.prev_value;
    syscall_code[2]
}

#[inline(always)]
fn get_num_extra_cycles<AB: ZKMAirBuilder>(local: &SyscallInstrColumns<AB::Var>) -> AB::Var {
    // num_extra_cycles is stored in byte 3.
    let syscall_code = local.op_a_access.prev_value;
    syscall_code[3]
}

#[inline(always)]
fn is_send_table<AB: ZKMAirBuilder>(local: &SyscallInstrColumns<AB::Var>) -> AB::Expr {
    // We interpret the syscall_code as little-endian bytes and interpret each byte as a u8
    local.is_sys_linux + get_send_table::<AB>(local)
}

impl SyscallInstrsChip {
    /// Constraints related to the SYSCALL opcode.
    ///
    /// This method will do the following:
    /// 1. Send the syscall to the precompile table, if needed.
    /// 2. Check for valid op_a values.
    pub(crate) fn eval_syscall<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &SyscallInstrColumns<AB::Var>,
    ) {
        let syscall_id = get_syscall_id::<AB>(local);
        let send_to_table = is_send_table::<AB>(local);

        builder.assert_bool(get_send_table::<AB>(local));
        builder.assert_bool(local.is_sys_linux);
        builder.assert_bool(send_to_table.clone());

        // Constrain is_sys_linux bidirectionally to prevent misrouting between
        // linux and precompile syscall paths.
        // is_prev_a1_zero.result = 1 iff prev_a_value[1] == 0.
        // is_sys_linux must be the inverse: 1 iff prev_a_value[1] != 0.
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.op_a_access.prev_value[1].into(),
            local.is_prev_a1_zero,
            local.is_real.into(),
        );
        builder
            .when(local.is_real)
            .assert_eq(local.is_sys_linux, AB::Expr::one() - local.is_prev_a1_zero.result);

        // SAFETY: Assert that for non real row, the send_to_table value is 0 so that the `send_syscall`
        // interaction is not activated.
        builder.when(AB::Expr::one() - local.is_real).assert_zero(send_to_table.clone());

        // KoalaBear range checks on op_b and op_c, activated by stored flags.
        // Only required on the precompile bridge (where args travel as a single reduced field
        // element), for `is_halt` (exit code is reduced), and for `is_commit_deferred_proofs`
        // (digest element is reduced). Linux syscalls travel via half-word packed columns in
        // `SyscallChip`, which are U16-range-checked there — reduce() collision is impossible,
        // so the KoalaBear range check is not needed (and would reject legal u32 args like
        // AT_FDCWD = 0xFFFFFF9C).
        let send_to_precompile: AB::Expr = get_send_table::<AB>(local).into();
        let op_b_check_active: AB::Expr = send_to_precompile.clone() + local.is_halt.into();
        let op_c_check_active: AB::Expr =
            send_to_precompile + local.is_commit_deferred_proofs.result.into();
        builder.assert_bool(local.op_b_check);
        builder.assert_bool(local.op_c_check);
        builder.when(op_b_check_active).assert_one(local.op_b_check);
        builder.when(op_c_check_active).assert_one(local.op_c_check);
        builder.when_not(local.is_real).assert_zero(local.op_b_check);
        builder.when_not(local.is_real).assert_zero(local.op_c_check);

        KoalaBearWordRangeChecker::<AB::F>::range_check::<AB>(
            builder,
            local.op_b_access.prev_value,
            local.op_b_range_check,
            local.op_b_check.into(),
        );
        KoalaBearWordRangeChecker::<AB::F>::range_check::<AB>(
            builder,
            local.op_c_access.prev_value,
            local.op_c_range_check,
            local.op_c_check.into(),
        );

        builder.send_syscall(
            local.state.clk_high,
            clk_low_expr::<AB>(&local.state),
            syscall_id.clone(),
            local.op_b_access.prev_value.reduce::<AB>(),
            local.op_c_access.prev_value.reduce::<AB>(),
            send_to_table,
            LookupScope::Local,
        );

        // Send full Word bytes for linux syscalls to link op_a (result), op_b (a0), op_c (a1)
        // with SysLinuxChip via SyscallChip bridge.
        builder.send_syscall_result(
            local.state.clk_high,
            clk_low_expr::<AB>(&local.state),
            local.op_a_access.value,
            local.op_b_access.prev_value,
            local.op_c_access.prev_value,
            local.is_sys_linux,
            LookupScope::Local,
        );

        // Compute whether this syscall is ENTER_UNCONSTRAINED.
        let is_enter_unconstrained = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                syscall_id.clone()
                    - AB::Expr::from_canonical_u32(SyscallCode::ENTER_UNCONSTRAINED.syscall_id()),
                local.is_enter_unconstrained,
                local.is_real.into(),
            );
            local.is_enter_unconstrained.result
        };

        builder
            .when(local.is_real)
            .when_not(is_enter_unconstrained)
            .assert_eq(local.syscall_id, syscall_id.clone());

        // The syscall_id should be EXIT_UNCONSTRAINED when is_enter_unconstrained is true.
        builder.when(local.is_real).when(is_enter_unconstrained).assert_eq(
            local.syscall_id,
            AB::Expr::from_canonical_u32(SyscallCode::EXIT_UNCONSTRAINED.syscall_id()),
        );

        // Compute whether this syscall is HINT_LEN.
        let is_hint_len = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                syscall_id.clone()
                    - AB::Expr::from_canonical_u32(SyscallCode::SYSHINTLEN.syscall_id()),
                local.is_hint_len,
                local.is_real.into(),
            );
            local.is_hint_len.result
        };

        // `op_a_val` is constrained.

        // When syscall_id is ENTER_UNCONSTRAINED, the new value of op_a should be 0.
        let zero_word = Word::<AB::F>::from(0);
        builder
            .when(local.is_real)
            .when(is_enter_unconstrained)
            .assert_word_eq(local.op_a_access.value, zero_word);

        // When the syscall is not one of ENTER_UNCONSTRAINED or HINT_LEN, op_a shouldn't change.
        builder
            .when(local.is_real)
            .when_not(is_enter_unconstrained + is_hint_len + local.is_sys_linux)
            .assert_word_eq(local.op_a_access.value, local.op_a_access.prev_value);

        // is_sys_linux is now bidirectionally constrained via is_prev_a1_zero above.
        // When is_sys_linux = 0, prev_a[1] = 0 follows from the IsZero constraint.
        // SAFETY: This leaves the case where syscall is `HINT_LEN`.
        // In this case, `op_a`'s value can be arbitrary, but it still must be a valid word if `is_real = 1`.
        // This is due to `op_a_val` being connected to the CpuChip.
        // In the CpuChip, `op_a_val` is constrained to be a valid word via `RegisterReader::eval`.
        // As this is a syscall for HINT, the value itself being arbitrary is fine, as long as it is a valid word.

        // The old operand_range_check is now subsumed by op_b_range_check (for halt)
        // and op_c_range_check (for commit_deferred_proofs).
    }

    /// Constraints related to the COMMIT and COMMIT_DEFERRED_PROOFS instructions.
    pub(crate) fn eval_commit<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &SyscallInstrColumns<AB::Var>,
        commit_digest: [Word<AB::PublicVar>; PV_DIGEST_NUM_WORDS],
        deferred_proofs_digest: [AB::PublicVar; POSEIDON_NUM_WORDS],
    ) {
        let (is_commit, is_commit_deferred_proofs) =
            self.get_is_commit_related_syscall(builder, local);

        // Verify the index bitmap.
        let mut bitmap_sum = AB::Expr::zero();
        // They should all be bools.
        for bit in local.index_bitmap.iter() {
            builder.when(local.is_real).assert_bool(*bit);
            bitmap_sum = bitmap_sum.clone() + (*bit).into();
        }
        // When the syscall is COMMIT or COMMIT_DEFERRED_PROOFS, there should be one set bit.
        builder
            .when(local.is_real)
            .when(is_commit.clone() + is_commit_deferred_proofs.clone())
            .assert_one(bitmap_sum.clone());
        // When it's some other syscall, there should be no set bits.
        builder
            .when(local.is_real)
            .when(AB::Expr::one() - (is_commit.clone() + is_commit_deferred_proofs.clone()))
            .assert_zero(bitmap_sum);

        // Verify that word_idx corresponds to the set bit in index bitmap.
        for (i, bit) in local.index_bitmap.iter().enumerate() {
            builder
                .when(local.is_real)
                .when(*bit)
                .assert_eq(local.op_b_access.prev_value[0], AB::Expr::from_canonical_u32(i as u32));
        }
        // Verify that the 3 upper bytes of the word_idx are 0.
        for i in 0..3 {
            builder
                .when(local.is_real)
                .when(is_commit.clone() + is_commit_deferred_proofs.clone())
                .assert_zero(local.op_b_access.prev_value[i + 1]);
        }

        // Retrieve the expected public values digest word to check against the one passed into the
        // commit syscall. Note that for the interaction builder, it will not have any digest words,
        // since it's used during AIR compilation time to parse for all send/receives. Since
        // that interaction builder will ignore the other constraints of the air, it is safe
        // to not include the verification check of the expected public values digest word.
        let expected_pv_digest_word = builder.index_word_array(&commit_digest, &local.index_bitmap);

        let digest_word = local.op_c_access.prev_value;

        // Verify the public_values_digest_word.
        builder
            .when(local.is_real)
            .when(is_commit.clone())
            .assert_word_eq(expected_pv_digest_word, digest_word);

        let expected_deferred_proofs_digest_element =
            builder.index_array(&deferred_proofs_digest, &local.index_bitmap);

        // op_c_value is KoalaBear range-checked via op_c_check (activated by is_commit_deferred_proofs).
        builder
            .when(local.is_real)
            .when(is_commit_deferred_proofs.clone())
            .assert_eq(expected_deferred_proofs_digest_element, digest_word.reduce::<AB>());
    }

    /// Constraint related to the halt and unimpl instruction.
    pub(crate) fn eval_halt_unimpl<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &SyscallInstrColumns<AB::Var>,
        public_values: &PublicValues<Word<AB::PublicVar>, AB::PublicVar>,
    ) {
        // `next_pc` is constrained for the case where `is_halt` is true to be `0`
        builder.when(local.is_halt).assert_zero(local.next_pc);

        // op_b_value is KoalaBear range-checked via op_b_check (activated by is_halt).
        // Check that the `op_b_value` reduced is the `public_values.exit_code`.
        builder
            .when(local.is_halt)
            .assert_eq(local.op_b_access.prev_value.reduce::<AB>(), public_values.exit_code);
    }

    /// Returns a boolean expression indicating whether the instruction is a HALT instruction.
    pub(crate) fn eval_is_halt_syscall<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &SyscallInstrColumns<AB::Var>,
    ) {
        // `is_halt` is checked to be correct in `eval_is_halt_syscall`.
        let syscall_id = get_syscall_id::<AB>(local);

        // Compute whether this syscall is HALT.
        let is_halt = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                syscall_id.clone() - AB::Expr::from_canonical_u32(SyscallCode::HALT.syscall_id()),
                local.is_halt_check,
                local.is_real.into(),
            );
            local.is_halt_check.result
        };

        // Compute whether this syscall is SYS_EXIT_GROUP.
        let is_exit_group = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                syscall_id - AB::Expr::from_canonical_u32(SyscallCode::SYS_EXT_GROUP.syscall_id()),
                local.is_exit_group_check,
                local.is_real.into(),
            );
            local.is_exit_group_check.result
        };

        let is_halt_or_exit_group = is_halt + is_exit_group;

        // Verify that the is_halt flag is correct.
        // If `is_real = 0`, then `local.is_halt = 0`.
        // If `is_real = 1`, then `is_halt_check.result or is_exit_group_check.result` will be correct, so `local.is_halt` is correct.
        builder.assert_eq(local.is_halt, is_halt_or_exit_group * local.is_real);
    }

    /// Returns two boolean expression indicating whether the instruction is a COMMIT or
    /// COMMIT_DEFERRED_PROOFS instruction.
    pub(crate) fn get_is_commit_related_syscall<AB: ZKMAirBuilder>(
        &self,
        builder: &mut AB,
        local: &SyscallInstrColumns<AB::Var>,
    ) -> (AB::Expr, AB::Expr) {
        let syscall_id = get_syscall_id::<AB>(local);

        // Compute whether this syscall is COMMIT.
        let is_commit = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                syscall_id.clone() - AB::Expr::from_canonical_u32(SyscallCode::COMMIT.syscall_id()),
                local.is_commit,
                local.is_real.into(),
            );
            local.is_commit.result
        };

        // Compute whether this syscall is COMMIT_DEFERRED_PROOFS.
        let is_commit_deferred_proofs = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                syscall_id
                    - AB::Expr::from_canonical_u32(
                        SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id(),
                    ),
                local.is_commit_deferred_proofs,
                local.is_real.into(),
            );
            local.is_commit_deferred_proofs.result
        };

        (is_commit.into(), is_commit_deferred_proofs.into())
    }

    /// Returns the number of extra cycles from an SYSCALL instruction.
    pub(crate) fn get_num_extra_syscall_cycles<AB: ZKMAirBuilder>(
        &self,
        local: &SyscallInstrColumns<AB::Var>,
    ) -> AB::Expr {
        let num_extra_cycles = get_num_extra_cycles::<AB>(local);

        // If `is_real = 0`, then the return value is `0` regardless of `num_extra_cycles`.
        // If `is_real = 1`, then `num_extra_cycles` will be correct.
        num_extra_cycles * local.is_real
    }
}
