use zkm_derive::AlignedBorrow;
use zkm_hypercube::word::Word;

/// Memory read access.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryReadCols<T> {
    pub access: MemoryAccessCols<T>,
}

/// Memory write access.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryWriteCols<T> {
    pub prev_value: Word<T>,
    pub access: MemoryAccessCols<T>,
}

/// Memory read-write access.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryReadWriteCols<T> {
    pub prev_value: Word<T>,
    pub access: MemoryAccessCols<T>,
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessCols<T> {
    /// The value of the memory access.
    pub value: Word<T>,

    /// The previous access's `clk_high` and `clk_low`.
    pub prev_high: T,
    pub prev_low: T,

    /// True if the current access's `clk_high` == the previous access's `clk_high`, else false.
    pub compare_low: T,

    /// The following columns are decomposed limbs for the difference between the current
    /// access's comparison value and the previous access's comparison value. The comparison
    /// value is the accesses' `clk_low` if `compare_low` is set, else their `clk_high`.

    /// This column is the least significant 16 bit limb of current access timestamp - prev access
    /// timestamp.
    pub diff_16bit_limb: T,

    /// This column is the most significant 8 bit limb of current access timestamp - prev access
    /// timestamp.
    pub diff_8bit_limb: T,
}

/// The common columns for all memory access types.
pub trait MemoryCols<T> {
    fn access(&self) -> &MemoryAccessCols<T>;

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T>;

    fn prev_value(&self) -> &Word<T>;

    fn prev_value_mut(&mut self) -> &mut Word<T>;

    fn value(&self) -> &Word<T>;

    fn value_mut(&mut self) -> &mut Word<T>;
}

impl<T> MemoryCols<T> for MemoryReadCols<T> {
    fn access(&self) -> &MemoryAccessCols<T> {
        &self.access
    }

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T> {
        &mut self.access
    }

    fn prev_value(&self) -> &Word<T> {
        &self.access.value
    }

    fn prev_value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }

    fn value(&self) -> &Word<T> {
        &self.access.value
    }

    fn value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }
}

impl<T> MemoryCols<T> for MemoryWriteCols<T> {
    fn access(&self) -> &MemoryAccessCols<T> {
        &self.access
    }

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T> {
        &mut self.access
    }

    fn prev_value(&self) -> &Word<T> {
        &self.prev_value
    }

    fn prev_value_mut(&mut self) -> &mut Word<T> {
        &mut self.prev_value
    }

    fn value(&self) -> &Word<T> {
        &self.access.value
    }

    fn value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }
}

impl<T> MemoryCols<T> for MemoryReadWriteCols<T> {
    fn access(&self) -> &MemoryAccessCols<T> {
        &self.access
    }

    fn access_mut(&mut self) -> &mut MemoryAccessCols<T> {
        &mut self.access
    }

    fn prev_value(&self) -> &Word<T> {
        &self.prev_value
    }

    fn prev_value_mut(&mut self) -> &mut Word<T> {
        &mut self.prev_value
    }

    fn value(&self) -> &Word<T> {
        &self.access.value
    }

    fn value_mut(&mut self) -> &mut Word<T> {
        &mut self.access.value
    }
}

/// A utility method to convert a slice of memory access columns into a vector of values.
/// This is useful for comparing the values of a memory access to limbs.
pub fn value_as_limbs<T: Clone, M: MemoryCols<T>>(memory: &[M]) -> Vec<T> {
    memory.iter().flat_map(|m| m.value().clone().into_iter()).collect()
}

/// Register access timestamp columns for the cheap register-access scheme: unlike
/// [`MemoryAccessCols`], which must handle a `clk_high` comparison against an arbitrary previous
/// access, a register access can assume its previous access shares the same `clk_high` -- an
/// invariant [`crate::memory::MemoryBumpChip`] maintains globally by re-stamping ("bumping") any
/// register access that would otherwise cross a `clk_high` boundary. That leaves only the
/// low-limb comparison to check here, and even the high limb of *that* difference is derived
/// (not stored) via a field-inverse trick -- see [`crate::air::MemoryAirBuilder::eval_register_access_timestamp`].
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RegisterAccessTimestamp<T> {
    /// The previous access's `clk_low` (its `clk_high` is assumed equal to this access's).
    pub prev_low: T,
    /// The least significant 16 bits of `clk_low - prev_low - 1`.
    pub diff_low_limb: T,
}

/// Cheap register *read* access columns, using [`RegisterAccessTimestamp`] in place of the full
/// [`MemoryAccessCols`] timestamp comparison. 6 bytes total (`prev_value`: 4, `access_timestamp`:
/// 2), vs. 9 for the general-purpose scheme. A read's value never changes, so `prev_value` alone
/// (used as both the "previous" and "current" tuple's value in the register-consistency
/// interaction) is enough -- for a write, see [`RegisterWriteAccessCols`] instead.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RegisterAccessCols<T> {
    /// The value of this register before the access.
    pub prev_value: Word<T>,
    /// The access's timestamp-consistency columns.
    pub access_timestamp: RegisterAccessTimestamp<T>,
}

/// Cheap register *write* access columns. Unlike [`RegisterAccessCols`], a write's post-access
/// value must be its own witnessed column (`value`), not merely an expression the caller derives
/// (e.g. an ALU result masked for "writes to register 0 are always discarded to zero") --
/// interaction values/multiplicities on the register-consistency bus must stay affine in the
/// trace columns, and a masked expression like `(1 - op_a_0) * alu_result` is degree 2. Storing
/// `value` as its own column keeps the interaction affine; the caller separately asserts (via an
/// ordinary, non-interaction constraint) that `value` equals zero or the intended computed
/// result, as appropriate. 10 bytes total (`prev_value`: 4, `value`: 4, `access_timestamp`: 2).
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct RegisterWriteAccessCols<T> {
    /// The value of this register before the access.
    pub prev_value: Word<T>,
    /// The value of this register after the access.
    pub value: Word<T>,
    /// The access's timestamp-consistency columns.
    pub access_timestamp: RegisterAccessTimestamp<T>,
}
