use super::program::MAX_MEMORY;
use crate::register::NUM_REGISTERS;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use vec_map::VecMap;

/// A memory.
///
/// Consists of registers, as well as a page table for main memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct Memory<T: Copy> {
    /// The registers.
    pub registers: Registers<T>,
    /// The page table.
    pub page_table: PagedMemory<T>,
}

impl<V: Copy + 'static> IntoIterator for Memory<V> {
    type Item = (u32, V);

    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.registers.into_iter().chain(self.page_table))
    }
}

impl<T: Copy + Default> Default for Memory<T> {
    fn default() -> Self {
        Self { registers: Registers::default(), page_table: PagedMemory::default() }
    }
}

impl<T: Copy> Memory<T> {
    /// Initialize a new memory with preallocated page table.
    pub fn new_preallocated() -> Self {
        Self { registers: Registers::default(), page_table: PagedMemory::new_preallocated() }
    }

    /// Get an entry for the given address.
    ///
    /// When possible, prefer directly accessing the `page_table` or `registers` fields.
    /// This method often incurs unnecessary branching.
    #[inline]
    pub fn entry(&mut self, addr: u32) -> Entry<'_, T> {
        if addr < NUM_REGISTERS as u32 {
            self.registers.entry(addr)
        } else {
            self.page_table.entry(addr)
        }
    }

    /// Insert a value into the memory.
    ///
    /// When possible, prefer directly accessing the `page_table` or `registers` fields.
    /// This method often incurs unnecessary branching.   
    #[inline]
    pub fn insert(&mut self, addr: u32, value: T) -> Option<T> {
        if addr < NUM_REGISTERS as u32 {
            self.registers.insert(addr, value)
        } else {
            self.page_table.insert(addr, value)
        }
    }

    /// Get a value from the memory.
    ///
    /// When possible, prefer directly accessing the `page_table` or `registers` fields.
    /// This method often incurs unnecessary branching.
    #[inline]
    pub fn get(&self, addr: u32) -> Option<&T> {
        if addr < NUM_REGISTERS as u32 {
            self.registers.get(addr)
        } else {
            self.page_table.get(addr)
        }
    }

    /// Remove a value from the memory.
    ///
    /// When possible, prefer directly accessing the `page_table` or `registers` fields.
    /// This method often incurs unnecessary branching.
    #[inline]
    pub fn remove(&mut self, addr: u32) -> Option<T> {
        if addr < NUM_REGISTERS as u32 {
            self.registers.remove(addr)
        } else {
            self.page_table.remove(addr)
        }
    }

    /// Clear the memory.
    #[inline]
    pub fn clear(&mut self) {
        self.registers.clear();
        self.page_table.clear();
    }
}

impl<V: Copy + Default> FromIterator<(u32, V)> for Memory<V> {
    fn from_iter<T: IntoIterator<Item = (u32, V)>>(iter: T) -> Self {
        let mut memory = Self::new_preallocated();
        for (addr, value) in iter {
            memory.insert(addr, value);
        }
        memory
    }
}

/// An array of NUM_REGISTERS registers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct Registers<T: Copy> {
    pub registers: Vec<Option<T>>,
}

impl<T: Copy> Default for Registers<T> {
    fn default() -> Self {
        Self { registers: vec![None; NUM_REGISTERS] }
    }
}

impl<T: Copy> Registers<T> {
    /// Get an entry for the given register.
    #[inline]
    pub fn entry(&mut self, addr: u32) -> Entry<'_, T> {
        let entry = &mut self.registers[addr as usize];
        match entry {
            Some(v) => Entry::Occupied(OccupiedEntry { entry: v }),
            None => Entry::Vacant(VacantEntry { entry }),
        }
    }

    /// Insert a value into the registers.
    ///
    /// Assumes addr < NUM_REGISTERS.
    #[inline]
    pub fn insert(&mut self, addr: u32, value: T) -> Option<T> {
        self.registers[addr as usize].replace(value)
    }

    /// Remove a value from the registers, and return it if it exists.
    ///
    /// Assumes addr < NUM_REGISTERS.
    #[inline]
    pub fn remove(&mut self, addr: u32) -> Option<T> {
        self.registers[addr as usize].take()
    }

    /// Get a reference to the value at the given address, if it exists.
    ///
    /// Assumes addr < NUM_REGISTERS.
    #[inline]
    pub fn get(&self, addr: u32) -> Option<&T> {
        self.registers[addr as usize].as_ref()
    }

    /// Clear the registers.
    #[inline]
    pub fn clear(&mut self) {
        self.registers.fill(None);
    }
}

impl<V: Copy> FromIterator<(u32, V)> for Registers<V> {
    fn from_iter<T: IntoIterator<Item = (u32, V)>>(iter: T) -> Self {
        let mut mmu = Self::default();
        for (k, v) in iter {
            mmu.insert(k, v);
        }
        mmu
    }
}

impl<V: Copy + 'static> IntoIterator for Registers<V> {
    type Item = (u32, V);

    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(
            self.registers
                .into_iter()
                .enumerate()
                .filter_map(move |(i, v)| v.map(|v| (i as u32, v))),
        )
    }
}

/// A page of memory.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page<V>(VecMap<V>);

impl<V> Default for Page<V> {
    fn default() -> Self {
        Self(VecMap::default())
    }
}

const LOG_PAGE_LEN: usize = 14;
const PAGE_LEN: usize = 1 << LOG_PAGE_LEN;
const MAX_PAGE_COUNT: usize = MAX_MEMORY / 4 / PAGE_LEN + 1;
const NO_PAGE: u16 = u16::MAX;
const PAGE_MASK: usize = PAGE_LEN - 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "V: Serialize"))]
#[serde(bound(deserialize = "V: DeserializeOwned"))]
pub struct NewPage<V>(Vec<Option<V>>);

impl<V: Copy> NewPage<V> {
    pub fn new() -> Self {
        Self(vec![None; PAGE_LEN])
    }
}

impl<V: Copy> Default for NewPage<V> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

/// Paged memory. Balances both memory locality and total memory usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "V: Serialize"))]
#[serde(bound(deserialize = "V: DeserializeOwned"))]
pub struct PagedMemory<V: Copy> {
    /// The internal page table.
    pub page_table: Vec<NewPage<V>>,
    pub index: Vec<u16>,
}

impl<V: Copy> PagedMemory<V> {
    /// The number of lower bits to ignore, since addresses (except registers) are a multiple of 4.
    const NUM_IGNORED_LOWER_BITS: usize = 2;

    /// Create a `PagedMemory` with capacity `MAX_PAGE_COUNT`.
    pub fn new_preallocated() -> Self {
        Self { page_table: Vec::new(), index: vec![NO_PAGE; MAX_PAGE_COUNT] }
    }

    /// Get a reference to the memory value at the given address, if it exists.
    pub fn get(&self, addr: u32) -> Option<&V> {
        let (upper, lower) = Self::indices(addr);
        let index = self.index[upper];
        if index == NO_PAGE {
            None
        } else {
            self.page_table[index as usize].0[lower].as_ref()
        }
    }

    /// Get a mutable reference to the memory value at the given address, if it exists.
    pub fn get_mut(&mut self, addr: u32) -> Option<&mut V> {
        let (upper, lower) = Self::indices(addr);
        let index = self.index[upper];
        if index == NO_PAGE {
            None
        } else {
            self.page_table[index as usize].0[lower].as_mut()
        }
    }

    /// Insert a value at the given address. Returns the previous value, if any.
    #[inline]
    pub fn insert(&mut self, addr: u32, value: V) -> Option<V> {
        let (upper, lower) = Self::indices(addr);
        let mut index = self.index[upper];
        if index == NO_PAGE {
            index = self.page_table.len() as u16;
            self.index[upper] = index;
            self.page_table.push(NewPage::new());
        }
        self.page_table[index as usize].0[lower].replace(value)
    }

    /// Remove the value at the given address if it exists, returning it.
    pub fn remove(&mut self, addr: u32) -> Option<V> {
        let (upper, lower) = Self::indices(addr);
        let index = self.index[upper];
        if index == NO_PAGE {
            None
        } else {
            self.page_table[index as usize].0[lower].take()
        }
    }

    /// Gets the memory entry for the given address.
    #[inline]
    pub fn entry(&mut self, addr: u32) -> Entry<'_, V> {
        let (upper, lower) = Self::indices(addr);
        let index = self.index[upper];
        if index == NO_PAGE {
            let index = self.page_table.len();
            self.index[upper] = index as u16;
            self.page_table.push(NewPage::new());
            Entry::Vacant(VacantEntry { entry: &mut self.page_table[index].0[lower] })
        } else {
            let option = &mut self.page_table[index as usize].0[lower];
            match option {
                Some(v) => Entry::Occupied(OccupiedEntry { entry: v }),
                None => Entry::Vacant(VacantEntry { entry: option }),
            }
        }
    }

    /// Returns an iterator over the occupied addresses.
    pub fn keys(&self) -> impl Iterator<Item = u32> + '_ {
        self.index.iter().enumerate().filter(|(_, &i)| i != NO_PAGE).flat_map(|(i, index)| {
            let upper = i << LOG_PAGE_LEN;
            self.page_table[*index as usize]
                .0
                .iter()
                .enumerate()
                .filter_map(move |(lower, v)| v.map(|_| Self::decompress_addr(upper + lower)))
        })
    }

    /// Get the exact number of addresses in use. This function iterates through each page
    /// and is therefore somewhat expensive.
    pub fn exact_len(&self) -> usize {
        self.index
            .iter()
            .filter(|&&i| i != NO_PAGE)
            .map(|index| self.page_table[*index as usize].0.iter().filter(|v| v.is_some()).count())
            .sum()
    }

    /// Estimate the number of addresses in use.
    pub fn estimate_len(&self) -> usize {
        self.index.iter().filter(|&i| *i != NO_PAGE).count() * PAGE_LEN
    }

    /// Clears the page table. Drops all `Page`s, but retains the memory used by the table itself.
    pub fn clear(&mut self) {
        self.page_table.clear();
        self.index.fill(NO_PAGE);
    }

    /// Break apart an address into an upper and lower index.
    #[inline]
    const fn indices(addr: u32) -> (usize, usize) {
        let index = Self::compress_addr(addr);
        (index >> LOG_PAGE_LEN, index & PAGE_MASK)
    }

    /// Compress an address from the sparse address space to a contiguous space.
    #[inline]
    const fn compress_addr(addr: u32) -> usize {
        addr as usize >> Self::NUM_IGNORED_LOWER_BITS
    }

    /// Decompress an address from a contiguous space to the sparse address space.
    #[inline]
    const fn decompress_addr(addr: usize) -> u32 {
        (addr << Self::NUM_IGNORED_LOWER_BITS) as u32
    }
}

impl<V: Copy> Default for PagedMemory<V> {
    fn default() -> Self {
        Self { page_table: Vec::new(), index: vec![NO_PAGE; MAX_PAGE_COUNT] }
    }
}

/// A `PagedMemory` variant supporting O(1) unconstrained-mode enter/discard (rather than a
/// diff-based undo log, which is O(touched-addresses) to unwind).
///
/// Entering COW mode (`copy_on_write`) moves the current, fully-populated memory into `original`
/// (a plain field move -- the `Vec`s' pointer/length/capacity, not their contents -- so this is
/// O(1) regardless of how much memory is already touched) and starts a fresh, empty `copy`
/// overlay. Every write during the block goes into `copy`, lazily duplicating the touched
/// address's prior value from `original` on first write to that address (not its whole page).
/// Reads check `copy` first, falling back to `original` for addresses `copy` hasn't touched.
/// Discarding on exit (`discard_cow`) just drops `copy` and restores `original` -- no per-address
/// undo/replay loop, unlike a diff-based approach.
pub(crate) enum MaybeCowMemory<V: Copy> {
    Owned(PagedMemory<V>),
    Cow { copy: PagedMemory<V>, original: PagedMemory<V> },
}

impl<V: Copy> Default for MaybeCowMemory<V> {
    fn default() -> Self {
        Self::Owned(PagedMemory::default())
    }
}

impl<V: Copy> MaybeCowMemory<V> {
    pub(crate) fn get(&self, addr: u32) -> Option<&V> {
        match self {
            Self::Owned(memory) => memory.get(addr),
            Self::Cow { copy, original } => copy.get(addr).or_else(|| original.get(addr)),
        }
    }

    /// Gets the memory entry for the given address -- in COW mode, lazily duplicates the
    /// address's value from `original` into `copy` on first touch (if it existed there at all)
    /// before handing out `copy`'s own entry, mirroring `PagedMemory::entry`'s vacant/occupied
    /// semantics exactly from the caller's point of view.
    pub(crate) fn entry(&mut self, addr: u32) -> Entry<'_, V> {
        match self {
            Self::Owned(memory) => memory.entry(addr),
            Self::Cow { copy, original } => {
                if copy.get(addr).is_none() {
                    if let Some(&value) = original.get(addr) {
                        copy.insert(addr, value);
                    }
                }
                copy.entry(addr)
            }
        }
    }

    /// Returns an iterator over the occupied addresses. Only meaningful (and only called) once
    /// the whole run has finished -- by then any unconstrained block must have been exited for
    /// real (`Self::discard_cow`), so `Self::Owned` is the only state this should ever see.
    pub(crate) fn keys(&self) -> impl Iterator<Item = u32> + '_ {
        match self {
            Self::Owned(memory) => memory.keys(),
            Self::Cow { .. } => {
                unreachable!("keys() called while still in an unconstrained block")
            }
        }
    }

    /// Enters COW mode -- O(1), see the type's own doc comment. A no-op if already in COW mode.
    pub(crate) fn copy_on_write(&mut self) {
        if matches!(self, Self::Owned(_)) {
            let taken = std::mem::take(self);
            let Self::Owned(memory) = taken else { unreachable!() };
            *self = Self::Cow { copy: PagedMemory::default(), original: memory };
        }
    }

    /// Discards everything written since `copy_on_write` and reverts to the pre-block state --
    /// O(1), see the type's own doc comment. A no-op if not currently in COW mode.
    pub(crate) fn discard_cow(&mut self) {
        if let Self::Cow { original, .. } = self {
            *self = Self::Owned(std::mem::take(original));
        }
    }
}

/// An entry of `PagedMemory` or `Registers`, for in-place manipulation.
pub enum Entry<'a, V: Copy> {
    Vacant(VacantEntry<'a, V>),
    Occupied(OccupiedEntry<'a, V>),
}

impl<'a, V: Copy> Entry<'a, V> {
    /// Ensures a value is in the entry, inserting the provided value if necessary.
    /// Returns a mutable reference to the value.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Vacant(entry) => entry.insert(default),
            Entry::Occupied(entry) => entry.into_mut(),
        }
    }

    /// Ensures a value is in the entry, computing a value if necessary.
    /// Returns a mutable reference to the value.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Vacant(entry) => entry.insert(default()),
            Entry::Occupied(entry) => entry.into_mut(),
        }
    }

    /// Provides in-place mutable access to an occupied entry before any potential inserts into the
    /// map.
    pub fn and_modify<F: FnOnce(&mut V)>(mut self, f: F) -> Self {
        match &mut self {
            Entry::Vacant(_) => {}
            Entry::Occupied(entry) => f(entry.get_mut()),
        }
        self
    }
}

/// A vacant entry, for in-place manipulation.
pub struct VacantEntry<'a, V: Copy> {
    entry: &'a mut Option<V>,
}

impl<'a, V: Copy> VacantEntry<'a, V> {
    /// Insert a value into the `VacantEntry`, returning a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        // By construction, the slot in the page is `None`.
        *self.entry = Some(value);
        self.entry.as_mut().unwrap()
    }
}

/// An occupied entry, for in-place manipulation.
pub struct OccupiedEntry<'a, V> {
    entry: &'a mut V,
}

impl<'a, V: Copy> OccupiedEntry<'a, V> {
    /// Get a reference to the value in the `OccupiedEntry`.
    #[inline]
    pub fn get(&self) -> &V {
        self.entry
    }

    /// Get a mutable reference to the value in the `OccupiedEntry`.
    #[inline]
    pub fn get_mut(&mut self) -> &mut V {
        self.entry
    }

    /// Insert a value in the `OccupiedEntry`, returning the previous value.
    #[inline]
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.entry, value)
    }

    /// Converts the `OccupiedEntry` the into a mutable reference to the associated value.
    pub fn into_mut(self) -> &'a mut V {
        self.entry
    }

    /// Removes the value from the `OccupiedEntry` and returns it.
    pub fn remove(self) -> V {
        *self.entry
    }
}

impl<V: Copy> FromIterator<(u32, V)> for PagedMemory<V> {
    fn from_iter<T: IntoIterator<Item = (u32, V)>>(iter: T) -> Self {
        let mut mmu = Self::new_preallocated();
        for (k, v) in iter {
            mmu.insert(k, v);
        }
        mmu
    }
}

impl<V: Copy + 'static> IntoIterator for PagedMemory<V> {
    type Item = (u32, V);

    type IntoIter = Box<dyn Iterator<Item = Self::Item>>;

    fn into_iter(mut self) -> Self::IntoIter {
        Box::new(self.index.into_iter().enumerate().filter(|(_, i)| *i != NO_PAGE).flat_map(
            move |(i, index)| {
                let upper = i << LOG_PAGE_LEN;
                std::mem::take(&mut self.page_table[index as usize])
                    .0
                    .into_iter()
                    .enumerate()
                    .filter_map(move |(lower, v)| {
                        v.map(|v| (Self::decompress_addr(upper + lower), v))
                    })
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maybe_cow_memory_discards_writes_and_new_addresses_on_discard() {
        let mut mem: MaybeCowMemory<u32> = MaybeCowMemory::default();
        *mem.entry(4).or_insert(0) = 100;
        *mem.entry(8).or_insert(0) = 200;
        assert_eq!(mem.get(4), Some(&100));
        assert_eq!(mem.get(8), Some(&200));

        mem.copy_on_write();
        assert_eq!(mem.get(4), Some(&100), "reads must see through to the original while unwritten");
        assert_eq!(mem.get(8), Some(&200));

        // Overwrite an existing address.
        *mem.entry(4).or_insert(0) = 999;
        assert_eq!(mem.get(4), Some(&999), "reads must see the overlay once written");
        // Write a brand-new address that never existed before entering COW mode.
        *mem.entry(12).or_insert(0) = 777;
        assert_eq!(mem.get(12), Some(&777));
        // Untouched address must still read through to the original.
        assert_eq!(mem.get(8), Some(&200));

        mem.discard_cow();
        assert_eq!(mem.get(4), Some(&100), "discard must revert the overwritten address");
        assert_eq!(mem.get(8), Some(&200), "discard must leave the untouched address alone");
        assert_eq!(mem.get(12), None, "discard must make the never-before-existing address vanish again");
    }

    #[test]
    fn maybe_cow_memory_entry_and_modify_matches_owned_semantics() {
        // `entry().or_insert()`/`.and_modify()` must behave identically whether or not we're in
        // COW mode -- this is the exact pattern `MinimalExecutor::mr`/`mw` rely on.
        let mut mem: MaybeCowMemory<u32> = MaybeCowMemory::default();
        mem.copy_on_write();
        let v = mem.entry(16).or_insert(0);
        *v += 1;
        assert_eq!(mem.get(16), Some(&1));
        let v = mem.entry(16).or_insert(0);
        *v += 1;
        assert_eq!(mem.get(16), Some(&2));
    }

    #[test]
    fn maybe_cow_memory_nested_copy_on_write_is_a_no_op() {
        let mut mem: MaybeCowMemory<u32> = MaybeCowMemory::default();
        *mem.entry(4).or_insert(0) = 1;
        mem.copy_on_write();
        *mem.entry(8).or_insert(0) = 2;
        mem.copy_on_write(); // already in COW mode -- must not start a fresh overlay
        assert_eq!(mem.get(4), Some(&1));
        assert_eq!(mem.get(8), Some(&2));
        mem.discard_cow();
        assert_eq!(mem.get(4), Some(&1));
        assert_eq!(mem.get(8), None);
    }
}
