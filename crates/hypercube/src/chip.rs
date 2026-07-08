use std::{fmt::Display, hash::Hash, sync::Arc};
use thousands::Separable;

use slop_air::{Air, BaseAir, PairBuilder};
use slop_algebra::{Field, PrimeField32};
use slop_matrix::dense::RowMajorMatrix;
use slop_uni_stark::{get_max_constraint_degree, get_symbolic_constraints, SymbolicAirBuilder};

use crate::{
    air::{MachineAir, ZKMAirBuilder},
    lookup::{InteractionBuilder, Lookup, LookupKind},
    PROOF_MAX_NUM_PVS,
};

/// The maximum constraint degree for a chip.
pub const MAX_CONSTRAINT_DEGREE: usize = 3;

/// An Air that encodes lookups based on interactions.
pub struct Chip<F: Field, A> {
    pub air: Arc<A>,
    pub sends: Arc<Vec<Lookup<F>>>,
    pub receives: Arc<Vec<Lookup<F>>>,
    pub num_constraints: usize,
}

impl<F: Field, A: MachineAir<F>> std::fmt::Debug for Chip<F, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chip")
            .field("air", &self.air.name())
            .field("sends", &self.sends.len())
            .field("receives", &self.receives.len())
            .field("num_constraints", &self.num_constraints)
            .finish()
    }
}

impl<F: Field, A> Clone for Chip<F, A> {
    fn clone(&self) -> Self {
        Self {
            air: self.air.clone(),
            sends: self.sends.clone(),
            receives: self.receives.clone(),
            num_constraints: self.num_constraints,
        }
    }
}

impl<F: Field, A> Chip<F, A> {
    #[must_use]
    pub fn sends(&self) -> &[Lookup<F>] {
        &self.sends
    }

    #[must_use]
    pub fn receives(&self) -> &[Lookup<F>] {
        &self.receives
    }

    #[must_use]
    pub fn into_inner(self) -> Option<A> {
        Arc::into_inner(self.air)
    }
}

impl<F: PrimeField32, A: MachineAir<F>> Chip<F, A> {
    /// Returns whether the given chip is included in the execution record of the shard.
    pub fn included(&self, shard: &A::Record) -> bool {
        self.air.included(shard)
    }
}

impl<F, A> Chip<F, A>
where
    F: Field,
    A: BaseAir<F>,
{
    /// Records the interactions and constraint degree from the air and creates a new chip.
    pub fn new(air: A) -> Self
    where
        A: MachineAir<F> + Air<InteractionBuilder<F>> + Air<SymbolicAirBuilder<F>>,
    {
        let mut builder = InteractionBuilder::new(air.preprocessed_width(), air.width());
        air.eval(&mut builder);
        let (sends, receives) = builder.interactions();

        let nb_byte_sends = sends.iter().filter(|s| s.kind == LookupKind::Byte).count();
        let nb_byte_receives = receives.iter().filter(|r| r.kind == LookupKind::Byte).count();
        tracing::trace!("chip {} has {} byte interactions", air.name(), nb_byte_sends + nb_byte_receives);

        let mut max_constraint_degree = get_max_constraint_degree(&air, air.preprocessed_width(), PROOF_MAX_NUM_PVS);

        if !sends.is_empty() || !receives.is_empty() {
            max_constraint_degree = std::cmp::max(max_constraint_degree, MAX_CONSTRAINT_DEGREE);
        }
        assert!(max_constraint_degree > 0);
        assert!(max_constraint_degree <= MAX_CONSTRAINT_DEGREE);

        let num_constraints = get_symbolic_constraints(&air, air.preprocessed_width(), PROOF_MAX_NUM_PVS).len();

        let sends = Arc::new(sends);
        let receives = Arc::new(receives);
        let air = Arc::new(air);
        Self { air, sends, receives, num_constraints }
    }

    #[inline]
    #[must_use]
    pub fn num_interactions(&self) -> usize {
        self.sends.len() + self.receives.len()
    }

    #[inline]
    #[must_use]
    pub fn num_sent_byte_lookups(&self) -> usize {
        self.sends.iter().filter(|i| i.kind == LookupKind::Byte).count()
    }

    #[inline]
    #[must_use]
    pub fn num_sends_by_kind(&self, kind: LookupKind) -> usize {
        self.sends.iter().filter(|i| i.kind == kind).count()
    }

    #[inline]
    #[must_use]
    pub fn num_receives_by_kind(&self, kind: LookupKind) -> usize {
        self.receives.iter().filter(|i| i.kind == kind).count()
    }

    #[inline]
    #[must_use]
    pub fn cost(&self) -> u64
    where
        A: MachineAir<F>,
    {
        let preprocessed_cols = self.preprocessed_width();
        let main_cols = self.width();
        (preprocessed_cols + main_cols) as u64
    }
}

impl<F, A> BaseAir<F> for Chip<F, A>
where
    F: Field,
    A: BaseAir<F> + Send,
{
    fn width(&self) -> usize {
        self.air.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        panic!("Chip should not use the `BaseAir` method, but the `MachineAir` method.")
    }
}

impl<F, A> MachineAir<F> for Chip<F, A>
where
    F: Field,
    A: MachineAir<F>,
{
    type Record = A::Record;
    type Program = A::Program;
    type Error = A::Error;

    fn name(&self) -> String {
        self.air.name()
    }

    fn preprocessed_width(&self) -> usize {
        <A as MachineAir<F>>::preprocessed_width(&self.air)
    }

    fn generate_preprocessed_trace(&self, program: &A::Program) -> Option<RowMajorMatrix<F>> {
        <A as MachineAir<F>>::generate_preprocessed_trace(&self.air, program)
    }

    fn num_rows(&self, input: &A::Record) -> Option<usize> {
        <A as MachineAir<F>>::num_rows(&self.air, input)
    }

    fn generate_trace(&self, input: &A::Record, output: &mut A::Record) -> Result<RowMajorMatrix<F>, Self::Error> {
        self.air.generate_trace(input, output)
    }

    fn generate_dependencies(&self, input: &A::Record, output: &mut A::Record) -> Result<(), Self::Error> {
        self.air.generate_dependencies(input, output)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        self.air.included(shard)
    }

    fn commit_scope(&self) -> crate::air::LookupScope {
        self.air.commit_scope()
    }

    fn local_only(&self) -> bool {
        self.air.local_only()
    }

    fn local_only_row_sensitive(&self) -> bool {
        self.air.local_only_row_sensitive()
    }

    fn preprocessed_num_rows(&self, program: &A::Program, instrs_len: usize) -> Option<usize> {
        <A as MachineAir<F>>::preprocessed_num_rows(&self.air, program, instrs_len)
    }

    fn picus_info(&self) -> crate::air::PicusInfo {
        self.air.picus_info()
    }

    fn selectors_partition_real_rows(&self) -> bool {
        self.air.selectors_partition_real_rows()
    }

    fn picus_selector_specialization_allowed(&self, phase: &str, selector_name: &str) -> bool {
        self.air.picus_selector_specialization_allowed(phase, selector_name)
    }
}

// Implement AIR directly on Chip, evaluating the execution trace constraints.
impl<F, A, AB> Air<AB> for Chip<F, A>
where
    F: Field,
    A: Air<AB> + MachineAir<F>,
    AB: ZKMAirBuilder<F = F> + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        self.air.eval(builder);
    }
}

impl<F, A> PartialEq for Chip<F, A>
where
    F: Field,
    A: MachineAir<F>,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.air.name() == other.air.name()
    }
}

impl<F: Field, A: MachineAir<F>> Eq for Chip<F, A> where F: Field + Eq {}

impl<F, A> Hash for Chip<F, A>
where
    F: Field,
    A: MachineAir<F>,
{
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.air.name().hash(state);
    }
}

impl<F: Field, A: MachineAir<F>> PartialOrd for Chip<F, A>
where
    F: Field,
    A: MachineAir<F>,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Field, A: MachineAir<F>> Ord for Chip<F, A>
where
    F: Field,
    A: MachineAir<F>,
{
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name().cmp(&other.name())
    }
}

/// Statistics about a chip.
#[derive(Debug, Clone)]
pub struct ChipStatistics<F> {
    name: String,
    height: usize,
    preprocessed_cols: usize,
    main_cols: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> ChipStatistics<F> {
    #[must_use]
    pub fn new<A: MachineAir<F>>(chip: &Chip<F, A>, height: usize) -> Self {
        let name = chip.name();
        let preprocessed_cols = chip.preprocessed_width();
        let main_cols = chip.width();
        Self { name, height, preprocessed_cols, main_cols, _marker: std::marker::PhantomData }
    }

    #[must_use]
    #[inline]
    pub const fn total_width(&self) -> usize {
        self.preprocessed_cols + self.main_cols
    }

    #[must_use]
    #[inline]
    pub const fn total_number_of_cells(&self) -> usize {
        self.total_width() * self.height
    }

    #[must_use]
    #[inline]
    pub const fn total_memory_size(&self) -> usize {
        self.total_number_of_cells() * std::mem::size_of::<F>()
    }
}

impl<F: Field> Display for ChipStatistics<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<15} | Prep Cols = {:<5} | Main Cols = {:<5} | Rows = {:<5} | Cells = {:<10}",
            self.name,
            self.preprocessed_cols.separate_with_underscores(),
            self.main_cols.separate_with_underscores(),
            self.height.separate_with_underscores(),
            self.total_number_of_cells().separate_with_underscores()
        )
    }
}
