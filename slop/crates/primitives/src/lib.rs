use core::marker::PhantomData;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FriConfig<F> {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    _marker: PhantomData<F>,
}

impl<F> Copy for FriConfig<F> {}

impl<F> Clone for FriConfig<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> FriConfig<F> {
    #[inline]
    pub const fn new(log_blowup: usize, num_queries: usize, proof_of_work_bits: usize) -> Self {
        Self { log_blowup, num_queries, proof_of_work_bits, _marker: PhantomData }
    }

    /// This FRI config relies on a conjecture for achieving 100 bits of security. Given Gruen and
    /// Diamond's recent result, we have increased the number of queries to 94 to be on the safe
    /// side. (With the original conjecture, we would achieve 100 bits of security with 84
    /// queries and 16 bits of grinding.)
    pub fn default_fri_config() -> Self {
        Self::new(1, 94, 16)
    }

    #[inline]
    pub const fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    #[inline]
    pub const fn num_queries(&self) -> usize {
        self.num_queries
    }

    #[inline]
    pub const fn proof_of_work_bits(&self) -> usize {
        self.proof_of_work_bits
    }
}
