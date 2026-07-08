use derive_where::derive_where;
use slop_algebra::FieldAlgebra;
use slop_alloc::{Backend, CpuBackend, HasBackend};
use slop_tensor::Tensor;
use std::borrow::{Borrow, BorrowMut};

#[derive(Debug, Clone)]
#[derive_where(PartialEq, Eq, Serialize, Deserialize; Tensor<F, A>)]
pub struct RsCodeWord<F, A: Backend = CpuBackend> {
    pub data: Tensor<F, A>,
}

impl<F: FieldAlgebra, A: Backend> RsCodeWord<F, A> {
    pub const fn new(data: Tensor<F, A>) -> Self {
        Self { data }
    }
}

impl<F: FieldAlgebra, A: Backend> Borrow<Tensor<F, A>> for RsCodeWord<F, A> {
    fn borrow(&self) -> &Tensor<F, A> {
        &self.data
    }
}

impl<F: FieldAlgebra, A: Backend> BorrowMut<Tensor<F, A>> for RsCodeWord<F, A> {
    fn borrow_mut(&mut self) -> &mut Tensor<F, A> {
        &mut self.data
    }
}

impl<F, A: Backend> HasBackend for RsCodeWord<F, A> {
    type Backend = A;

    #[inline]
    fn backend(&self) -> &Self::Backend {
        self.data.backend()
    }
}
