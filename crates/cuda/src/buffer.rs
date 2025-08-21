use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use anyhow::Context as _;
use cust::{
    memory::{DevicePointer, GpuBuffer},
    prelude::*,
};

use super::tracker::tracker;
use crate::scope;

pub trait Buffer<T>: Clone {
    fn name(&self) -> &'static str;

    fn size(&self) -> usize;

    fn slice(&self, offset: usize, size: usize) -> Self;

    fn get_at(&self, idx: usize) -> T;

    fn view<F: FnOnce(&[T])>(&self, f: F);

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F);

    fn to_vec(&self) -> Vec<T>;
}

struct RawBuffer {
    name: &'static str,
    buf: DeviceBuffer<u8>,
}

impl RawBuffer {
    pub fn new(name: &'static str, size: usize) -> Self {
        tracing::trace!("alloc: {size} bytes, {name}");
        tracker().lock().unwrap().alloc(size);
        let buf = unsafe { DeviceBuffer::uninitialized(size) }
            .context(format!("allocation failed on {name}: {size} bytes"))
            .unwrap();
        Self { name, buf }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        tracing::trace!("free: {} bytes, {}", self.buf.len(), self.name);
        tracker().lock().unwrap().free(self.buf.len());
    }
}

#[derive(Clone)]
pub struct CudaBuffer<T> {
    buffer: Rc<RefCell<RawBuffer>>,
    size: usize,
    offset: usize,
    marker: PhantomData<T>,
}

#[inline]
fn unchecked_cast<A, B>(a: &[A]) -> &[B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const B, new_len) }
}

#[inline]
fn unchecked_cast_mut<A, B>(a: &mut [A]) -> &mut [B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut B, new_len) }
}

impl<T> CudaBuffer<T> {
    pub fn new(name: &'static str, size: usize) -> Self {
        let bytes_len = std::mem::size_of::<T>() * size;
        assert!(bytes_len > 0);
        CudaBuffer {
            buffer: Rc::new(RefCell::new(RawBuffer::new(name, bytes_len))),
            size,
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn copy_from(name: &'static str, slice: &[T]) -> Self {
        // scope!("copy_from");
        let bytes_len = std::mem::size_of_val(slice);
        assert!(bytes_len > 0);
        let mut buffer = RawBuffer::new(name, bytes_len);
        let bytes = unchecked_cast(slice);
        buffer.buf.copy_from(bytes).unwrap();

        CudaBuffer {
            buffer: Rc::new(RefCell::new(buffer)),
            size: slice.len(),
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn as_device_ptr(&self) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow_mut().buf.as_device_ptr();
        let offset = self.offset * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    pub fn as_device_ptr_with_offset(&self, offset: usize) -> DevicePointer<u8> {
        let ptr = self.buffer.borrow_mut().buf.as_device_ptr();
        let offset = (self.offset + offset) * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }
}

impl<T: Clone> Buffer<T> for CudaBuffer<T> {
    fn name(&self) -> &'static str {
        self.buffer.borrow().name
    }

    fn size(&self) -> usize {
        self.size
    }

    fn slice(&self, offset: usize, size: usize) -> CudaBuffer<T> {
        assert!(offset + size <= self.size());
        CudaBuffer {
            buffer: self.buffer.clone(),
            size,
            offset: self.offset + offset,
            marker: PhantomData,
        }
    }

    fn get_at(&self, idx: usize) -> T {
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow_mut();
        let offset = (self.offset + idx) * item_size;
        let ptr = unsafe { buf.buf.as_device_ptr().offset(offset as isize) };
        let device_slice = unsafe { DeviceSlice::from_raw_parts(ptr, item_size) };
        let host_buf = device_slice.as_host_vec().unwrap();
        let slice: &[T] = unchecked_cast(&host_buf);
        slice[0].clone()
    }

    fn view<F: FnOnce(&[T])>(&self, f: F) {
        scope!("view");
        let item_size = std::mem::size_of::<T>();
        let buf = self.buffer.borrow_mut();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let ptr = unsafe { buf.buf.as_device_ptr().offset(offset as isize) };
        let device_slice = unsafe { DeviceSlice::from_raw_parts(ptr, len) };
        let host_buf = device_slice.as_host_vec().unwrap();
        let slice = unchecked_cast(&host_buf);
        f(slice);
    }

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F) {
        scope!("view_mut");
        let mut buf = self.buffer.borrow_mut();
        let mut host_buf = buf.buf.as_host_vec().unwrap();
        let slice = unchecked_cast_mut(&mut host_buf);
        f(&mut slice[self.offset..]);
        buf.buf.copy_from(&host_buf).unwrap();
    }

    fn to_vec(&self) -> Vec<T> {
        let buf = self.buffer.borrow_mut();
        let host_buf = buf.buf.as_host_vec().unwrap();
        let slice = unchecked_cast(&host_buf);
        slice.to_vec()
    }
}
