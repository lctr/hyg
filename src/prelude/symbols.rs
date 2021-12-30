#![allow(unused)]

use std::mem;
use std::ptr;

use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem::MaybeUninit,
};

/// Interner -- associates values with usize tags and allows bidirectional lookups

// Interned string
pub struct Symbol();

// stolen from https://doc.rust-lang.org/nightly/nightly-rustc/src/rustc_arena/lib.rs.html#1-607

// Single type arena
pub struct Arena<T> {
    // next object to be allocated
    ptr: Cell<*mut T>,
    // end of allocated area. reaching this allocates a new chunk
    end: Cell<*mut T>,
    chunks: RefCell<Vec<ArenaChunk<T>>>,
    // dropping the stadium causes all its owned instances of `T` to drop
    __: PhantomData<T>,
}

struct ArenaChunk<T> {
    storage: Box<[MaybeUninit<T>]>,
    // number of valid entries in chunk
    entries: usize,
}

impl<T> ArenaChunk<T> {
    #[inline]
    unsafe fn new(capacity: usize) -> Self {
        Self {
            storage: Box::new_uninit_slice(capacity),
            entries: 0,
        }
    }

    #[inline]
    unsafe fn destroy(&mut self, len: usize) {
        // The branch on needs_drop() is an -O1 performance optimization.
        // Without the branch, dropping TypedArena<u8> takes linear time.
        if std::mem::needs_drop::<T>() {
            ptr::drop_in_place(MaybeUninit::slice_assume_init_mut(&mut self.storage[..len]));
        }
    }

    // Returns a pointer to the first allocated object
    #[inline]
    fn start(&mut self) -> *mut T {
        MaybeUninit::slice_as_mut_ptr(&mut self.storage)
    }

    // Returns a pointer to the end of the allocated space
    #[inline]
    fn end(&mut self) -> *mut T {
        unsafe {
            if mem::size_of::<T>() == 0 {
                // A pointer as large as possible for zero-sized elements.
                !0 as *mut T
            } else {
                self.start().add(self.storage.len())
            }
        }
    }
}

// Stadium starts with PAGE-sized chunks, doubling in size until reaching PAGE_MAX size
const PAGE: usize = 4096;
const PAGE_MAX: usize = 2 * 1024 * 1024;

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self {
            // start both `ptr` and `end` at 0 in order to trigger growth upon allocation
            ptr: Cell::new(ptr::null_mut()),
            end: Cell::new(ptr::null_mut()),
            chunks: Default::default(),
            __: PhantomData,
        }
    }
}

trait IterExt<T> {
    fn alloc_from_iter(self, stadium: &Arena<T>) -> &mut [T];
}
