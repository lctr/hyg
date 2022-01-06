use std::{collections::HashMap, mem};

use super::traits::Intern;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Symbol(u32);

impl std::fmt::Display for Symbol {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "#{}", &(self.0))
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, ":Symbol {}", &(self.0))
    }
}

impl Symbol {
    pub fn get(&self) -> u32 {
        self.0
    }
}

impl From<u32> for Symbol {
    fn from(idx: u32) -> Self {
        Self(idx)
    }
}

impl From<Symbol> for usize {
    fn from(Symbol(i): Symbol) -> Self {
        i as usize
    }
}

/// String interner. Instead of allocating a new string during the compilation
/// process, all strings are instead interned and mapped to instances of type
/// `Symbol`, which unlike `&str` and `String`, are [`Copy`] and additionally
/// more lightweight.
#[derive(Clone, Debug)]
pub struct Lexicon {
    map: HashMap<&'static str, Symbol>,
    vec: Vec<&'static str>,
    buf: String,
    full: Vec<String>,
}

impl Intern for Lexicon {
    type Key = Symbol;
    type Value = str;

    fn intern(&mut self, value: &Self::Value) -> Self::Key {
        Lexicon::intern(self, value)
    }
}

impl Lexicon {
    /// Initial value just randomly guessed.
    /// This could/should maybe be optimized later.
    pub const BASE_CAPACITY: usize = 100;

    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.next_power_of_two();
        Self {
            map: HashMap::default(),
            vec: Vec::new(),
            buf: String::with_capacity(cap),
            full: Vec::new(),
        }
    }

    pub fn intern(&mut self, string: &str) -> Symbol {
        if let Some(&id) = self.map.get(string) {
            return id;
        }

        let string = unsafe { self.alloc(string) };
        let id = Symbol::from(self.map.len() as u32);

        self.map.insert(string, id);
        self.vec.push(string);

        debug_assert!(self.lookup(id) == string);
        debug_assert!(self.intern(string) == id);

        id
    }

    pub fn lookup(&self, id: Symbol) -> &str {
        self.vec[id.get() as usize]
    }

    unsafe fn alloc(
        &mut self,
        symbol: &str,
    ) -> &'static str {
        let cap = self.buf.capacity();
        if cap < self.buf.len() + symbol.len() {
            // just doubling isn't enough -- need to ensure the new string actually fits
            let new_cap = (cap.max(symbol.len()) + 1)
                .next_power_of_two();
            let new_buf = String::with_capacity(new_cap);
            let old_buf =
                mem::replace(&mut self.buf, new_buf);
            self.full.push(old_buf);
        }

        let interned = {
            let start = self.buf.len();
            self.buf.push_str(symbol);
            &self.buf[start..]
        };

        &*(interned as *const str)
    }
}

impl std::ops::Index<Symbol> for Lexicon {
    type Output = str;

    fn index(&self, index: Symbol) -> &Self::Output {
        self.lookup(index)
    }
}
