// required by string interner `new_uninit_slice` [`Box`] method
#![feature(new_uninit)]
#![feature(maybe_uninit_slice)]
#![feature(test)]

mod compiler;
mod prelude;

fn main() {
    println!("Hello, world!");
}

mod benchy {
    // extern crate test;
    // use test::Bencher;
}
