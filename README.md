# Hyg
I've really got an uncreative naming scheme going on here. The name *Hyg* has no real meaning, and is simply a more "haskelly" version of *Ryg*, which itself was a Rust implementation of the original, TypeScript based language *Wyg*. 

Phew, what a journey. While this project aims to properly exist at some point as a programming language, it is primarily an outcome an interest in learning more about compilers and language design. **For what it's worth,** this personal project is *not* production friendly.

## Goals
While *Wyg* took advantage of the Deno runtime to form a naïve scripting web-based environment, *Ryg* was re-imagined as I learned Rust (and admittedly, was essentially my Rust playground project of sorts), *Hyg* aims to incorporate:

- Type inference
- Algebraic data types
- First class functions
- Parametric polymorphism
- Tree-walking interpreter for AST
- Bytecode interpreter
- An Actual Compiler, *ideally* targetting LLVM
- FFI between Rust, and Haskell, with some support for JavaScript
