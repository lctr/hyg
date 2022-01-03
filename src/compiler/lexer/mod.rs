mod lexer;
mod source;
mod token;

pub use lexer::Lexer;
pub use source::Peek;
pub use token::{
    Assoc, BinOp, Comment, Keyword, NumFlag, Operator,
    Token, TokenError,
};

// helpful re-imports
pub use crate::prelude::span::{Location, Span};
