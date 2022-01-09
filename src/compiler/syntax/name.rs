use crate::{
    compiler::lexer::TokenError, prelude::symbol::Symbol,
};

use super::{Operator, Token};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Name {
    /// Used for identifiers
    Ident(Symbol),
    /// Used for operators
    Infix(Operator),
    /// Used for data constructors
    Data(Symbol),
    /// Used for classes -- play same syntactic role as TyCons,
    /// but within constraints?
    Class(Symbol),
}

impl std::fmt::Display for Name {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Name::Ident(s)
            | Name::Data(s)
            | Name::Class(s) => {
                write!(f, "{}", s)
            }
            Name::Infix(x) => write!(f, "{}", x),
        }
    }
}

impl std::convert::TryFrom<Token> for Name {
    type Error = TokenError;
    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Lower(s) => Ok(Name::Ident(s)),
            Token::Upper(s) => Ok(Name::Data(s)),
            Token::Operator(o) => Ok(Name::Infix(o)),
            t => Err(TokenError::Incompatible(format!("Failed Token -> Var conversion! Expected either `Ident`, `Sym`, or `Operator` variant, but found {:?}", t)))
        }
    }
}
