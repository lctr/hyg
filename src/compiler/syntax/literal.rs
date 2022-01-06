use crate::prelude::either::Either;

use super::{NumFlag, Token};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Literal {
    Char(char),
    Str(String),
    Bytes(Vec<u8>),
    // identical to internal data for a `Token` of variant `Num`. Not read until evaluation as they may be overloaded, e.g., the flags `Dec` and `Sci` may become f32 or f64.
    Num { data: String, flag: NumFlag },
}

impl Literal {
    /// Note: this method is primarily used for pattern matching purposes,
    /// hence this matches `Token`s whose variants coincide with those of
    /// `Literal`, EXCLUDING the `Literal::Sym` variant.
    /// This is because the `Literal::Sym` variant plays a role in matching data constructors as well as record field keys
    pub fn is_token_literal(token: &Token) -> bool {
        use Token::*;
        matches!(
            token,
            Char(_) | Bytes(_) | Str(_) | Num { .. }
        )
    }

    /// Takes a `Token` and, if its variant corresponds to a `Literal` variant,
    /// returns it wrapped in a `Some` variant.
    /// Returns `None` otherwise.
    ///
    /// *Note:* Unlike the `is_token_literal` method, this method ACCEPTS `Sym` `Token` variants.
    pub fn from_token(token: Token) -> Option<Self> {
        match token {
            Token::Char(c) => Some(Self::Char(c)),
            Token::Str(s) => Some(Self::Str(s)),
            Token::Bytes(bs) => Some(Self::Bytes(bs)),
            Token::Num { data, flag } => {
                Some(Self::Num { data, flag })
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Literal::Char(c) => write!(f, "'{:?}'", c),
            Literal::Str(s) => write!(f, "\"{}\"", s),
            Literal::Bytes(bs) => write!(f, "{:?}", bs),
            Literal::Num { data, .. } => {
                write!(f, "{}", data)
            }
        }
    }
}

/// Error type for `Token` -> `Literal` conversions
#[derive(
    Debug, Copy, Clone, Hash, PartialEq, Eq, Default,
)]
pub struct LitErr;

impl LitErr {
    pub const MSG: &'static str =
        "The argument provided was unable to be converted into a `Literal`! \n\
        `Token`s only successfully convert into `Literal`s if their variant \n\
        is one of the following: `Sym`, `Char`, `Str`, `Bytes`, `Num`.";
}

impl std::fmt::Display for LitErr {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", Self::MSG)
    }
}

impl std::error::Error for LitErr {
    fn description(&self) -> &str {
        LitErr::MSG
    }
}

impl std::convert::TryFrom<Token> for Literal {
    type Error = LitErr;
    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Char(c) => Ok(Self::Char(c)),
            Token::Str(s) => Ok(Self::Str(s)),
            Token::Bytes(bs) => Ok(Self::Bytes(bs)),
            Token::Num { data, flag } => {
                Ok(Self::Num { data, flag })
            }
            _ => Err(LitErr),
        }
    }
}

impl std::convert::TryFrom<Token> for f64 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseFloatError if `Num` type
    type Error = Either<Token, std::num::ParseFloatError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<f64>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for f32 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseFloatError if `Num` type
    type Error = Either<Token, std::num::ParseFloatError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<f32>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for usize {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<usize>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for isize {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<isize>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for i8 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<i8>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for u16 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<u16>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for i16 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<i16>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for u32 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<u32>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for i32 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<i32>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for u64 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<u64>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for i64 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<i64>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}

impl std::convert::TryFrom<Token> for u128 {
    /// On failure, return the token if not a `Num` type,
    /// or a ParseIntError if `Num` type
    type Error = Either<Token, std::num::ParseIntError>;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        match value {
            Token::Num { data, .. } => {
                str::parse::<u128>(data.as_str())
                    .map_err(Either::Right)
            }
            _ => Err(Either::Left(value)),
        }
    }
}
