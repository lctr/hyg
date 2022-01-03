use crate::strenum;

use super::Location;

strenum! { Keyword is_kw ::
    Do      "do"
    Let     "let"
    In      "in"
    If      "if"
    Then    "then"
    Else    "else"
    Case    "case"
    Of      "of"
    Where   "where"
    Data    "data"
    Class   "class"
    Fn      "fn"
    Import  "import"
    Export  "export"
    InfixL  "infixl"
    InfixR  "infixr"
    Derive  "deriving"

}

strenum! { NumFlag is_num_flag ::
    Bin "b"
    Oct "o"
    Hex "x"
    Dec "."
    Sci "e"
    Int ""
}

impl Default for NumFlag {
    fn default() -> Self {
        Self::Int
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Assoc {
    Left,
    Right,
    // None,
}

impl Assoc {
    /// Utility matching functions as for brevity when using in parser
    #[inline]
    pub fn is_left(&self) -> bool {
        matches!(self, Self::Left)
    }

    #[inline]
    pub fn is_right(&self) -> bool {
        !self.is_left()
    }
}

strenum! { each BinOp is_binary ::
    // TODO!
    // forward composition
    // a -> (a -> b) -> b
    PipeR     "|>"   @ 1 L
    // backward composition
    // (a -> b) -> a -> b
    PipeL     "<|"   @ 0 R
    Or        "||"   @ 2 L
    And       "&&"   @ 3 L
    NotEq     "!="
    Equal     "=="
    Less      "<"
    LessEq    "<="
    Greater   ">"
    GreaterEq "=>"   @ 5 L
    Plus      "+"
    Link      "<>"   @ 6 L
    Minus     "-"    @ 7 L
    Times     "*"
    Div       "/"
    Rem       "%"
    Mod       "mod"  @ 8 L
    Pow       "**"
    Raise     "^"    @ 9 R
    //
    CompL     "</"   @ 10 R
    CompR     "\\>"  @ 10 L
    //
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operator {
    Reserved(BinOp),
    Custom(String),
}

impl Operator {
    #![allow(unused)]
    pub const MIN_PREC: u8 = 1;
    pub const MAX_PREC: u8 = 10;

    pub fn from_ident(token: Token) -> Option<Self> {
        if let Token::Ident(s) = token {
            Some(Operator::Custom(s))
        } else {
            None
        }
    }
}

impl From<BinOp> for Operator {
    fn from(op: BinOp) -> Self {
        Operator::Reserved(op)
    }
}

impl From<String> for Operator {
    fn from(op: String) -> Self {
        Operator::Custom(op)
    }
}

impl std::fmt::Display for Operator {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Operator::Reserved(x) => write!(f, "{}", x),
            Operator::Custom(x) => write!(f, "{}", x),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Comment {
    Line(String),
    Block(String),
    // TODO:
    // Doc(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Kw(Keyword),
    Operator(Operator),
    Comment(Comment),
    // Identifiers
    Ident(String),
    // non-value labels / symbols
    Sym(String),
    // Built-in identifiers
    // Meta(String),
    Char(char),
    Str(String),
    Bytes(Vec<u8>),
    Num {
        data: String,
        flag: NumFlag,
    },
    Lambda,
    ColonEq,
    Underscore,
    Eq,
    At,
    Dot,
    Dot2,
    Dot3,
    Semi,
    Colon,
    Colon2,
    Comma,
    Pound,
    Bang,
    ParenL,
    ParenR,
    BrackL,
    BrackR,
    CurlyL,
    CurlyR,
    Pipe,
    ArrowR,
    ArrowL,
    Invalid {
        data: String,
        msg: String,
        pos: Location,
    },
    Eof,
}

impl Token {
    /// Utility method for matching on `Token`'s of the `Operator` variant.
    pub fn as_operator(&self) -> Option<Operator> {
        if let Token::Operator(op) = self {
            Some(op.clone())
        } else {
            None
        }
    }
    /// While having the `Display` trait implemented (and hence getting
    /// `to_string()` from `ToString` for free) allows converting any `Token`
    /// into their string (= original) representation, there's no need to
    /// allocate a new string for variants that already contain their string
    /// representation.
    ///
    /// This method effectively implements `Into<String>`, where no new strings
    /// are allocated for variants containing their string representations.
    ///
    /// Fieldless variants, on the other hand, do allocate a new string in that
    /// they essentially call `ToString`'s `to_string` method.
    pub fn get_string(self) -> String {
        match self {
            Token::Operator(Operator::Custom(s))
            | Token::Ident(s)
            | Token::Sym(s)
            | Token::Str(s)
            | Token::Num { data: s, .. } => s,
            _ => self.to_string(),
        }
    }

    /// Utility method to extract the inner comment of a `Token::Comment` variant as an `Option` type.
    /// Returns None if `Token` is not a `Comment` variant.
    pub fn as_comment(self) -> Option<Comment> {
        if let Token::Comment(comment) = self {
            Some(comment)
        } else {
            None
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Token::Ident(..)
                | Token::Sym(..)
                | Token::Num { .. }
                | Token::Char(..)
                | Token::Str(..)
                | Token::Bytes(..)
                | Token::Operator(..)
                | Token::Kw(..)
                | Token::Lambda
                | Token::ParenL
                | Token::BrackL
        )
    }

    pub fn as_u8(&self) -> Result<u8, TokenError> {
        if let Token::Num {
            data,
            flag: NumFlag::Int,
        } = self
        {
            u8::from_str_radix(data.as_str(), 10)
                .map_err(|err| TokenError::ParseInt(err))
        } else {
            Err(TokenError::Incompatible(format!("Unable to read `{}` as u8! Token must be a `Num` variant with an `Int` flag.", self)))
        }
    }

    pub fn as_assoc_spec(
        &self,
    ) -> Result<Assoc, TokenError> {
        match self {
            Token::Kw(Keyword::InfixL) => Ok(Assoc::Left),
            Token::Kw(Keyword::InfixR) => Ok(Assoc::Right),
            t => Err(TokenError::Incompatible(format!("The token `{}` cannot be coerced into an Assoc variant!", t)))
        }
    }
}

impl std::convert::TryFrom<Token> for Assoc {
    type Error = TokenError;
    fn try_from(value: Token) -> Result<Self, Self::Error> {
        value.as_assoc_spec()
    }
}

#[derive(Clone, Debug)]
pub enum TokenError {
    ParseInt(std::num::ParseIntError),
    ParseFloat(std::num::ParseFloatError),
    Incompatible(String),
}

impl std::fmt::Display for TokenError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        fn g(
            f: &mut std::fmt::Formatter<'_>,
            e: impl std::fmt::Display,
        ) -> std::fmt::Result {
            write!(f, "Token error: {}", e)
        }

        match self {
            TokenError::ParseInt(e) => g(f, e),
            TokenError::ParseFloat(e) => g(f, e),
            TokenError::Incompatible(e) => g(f, e),
        }
    }
}

impl From<std::num::ParseIntError> for TokenError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::ParseInt(err)
    }
}

impl From<std::num::ParseFloatError> for TokenError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::ParseFloat(err)
    }
}

impl From<&str> for TokenError {
    fn from(err: &str) -> Self {
        Self::Incompatible(err.into())
    }
}

impl std::convert::TryFrom<Token> for u8 {
    type Error = TokenError;

    fn try_from(value: Token) -> Result<Self, Self::Error> {
        value.as_u8()
    }
}

impl From<BinOp> for Token {
    fn from(op: BinOp) -> Self {
        Token::Operator(Operator::Reserved(op))
    }
}

impl From<Keyword> for Token {
    fn from(kw: Keyword) -> Self {
        Token::Kw(kw)
    }
}

impl Default for Token {
    fn default() -> Self {
        Token::Eof
    }
}

impl std::fmt::Display for Token {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Token::Kw(kw) => write!(f, "{}", kw.as_str()),
            Token::Comment(c) => write!(f, "{:?}", c),
            Token::Operator(Operator::Reserved(o)) => {
                write!(f, "{}", o)
            }
            Token::Operator(Operator::Custom(s))
            | Token::Ident(s)
            | Token::Str(s) => {
                write!(f, "{}", s)
            }
            Token::Sym(s) => write!(f, ":{}", s),
            Token::Char(c) => write!(f, "{}", c),
            Token::Num { data, .. } => {
                write!(f, "{}", data)
            }
            Token::Bytes(bytes) => write!(
                f,
                "{}",
                bytes
                    .iter()
                    .map(|b| *b as char)
                    .collect::<String>()
            ),
            Token::Lambda => write!(f, "\\"),
            Token::Eq => write!(f, "="),
            Token::Underscore => write!(f, "_"),
            Token::ColonEq => write!(f, ":="),
            Token::At => write!(f, "@"),
            Token::Dot => write!(f, "."),
            Token::Dot2 => write!(f, ".."),
            Token::Dot3 => write!(f, "..."),
            Token::Semi => write!(f, ";"),
            Token::Colon => write!(f, ":"),
            Token::Colon2 => write!(f, "::"),
            Token::Comma => write!(f, ","),
            Token::Pound => write!(f, "#"),
            Token::Bang => write!(f, "!"),
            Token::ParenL => write!(f, "("),
            Token::ParenR => write!(f, ")"),
            Token::BrackL => write!(f, "["),
            Token::BrackR => write!(f, "]"),
            Token::CurlyL => write!(f, "{}", '{'),
            Token::CurlyR => write!(f, "{}", '}'),
            Token::Pipe => write!(f, "|"),
            Token::ArrowR => write!(f, "->"),
            Token::ArrowL => write!(f, "<-"),
            Token::Invalid {
                data: val,
                msg,
                pos,
            } => write!(
                f,
                "Invalid: `{}`: {} at {}",
                val, msg, pos
            ),
            Token::Eof => write!(f, "\0"),
        }
    }
}
