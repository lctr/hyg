use std::collections::HashMap;

use crate::prelude::either::Either;

use super::{Assoc, BinOp, Operator};

#[derive(
    Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub struct Prec(u8);

impl Prec {
    pub const MAX: Self = Self(Operator::MAX_PREC);
    pub const LAST: Self = Self(0);

    pub fn new(precedence: u8) -> Self {
        Self(precedence)
    }
}

impl From<u8> for Prec {
    fn from(prec: u8) -> Self {
        Self(prec)
    }
}

impl From<Prec> for u8 {
    fn from(Prec(prec): Prec) -> Self {
        prec
    }
}

/// Configuration for operater precedence and associativity. This is the interface unifying predefined operator data (from `BinOp`) with user defined operators
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fixity {
    pub(crate) assoc: Assoc,
    pub(crate) prec: Prec,
}

impl Default for Fixity {
    fn default() -> Self {
        Self {
            assoc: Assoc::Left,
            prec: Prec::MAX,
        }
    }
}

impl From<Either<Prec, Prec>> for Fixity {
    fn from(either: Either<Prec, Prec>) -> Self {
        either.resolve(
            |prec| Fixity {
                assoc: Assoc::Left,
                prec,
            },
            |prec| Fixity {
                assoc: Assoc::Right,
                prec,
            },
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixityTable {
    operators: HashMap<Operator, Fixity>,
}

impl FixityTable {
    #[allow(unused)]
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    #[allow(unused)]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            operators: HashMap::with_capacity(cap),
        }
    }

    /// Insert a new (Operator, Fixity) pair to the operators map.
    /// This does not check whether a given `Operator` already exists
    /// in the map, overwriting it if so.
    pub fn insert(
        &mut self,
        operator: Operator,
        fixity: Fixity,
    ) {
        self.operators.insert(operator, fixity);
    }

    pub fn get(
        &mut self,
        operator: &Operator,
    ) -> Option<&Fixity> {
        self.operators.get(operator)
    }
}

impl Default for FixityTable {
    fn default() -> Self {
        Self {
            operators: [
                BinOp::Or,
                BinOp::And,
                BinOp::NotEq,
                BinOp::Equal,
                BinOp::Less,
                BinOp::LessEq,
                BinOp::Greater,
                BinOp::GreaterEq,
                BinOp::Plus,
                BinOp::Link,
                BinOp::Minus,
                BinOp::Times,
                BinOp::Div,
                BinOp::Rem,
                BinOp::Mod,
                BinOp::Pow,
                BinOp::Raise,
                BinOp::PipeL,
                BinOp::PipeR,
                BinOp::CompL,
                BinOp::CompR,
            ]
            .iter()
            .map(|op| {
                (
                    (*op).into(),
                    Fixity {
                        assoc: op.get_assoc(),
                        prec: (op.get_prec() as u8).into(),
                    },
                )
            })
            .collect(),
        }
    }
}

mod test {}
