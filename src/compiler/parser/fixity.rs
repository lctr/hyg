use std::collections::HashMap;

use super::{Assoc, BinOp, Operator};

/// Configuration for operater precedence and associativity. This is the interface unifying predefined operator data (from `BinOp`) with user defined operators
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fixity {
    pub(crate) assoc: Assoc,
    pub(crate) prec: u8,
}

impl Default for Fixity {
    fn default() -> Self {
        Self {
            assoc: Assoc::Left,
            prec: Operator::MAX_PREC,
        }
    }
}

#[derive(Debug)]
pub struct FixityTable {
    operators: HashMap<Operator, Fixity>,
}

impl FixityTable {
    pub fn new() -> Self {
        Self {
            operators: HashMap::new(),
        }
    }

    /// Insert a new (Operator, Fixity) pair to the operators map.
    /// This does not check whether a given `Operator` already exists
    /// in the map, overwriting it if so.
    pub fn insert(&mut self, operator: Operator, fixity: Fixity) {
        self.operators.insert(operator, fixity);
    }

    pub fn get(&mut self, operator: &Operator) -> Option<&Fixity> {
        self.operators.get(operator)
    }
}

impl Clone for FixityTable {
    fn clone(&self) -> Self {
        Self {
            operators: self.operators.clone(),
        }
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
            ]
            .iter()
            .map(|op| {
                (
                    (*op).into(),
                    Fixity {
                        assoc: op.get_assoc(),
                        prec: op.get_prec() as u8,
                    },
                )
            })
            .collect(),
        }
    }
}

mod test {}
