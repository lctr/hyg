use std::cmp::Ordering;

use super::traits::Newtype;

#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
pub struct Span(Location, Location);

impl Span {
    pub fn new(start: Location, end: Location) -> Self {
        Self(start, end)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Spanned<T>(Span, T);

impl<T> Spanned<T> {
    pub fn new(span: Span, item: T) -> Self {
        Self(span, item)
    }

    pub fn pair<S>(
        self,
        rhs: Spanned<S>,
    ) -> Spanned<(T, S)> {
        Spanned(Span(self.0 .0, rhs.0 .1), (self.1, rhs.1))
    }
}

impl<T> std::cmp::PartialOrd for Spanned<T>
where
    T: PartialEq,
{
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Newtype for Spanned<T> {
    type Inner = T;
    fn take(self) -> Self::Inner {
        self.1
    }
    fn get_mut(&mut self) -> &mut Self::Inner {
        &mut self.1
    }
    fn get_ref(&self) -> &Self::Inner {
        &self.1
    }
}

#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
pub struct Location {
    pos: u32,
    pub row: u32,
    pub col: u32,
}

impl std::fmt::Display for Location {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}:{}", self.row, self.col)
    }
}

impl Default for Location {
    fn default() -> Self {
        Self {
            pos: 0,
            row: 1,
            col: 0,
        }
    }
}

impl Location {
    pub fn new() -> Self {
        Self {
            pos: 0,
            row: 1,
            col: 0,
        }
    }
    /// Increments the `pos` and `col` fields by 1, leaving the `row`
    /// field untouched.
    pub fn incr_column(&mut self) {
        self.pos += 1;
        self.col += 1;
    }
    /// Increments the `pos` and `row` fields by 1, resetting the `col`
    /// field to `0`.
    pub fn incr_row(&mut self) {
        self.pos += 1;
        self.row += 1;
        self.col = 0;
    }
    /// Generates a dummy `Location` for situations in which the only
    /// requirement is the existence of a `Location`. Useful as an
    /// intermediate value.
    ///
    /// Note: dummy values can be identified by having a `row` value
    /// of `0`.
    /// This is not possible in regular `Location` structs, as a `Location`
    /// object is always initialized with a starting `row` value of `1`.
    pub fn dummy() -> Self {
        Self {
            pos: 0,
            row: 0,
            col: 0,
        }
    }

    /// Utility method for comparing `Location` structs
    pub fn row_eq(&self, rhs: &Self) -> bool {
        self.row == rhs.row
    }

    pub fn col_eq(&self, rhs: &Self) -> bool {
        self.col == rhs.col
    }

    /// First, we say a [`Location`] `x` *contains* a [`Location`] `y` if
    ///
    ///     x.row == y.row || (x.row < y.row && x.col < y.col)
    ///  
    /// i.e, any of the following hold:
    ///
    /// * `y` is on the same row (= line) as `x`
    /// * both `row` and `col` values of `y` are greater than that of `x`
    pub fn contains(&self, rhs: &Self) -> bool {
        self.row == rhs.row
            || (self.row < rhs.row && self.col < rhs.col)
    }
}

impl Positioned for Location {
    type Loc = Self;
    fn loc(&self) -> Self::Loc {
        *self
    }
    fn get_row(&self) -> u32 {
        self.row
    }
    fn get_column(&self) -> u32 {
        self.col
    }
    fn get_pos(&self) -> u32 {
        self.pos
    }
}

pub trait Positioned {
    type Loc: Positioned;
    /// Handle for underlying `Position<Idx>` item.
    fn loc(&self) -> Self::Loc;
    fn get_row(&self) -> u32 {
        self.loc().get_row()
    }
    fn get_column(&self) -> u32 {
        self.loc().get_column()
    }
    /// Returns the value of the `pos` field
    fn get_pos(&self) -> u32 {
        self.loc().get_pos()
    }
}
