#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span(pub Location, pub Location);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Location {
    pub pos: u32,
    pub row: u32,
    pub col: u32,
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    pub fn reset_column(&mut self) {
        self.pos += 1;
        self.col += 1;
    }
    pub fn reset_row(&mut self) {
        self.pos += 1;
        self.row += 1;
        self.col = 0;
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
