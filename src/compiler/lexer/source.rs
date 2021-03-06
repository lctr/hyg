use std::{iter::Peekable, str::Chars};

pub use crate::prelude::{
    span::{Location, Positioned},
    traits::Peek,
};

#[derive(Clone, Debug)]
pub struct Source<'s> {
    // bytes: &'s [u8],
    src: &'s str,
    loc: Location,
    chars: Peekable<Chars<'s>>,
}

impl<'s> Source<'s> {
    /// Line feed: Reset row += 1
    const LF: char = '\n';
    /// Tab character: Treated as two spaces
    #[allow(unused)]
    const TAB: char = '\t';
    /// Carriage-return: Ignored; NO-OP
    const CR: char = '\r';

    pub fn new(src: &'s str) -> Self {
        Self {
            src,
            // bytes: src.as_bytes(),
            loc: Location::new(),
            chars: src.chars().peekable(),
        }
    }

    fn sync_pos(&mut self, c: char) {
        match c {
            Self::LF => self.loc.incr_row(),
            Self::TAB => {
                // tabs are treated as 2 spaces
                // NOTE: maybe this isn't the best place for this, since we may
                // not want to modify how our position changes outside of lexer?
                for _ in 0..2 {
                    self.loc.incr_column();
                }
            }
            Self::CR => {}
            _ => self.loc.incr_column(),
        }
    }
}

impl<'s> From<&'s str> for Source<'s> {
    fn from(src: &'s str) -> Self {
        Source::new(src)
    }
}

impl<'s> From<&'s String> for Source<'s> {
    fn from(s: &'s String) -> Self {
        Self::from(s.as_str())
    }
}

impl<'t> Positioned for Source<'t> {
    type Loc = Location;
    fn loc(&self) -> Self::Loc {
        self.loc
    }
}

impl<'t> Peek for Source<'t> {
    type Peeked = char;

    #[inline]
    fn peek(&mut self) -> Option<&Self::Peeked> {
        self.chars.peek()
    }
    fn is_done(&mut self) -> bool {
        self.chars.peek().is_none()
    }
}

impl<'t> Iterator for Source<'t> {
    type Item = char;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.chars.next();
        if let Some(c) = &next {
            self.sync_pos(*c);
        };
        next
    }
}

#[cfg(test)]
mod test {
    // extern crate test;
    // use test::Bencher;
    use super::*;

    #[test]
    fn test_for_each() {
        let src = "hello world";
        let stream = Source::new(src);
        stream
            .zip(src.chars())
            .for_each(|(x, y)| assert_eq!(x, y));
    }

    #[test]
    fn test_peek() {
        let mut stream = Source::new("hi");
        assert_eq!(stream.peek(), Some(&'h'));
        stream.next();
        assert_eq!(stream.peek(), Some(&'i'));
        stream.next();
        assert_eq!(true, stream.is_done());
    }

    #[test]
    fn test_done() {
        let mut stream = Source::new("hi");
        stream.next();
        stream.next();
        assert_eq!(true, stream.is_done());
    }

    #[test]
    fn buffered_file() {
        use std::io::Read;
        let file = std::fs::File::open("Cargo.toml");
        if let Ok(file) = file {
            let mut buf_reader =
                std::io::BufReader::new(file);
            let mut contents = String::new();
            let _ =
                buf_reader.read_to_string(&mut contents);
            println!("contents {}", contents);
        }
    }
}
