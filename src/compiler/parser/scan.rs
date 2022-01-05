use crate::prelude::{
    either::Either, span::Positioned, traits::Peek,
};

pub trait Consume<'t>: Peek + Positioned {
    type Error: std::fmt::Display;

    /// Consume the provided token and return a `Result` containing a mutable reference to self if successful, or a `Self::Error` describing the error otherwise.
    fn eat(
        &mut self,
        token: &Self::Peeked,
    ) -> Result<&mut Self, Self::Error>;

    /// Returns the next item unwrapped. The underlying item to be iterated must be equivalently "null-terminated". By convention, this method should return the return value of `T::default()` if the `Option<&T>` returned by `peek` is `None`, otherwise it returns `T`,
    fn take_next(&mut self) -> Self::Peeked;

    /// Consumes the next item and applies the provided closure to the
    /// unerlying receiver along with the newly consumed item.
    ///
    /// Returns the result of the provided closure with a mutable reference
    /// to the underlying container, wrapped in a `Result`.
    ///
    /// Equivalent to
    /// ```
    /// let next = self.take_next();
    /// /* do something to `self` depending on `next` */
    /// ```
    ///  
    fn with_next<F, X>(
        &mut self,
        mut f: F,
    ) -> Result<X, Self::Error>
    where
        F: FnMut(
            Self::Peeked,
            &mut Self,
        ) -> Result<X, Self::Error>,
    {
        let tok = self.take_next();
        f(tok, self)
    }

    // TODO: redo this, probably factor out error reporting.
    /// Generate an error describing the expected token, the actual token, and the position in the stream. The position is given in an `Option` type -- this is to allow flexibility in position reporting.
    ///
    /// Since it may be more convenient to capture the position in the stream prior to running into an error, any `Position<u32>` provided in a `Some` variant will be used. In the event a `None` variant is passed in, the position by default will be called using the supertrait `Positioned`.  This has the effect that the position reported will correspond to the position in the stream *after which* the error was encountered.
    ///
    /// *Tldr;* Provide custom location wrapped in a `Some` variant for accurate error reporting, as the default position may correspond to the position in the stream *after which* the error was encountered.
    // &mut self since many calling methods will have borrowed self as mutable.
    fn unexpected(
        &mut self,
        expected: &Self::Peeked,
        actual: &Self::Peeked,
        pos: Option<Self::Loc>,
    ) -> Self::Error;
}

/// Minimum constraint necessary by higher order parsing methods.
///
/// This is a supertrait of `Consume`, which itself is a supertrait of `Peek`,
/// and `Positioned`. Methods in this trait do not directly produce a parent
/// node, but are instead used to compose parser actions, the results of which
/// are then used to form an expression.
///
/// Most (if not all) methods in this trait are implemented by default thanks
/// its chain of super traits, i.e., are effectively built on the Peek trait,
/// which itself is a flavor of iterator (the `peekable` method on any iterator
/// produces a peekable ierator, for which the implementation of `Peek` is
/// nearly trivial).
pub trait Combinator<'t>: Consume<'t> {
    /// Given a function `parse :: &mut Self -> Result<X, E>`, return the vector of nodes of type `X` obtained by applying `parse` repeatedly,
    /// initially after consuming a token matching `start` parameter, after
    /// each delimiter matching the `sep` parameter, and terminating upon
    /// encountering a token matching the `end` parameter.
    ///
    /// For example, this method would be used with a parser `f` that parses
    /// numeric tokens to parse the input `[1, 2, 3]`.
    ///
    /// On failure, all consumed tokens are discared and the error
    ///     `E = Self::Error`
    /// is returned.
    fn delimited<F, X>(
        &mut self,
        start: Self::Peeked,
        sep: Self::Peeked,
        end: Self::Peeked,
        parse: F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        self.eat(&start)
            .and_then(|p| p.sep_by_until(sep, end, parse))
    }

    /// Given a parser, a separator and an end delimiter, this applies
    /// the parser and consumes the separator repeatedly until encountering
    /// the end delimiter. When it reaches the end delimiter, it will
    /// consume it and return the results in a vector.
    fn sep_by_until<F, X>(
        &mut self,
        sep: Self::Peeked,
        end: Self::Peeked,
        mut parse: F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![];
        let mut first = true;
        while !self.is_done() {
            if self.match_curr(&end) {
                break;
            };
            if first {
                first = false;
            } else {
                self.eat(&sep)?;
            };
            if self.match_curr(&end) {
                break;
            };
            nodes.push(parse(self)?);
        }
        self.eat(&end)?;
        Ok(nodes)
    }

    /// First applies the given parser, storing the result in a vector, and
    /// then checks to see if a separator `sep` is matched. If it is, it will
    /// consume the separator and apply the parser again, appending the result
    /// to the forementioned vector.
    /// The resulting vector of parsed nodes is returned when failing to match
    /// the given separator.
    ///
    /// *NOTE:* This method **assumes** the parser is to be run *before* the
    /// given separator.
    fn many_sep_by<F, X>(
        &mut self,
        sep: Self::Peeked,
        mut parse: F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![parse(self)?];
        while !self.is_done() && self.match_curr(&sep) {
            self.eat(&sep)?;
            nodes.push(parse(self)?);
        }
        Ok(nodes)
    }

    /// Checks the current token, applying the parser if a match
    /// is found **without consuming the separator**.  
    fn try_many_sep_by<F, X>(
        &mut self,
        sep: Self::Peeked,
        mut parse: F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![];
        loop {
            if self.is_done() || !self.match_curr(&sep) {
                break;
            }

            nodes.push(parse(self)?)
        }
        Ok(nodes)
    }

    /// Applies the given parser repeatedly until failure.
    /// Returns a tuple of the results and a mutable reference to
    /// the underlying container, wrapped in an `Ok` variant on success.
    ///
    /// If the parer fails on the first try, an error will be
    /// immediately returned. Otherwise, non-initial errors are ignored.
    fn many<F, X>(
        &mut self,
        mut parse: F,
    ) -> Result<(Vec<X>, &mut Self), Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![parse(self)?];
        while let Ok(x) = parse(self) {
            nodes.push(x);
        }
        Ok((nodes, self))
    }

    /// Given a predicate and a parser, apply the parser
    /// repeatedly as long as the predicate returns `true`.
    ///
    /// This method will always check the predicate prior to running
    /// the given parser.
    ///
    /// **Note:** the given predicate *must* be coercible to
    /// `fn` pointer, and hence **must not** capture any variables.
    fn many_while<F, X>(
        &mut self,
        pred: fn(&Self::Peeked) -> bool,
        mut parse: F,
    ) -> Result<(Vec<X>, &mut Self), Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![];
        while let Some(true) =
            self.peek().and_then(|t| Some(pred(t)))
        {
            nodes.push(parse(self)?);
        }
        Ok((nodes, self))
    }

    /// Given a number `max_empty_rows`, this method applies
    /// the parser `f` repeatedly while the diference in rows between
    /// each application is less than the given max `max_empty_rows`.
    ///
    /// For example, providing `2` as the `max_empty_rows` parameter,
    /// all content with *less* than 2 empty lines between them are
    /// parsed and included.
    fn row_sep<F, X>(
        &mut self,
        max_empty_rows: u32,
        mut f: F,
    ) -> Result<(Vec<X>, &mut Self), Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let mut nodes = vec![];
        let mut row = self.get_row();
        nodes.push(f(self)?);
        while self.get_row() - row < max_empty_rows {
            row = self.get_row();
            nodes.push(f(self)?);
        }
        Ok((nodes, self))
    }

    /// Repeats the same parser as long as the column position (when this
    /// method was originally called) is maintained.
    ///  
    /// If an error is encountered, all parsed nodes are discarded and an
    /// error is returned, otherwise the results are collected and returned
    /// in a vector.
    fn many_col_aligned<F, X>(
        &mut self,
        mut f: F,
    ) -> Result<Vec<X>, Self::Error>
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let col0 = self.get_column();
        let mut nodes = vec![];
        nodes.push(f(self)?);
        while !self.is_done() && self.get_column() == col0 {
            nodes.push(f(self)?)
        }
        Ok(nodes)
    }

    /// Consunes and discards the specified token. Returns a mutable reference
    /// to `self` for chaining.
    fn ignore<F, X>(
        &mut self,
        tok: &Self::Peeked,
    ) -> &mut Self
    where
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        if self.match_curr(tok) {
            self.take_next();
        }
        self
    }
}

pub trait Layout<'t>: Combinator<'t> {
    /// Returns the `row` value of the given location IFF:
    ///
    /// * The `row` values of both the given and the current location are
    ///   the same
    /// * The `col` value of the current location is strictly *higher*
    ///   than that of the given location
    ///
    /// Otherwise, it returns the `row` value of the current location.
    ///
    /// This is a utility method used to generate the base row value used
    /// when parsing based on layout and an initial "effective" `row` value.
    ///
    fn fresh_row(&mut self, start: Self::Loc) -> u32 {
        let start_row = start.get_row();
        let curr_row = self.get_row();
        if start_row == curr_row
            || self.get_column() > start.get_column()
        {
            start_row
        } else {
            curr_row
        }
    }

    /// Given a predicate, and a closure, this method uses position to
    /// determine whether to apply the closure once, yielding an `Either::Left`
    /// with the result, or repeatedly, yielding an `Either::Right` with the
    /// first result and an array of the later results in a tuple.
    ///
    /// Assuming no errors were encountered, the above results are returned in
    /// a `Result::Ok` variant, otherwise short-circuiting and returning an
    /// error.
    ///
    /// This method will only repeat the closure provided as long as the
    /// indentation after each application **exceeds** the original indentation
    /// prior to applying the closure for the first time.
    ///
    ///
    /// The aim of this method is to satisfy the `off-side` rule, wherein a
    ///
    fn within_offside<P, F, X>(
        &mut self,
        // start: Self::Loc,
        pred: P,
        mut parse: F,
    ) -> Result<Either<X, (X, Vec<X>)>, Self::Error>
    where
        P: Fn(&Self::Peeked) -> bool,
        F: FnMut(&mut Self) -> Result<X, Self::Error>,
    {
        let start = self.loc();
        let start_row = self.get_row();
        let start_col = self.get_column();
        let head = parse(self)?;

        if self.is_done()
            || self
                .peek()
                .and_then(|t| Some(!pred(t)))
                .unwrap_or_else(|| false)
            || self.get_column() <= start_col
            || self.get_row() > start_row
        {
            return Ok(Either::Left(head));
        }

        let mut nodes = vec![];
        let mut row = self.fresh_row(start);

        while !self.is_done()
            && self.get_row() == row
            && self
                .peek()
                .and_then(|t| Some(pred(t)))
                .unwrap_or_else(|| false)
        {
            nodes.push(parse(self)?);
            if self.get_column() > start_col {
                row = self.get_row()
            }
        }

        Ok(Either::Right((head, nodes)))
    }
}
