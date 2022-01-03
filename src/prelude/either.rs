#![allow(unused)]

pub enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Either<L, R> {
    pub fn is_left(&self) -> bool {
        matches!(self, Self::Left(_))
    }

    pub fn is_right(&self) -> bool {
        !self.is_left()
    }

    /// Given two closures, both of which have the same return type, applies
    /// whichever one is compatible with the inner type, returning the
    /// resolved value instead of an `Either` type.
    pub fn resolve<F, G, X>(self, left: F, right: G) -> X
    where
        F: FnOnce(L) -> X,
        G: FnOnce(R) -> X,
    {
        match self {
            Either::Left(l) => left(l),
            Either::Right(r) => right(r),
        }
    }

    /// Consumes the inner value of type `L` and returns it as n `Option<L>`.
    /// If `self` contained a value of type `R`, `None` is returned.
    pub fn take_left(self) -> Option<L> {
        if let Either::Left(l) = self {
            Some(l)
        } else {
            None
        }
    }

    /// Consumes inner value of type `R` and returns it as an `Option<R>`.
    /// If `self` contained a value of type `L`, `None` is returned.
    pub fn take_right(self) -> Option<R> {
        if let Either::Right(r) = self {
            Some(r)
        } else {
            None
        }
    }

    /// Applies a function to the wrapped `Left` value, leaving the `Right`
    /// value untouched.
    pub fn map_left<F, X>(self, f: F) -> Either<X, R>
    where
        F: FnOnce(L) -> X,
    {
        match self {
            Self::Left(l) => Either::Left(f(l)),
            Self::Right(r) => Either::Right(r),
        }
    }

    /// Applies a function to the wrapped `Right` value, leaving the `Left`
    /// value untouched.
    pub fn map_right<F, Y>(self, f: F) -> Either<L, Y>
    where
        F: FnOnce(R) -> Y,
    {
        match self {
            Self::Left(l) => Either::Left(l),
            Self::Right(r) => Either::Right(f(r)),
        }
    }

    /// Returns an `Either` variant containing a reference to the inner value.
    pub fn as_ref(&self) -> Either<&L, &R> {
        match *self {
            Self::Left(ref l) => Either::Left(l),
            Self::Right(ref r) => Either::Right(r),
        }
    }

    /// Returns an `Either` variant containing a mutable reference to the inner
    /// value.
    pub fn as_mut(&mut self) -> Either<&mut L, &mut R> {
        match *self {
            Self::Left(ref mut l) => Either::Left(l),
            Self::Right(ref mut r) => Either::Right(r),
        }
    }

    /// Consumes the inner value and returns it in the alternate `Either`
    /// variant.
    pub fn transpose(self) -> Either<R, L> {
        match self {
            Self::Left(lr) => Either::Right(lr),
            Self::Right(rl) => Either::Left(rl),
        }
    }

    /// Makes a new copy of the inner value, returning it in the same `Either`
    /// variant.
    pub fn copied(&self) -> Either<L, R>
    where
        L: Copy,
        R: Copy,
    {
        match self {
            Either::Left(l) => Either::Left(*l),
            Either::Right(r) => Either::Right(*r),
        }
    }

    /// Clones the inner value and returns it in the same `Either` variant.
    pub fn cloned(&self) -> Either<L, R>
    where
        L: Clone,
        R: Clone,
    {
        match self {
            Either::Left(l) => Either::Left(l.clone()),
            Either::Right(r) => Either::Right(r.clone()),
        }
    }

    /// Converts the inner `IntoIterator` value into an `Iterator`, returning
    /// it wrapped in the corresponding `Either` variant.
    pub fn into_iter(
        self,
    ) -> Either<L::IntoIter, R::IntoIter>
    where
        L: IntoIterator,
        R: IntoIterator<Item = L::Item>,
    {
        match self {
            Self::Left(l) => Either::Left(l.into_iter()),
            Self::Right(r) => Either::Right(r.into_iter()),
        }
    }
}
