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

    pub fn take_left(self) -> Option<L> {
        if let Either::Left(l) = self {
            Some(l)
        } else {
            None
        }
    }

    pub fn take_right(self) -> Option<R> {
        if let Either::Right(r) = self {
            Some(r)
        } else {
            None
        }
    }

    pub fn map_left<F, X>(self, f: F) -> Either<X, R>
    where
        F: FnOnce(L) -> X,
    {
        match self {
            Self::Left(l) => Either::Left(f(l)),
            Self::Right(r) => Either::Right(r),
        }
    }

    pub fn map_right<F, Y>(self, f: F) -> Either<L, Y>
    where
        F: FnOnce(R) -> Y,
    {
        match self {
            Self::Left(l) => Either::Left(l),
            Self::Right(r) => Either::Right(f(r)),
        }
    }

    pub fn as_ref(&self) -> Either<&L, &R> {
        match *self {
            Self::Left(ref l) => Either::Left(l),
            Self::Right(ref r) => Either::Right(r),
        }
    }

    pub fn as_mut(&mut self) -> Either<&mut L, &mut R> {
        match *self {
            Self::Left(ref mut l) => Either::Left(l),
            Self::Right(ref mut r) => Either::Right(r),
        }
    }

    pub fn transpose(self) -> Either<R, L> {
        match self {
            Self::Left(lr) => Either::Right(lr),
            Self::Right(rl) => Either::Left(rl),
        }
    }

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
