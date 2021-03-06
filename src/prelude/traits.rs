/// Trait extending `Peekable<T>` functionality to any type `I` that
/// behaves like (but is not required to be) an `Iterator<Item = T>`.
///
/// This trait *does not* require the base type to be an `Iterator`.
/// However, every `Iterator<T>` can automatically produce a type
/// whose `peek` method would not need to be implemented in order
/// to implement `Peek`.
pub trait Peek {
    type Peeked: std::cmp::PartialEq + Default;

    fn peek(&mut self) -> Option<&Self::Peeked>;

    fn is_done(&mut self) -> bool;

    fn match_curr(&mut self, item: &Self::Peeked) -> bool {
        if let Some(t) = self.peek() {
            item == t
        } else {
            false
        }
    }
}

/// Utility trait for structs following the `newtype` pattern.
///
/// Since newtypes generally don't have their inner data publically
/// accessible, it follows that relevant accessor methods must be
/// implemented for each newtype, leading to a lot of boilerplate.
///
/// This trait aims to standardize the accessor methods used on simple,
/// single field tuple structs (aka, structs following the `newtype`
/// design pattern).
pub trait Newtype {
    /// The type of the element wrapped by the newtype.
    type Inner;

    /// Standard accessor methods for the inner type.
    fn get_ref(&self) -> &Self::Inner;

    fn get_mut(&mut self) -> &mut Self::Inner;

    /// Consumes and returns the inner type.
    fn take(self) -> Self::Inner;

    /// Applies a closure to a reference of the inner value.
    fn apply_ref<F, X>(&self, f: F) -> X
    where
        F: FnOnce(&Self::Inner) -> X,
    {
        f(self.get_ref())
    }

    /// Applies a closure to a mutable reference of the inner value.
    ///
    /// *Note*: Unlike the other `apply` methods, this method accepts
    /// a closure and has no return value.
    fn apply_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Self::Inner),
    {
        f(self.get_mut());
    }
}

pub trait Identity {
    type Id: PartialEq;
    fn id(self) -> Self::Id;
}

pub trait Intern {
    type Key;
    type Value: ?Sized;

    /// Stores the given value if it is not currently already stored. Returns the key used by the `Self` to retrieve the stored value.
    fn intern(&mut self, value: &Self::Value) -> Self::Key;
}
