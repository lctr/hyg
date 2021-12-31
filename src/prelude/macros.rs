/// Enums corresponding to a literal, but that *don't* hold any data values,
/// instead dynamically converting between literal and enum. Literal values are
/// automatically recorded in doc comments for each enum variant.
#[macro_export]
macro_rules! strenum {
    (L) => {Assoc::Left};
    (R) => {Assoc::Right};
    (meta $m:meta) => {$(#[$m])*};
    (
        $(#[$meta:meta])*
        $opk:ident :: $(
            $(#[$metas:meta])*
            $name:ident $lit:literal
        )+
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $opk {
            $(
                $(#[$metas])*
                #[doc = $lit]
                $name
            ),+
        }

        impl std::fmt::Display for $opk {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{:?}", &self)
            }
        }
    };
    (ident $t:tt) => {$t};
    ($opk:ident $is_kind:tt :: $($name:ident $lit:literal)+) => {
        strenum! { $opk :: $($name $lit)+ }

        #[allow(unused)]
        impl $opk {
            pub fn $is_kind(word: &str) -> bool {
                match word {
                    $($lit => {true})+
                    _ => false
                }
            }
            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $($lit => {Some($opk::$name)})+
                    _ => None
                }
            }
            pub fn as_str(&self) -> &str {
                match self {
                    $($opk::$name => {$lit})+
                }
            }
        }
    };
    (each $id:ident $is_kind:ident ::
        $(
            $(
                $name:ident $lit:literal
            )+
            @ $prec:literal $assoc:ident
        )+
    ) => {
        strenum! {$id $is_kind :: $($($name $lit)+)+ }

        #[allow(unused)]
        impl $id {
            pub fn get_prec(&self) -> usize {
                match self {
                    $($($id::$name => {$prec})+)+
                }
            }

            pub fn get_assoc(&self) -> Assoc {
                match self {
                    $($($id::$name => {strenum!($assoc)})+)+
                }
            }

            pub fn is_left_assoc(&self) -> bool {
                matches!(self.get_assoc(), strenum!(L))
            }

            pub fn is_right_assoc(&self) -> bool {
                matches!(self.get_assoc(), strenum!(R))
            }
        }
    };
}
