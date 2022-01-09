use crate::prelude::symbol::Symbol;

use super::{
    decl::{Clause, DataVariant, Decl, Type},
    expr::Expr,
    fixity::{Fixity, FixityTable},
    name::Name,
    pattern::{Constraint, Pat},
    Operator,
};

/// The portion of an `import` statement holding the *module*
/// (or *submodule*) to be imported.
///
/// When found alone, this directive imports everything from
/// a given module path chain, potentially renaming (and thus
/// qualifying) it if indicated.
///
/// * Import everything from the `Std` module `IO` submodule:
///
///     `import Std.IO`
///
/// * Import everything from the `Std` module `IO` submodule,
/// using the namespace `IO`:
///
///     `import Std.IO as IO`
///
/// **Note:** Renaming a module import qualifies it. The same does not
/// occur for renamed imported *items*.
///
/// This means that `import X.Y as Z` exposes `X.Y` through the namespace
/// `Z`, requiring the prefix `Z.` in order to access imported elements.
///
/// However, `import X (y as z)` would simply rename the import `y` to `z`,
/// and `z` would be accessible throughout the module without need for
/// qualification.
///
/// Lastly, both may be renamed in the same statement, applying both
/// aforementioned rules, e.g., `import X as Y (y as z)` would rename and
/// qualify the import `X` under the namespace `Y`, and the imported `y`
/// would be renamed to `z` such that this import is accessed throughout
/// the module with the identifier `Y.z`.
///
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImportPath {
    path: Vec<Symbol>,
    rename: Option<Symbol>,
}

/// The tail end of an import statement, containing the name of the item to be imported, with possibility of renaming the imported item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImportItem {
    name: Symbol,
    rename: Option<Symbol>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Import {
    path: ImportPath,
    items: Vec<ImportItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Module {
    /// Module names may be left implicit. If `None`, then the filename (if
    /// any) is used as the module name.
    name: Option<Symbol>,
    ast: Ast,
}

impl Module {
    pub fn new(ast: Ast, name: Option<Symbol>) -> Self {
        Self { name, ast }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InfixDecl {
    pub fixity: Fixity,
    pub infixes: Vec<Operator>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DataDecl {
    pub name: Name,
    pub constraints: Vec<Constraint<Name, Name>>,
    pub poly: Vec<Name>,
    pub variants: Vec<DataVariant>,
    pub derives: Vec<Name>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AliasDecl {
    pub name: Name,
    pub poly: Vec<Name>,
    pub rhs: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClassDecl {
    pub name: Name,
    pub constraints: Vec<Constraint<Name, Name>>,
    pub defs: Vec<Clause<Pat, Expr, Decl>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionDecl {
    pub name: Name,
    /// Each definition corresponds to an equation wherein
    pub defs: Vec<Clause<Pat, Expr, Decl>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AnnotationDecl {
    pub name: Name,
    pub tipo: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstanceDecl {
    pub who: Type,
    pub constraints: Vec<Constraint<Name, Name>>,
    pub defs: Vec<Clause<Pat, Expr, Decl>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Ast {
    pub imports: Vec<Import>,
    pub main: Option<Decl>,
    infixes: Vec<InfixDecl>,
    aliases: Vec<AliasDecl>,
    functions: Vec<FunctionDecl>,
    annotations: Vec<AnnotationDecl>,
    datatypes: Vec<DataDecl>,
    classes: Vec<ClassDecl>,
    instances: Vec<InstanceDecl>,
}

impl Ast {
    pub fn push_decl(&mut self, decl: Decl) {
        match decl {
            Decl::Infix { fixity, infixes } => self
                .infixes
                .push(InfixDecl { fixity, infixes }),
            Decl::Data {
                name,
                constraints,
                poly,
                variants,
                derives,
            } => self.datatypes.push(DataDecl {
                name,
                constraints,
                poly,
                variants,
                derives,
            }),
            Decl::Alias { name, poly, rhs } => self
                .aliases
                .push(AliasDecl { name, poly, rhs }),
            Decl::Class {
                name,
                constraints,
                defs,
            } => self.classes.push(ClassDecl {
                name,
                constraints,
                defs,
            }),
            Decl::Function { name, defs } => self
                .functions
                .push(FunctionDecl { name, defs }),
            Decl::Annotation { name, tipo } => self
                .annotations
                .push(AnnotationDecl { name, tipo }),
            Decl::Instance {
                who,
                constraints,
                defs,
            } => self.instances.push(InstanceDecl {
                who,
                constraints,
                defs,
            }),
        }
    }
}
