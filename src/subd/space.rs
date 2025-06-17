use crate::basis::space::Space;
use crate::subd::basis::CatmarkBasis;

/// Function space for Catmull-Clark basis functions.
pub type CatmarkSpace<'a, T, const M: usize> = Space<T, (T, T), CatmarkBasis<'a, T, M>, 2>;