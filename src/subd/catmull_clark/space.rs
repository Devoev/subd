use crate::space::Space;
use crate::subd::catmull_clark::basis::CatmarkBasis;

/// Function space for Catmull-Clark basis functions.
pub type CatmarkSpace<'a, T, const M: usize> = Space<T, CatmarkBasis<'a, T, M>>;