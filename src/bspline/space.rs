use crate::basis::space::Space;
use crate::bspline::global_basis::MultiBsplineBasis;

// todo: add more type aliases and proper documentation (type parameters, special cases...)

/// Function space of [`D`]-variate scalar B-Spline basis functions.
pub type BsplineSpace<T, X, const D: usize> = Space<T, X, MultiBsplineBasis<T, D>, D>;