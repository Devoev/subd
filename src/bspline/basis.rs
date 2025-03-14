use crate::knots::knot_span::KnotSpan;
use nalgebra::{DVector, RealField};

/// A basis for a spline function space.
/// 
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`Knt`] : Knot values and parametric values.
/// - [`Idx`] : Index type of the knot vector.
pub trait Basis<T: RealField, Knt: Copy, Idx> where Self: Sized {
    /// An iterator over linear indices.
    type LinIndices: Iterator<Item=usize>;
    
    /// Returns the total number of basis functions.
    fn num(&self) -> usize;
    
    /// Finds the [`KnotSpan`] containing the given parametric value `t`.
    fn find_span(&self, t: Knt) -> Result<KnotSpan<Idx>, ()>;
    
    /// Finds the indices of nonzero basis functions in the given `span`.
    fn nonzero(&self, span: &KnotSpan<Idx>) -> Self::LinIndices;
    
    /// Evaluates the nonzero basis functions in the `span` at the parametric point `t`.
    fn eval(&self, t: Knt, span: &KnotSpan<Idx>) -> DVector<T>;
}

/// Conversion into breakpoints.
pub trait IntoBreaks<'a> {
    /// An iterator that yields the knot breakpoints.
    type Breaks: Iterator;

    /// An iterator that yields the breakpoints with multiplicity.
    type BreaksWithMultiplicity: Iterator;

    /// Returns an iterator over the breaks, i.e. unique knot values.
    fn breaks(&'a self) -> Self::Breaks;

    /// Returns an iterator over `(multiplicity, break)` pairs.
    fn breaks_with_multiplicity(&'a self) -> Self::BreaksWithMultiplicity;
}