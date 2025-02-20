use std::ops::Index;

/// A knot vector of increasing knot values of type [T].
pub trait Knots<T, N, P, Idx> : Index<Idx, Output=T> {
    /// Returns the numbers of basis functions on this knot vector for each parametric direction.
    fn nums(&self) -> N;
    
    /// Returns the degrees of basis functions on this knot vector for each parametric direction.
    fn degrees(&self) -> P;
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