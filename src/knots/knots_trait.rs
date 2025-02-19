use std::ops::Index;
use nalgebra::RealField;

/// A [D]-variate knot vector of increasing knot values of type [T].
pub trait Knots<'a, T: RealField, const D: usize, Idx> : Index<Idx> {

    /// An iterator that yields the knot breakpoints.
    type Breaks: Iterator;

    /// An iterator that yields the breakpoints with multiplicity.
    type BreaksWithMultiplicity: Iterator;

    /// Returns an iterator over the breaks, i.e. unique knot values.
    fn breaks(&'a self) -> Self::Breaks;

    /// Returns an iterator over `(multiplicity, break)` pairs.
    fn breaks_with_multiplicity(&'a self) -> Self::BreaksWithMultiplicity;
}

// move breakpoint stuff to Breakpoints trait