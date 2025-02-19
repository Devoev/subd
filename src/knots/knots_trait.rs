use std::ops::Index;
use nalgebra::RealField;
use crate::knots::knot_vec::BreaksWithMultiplicity;

/// A [D]-variate knot vector of increasing knot values of type [T].
pub trait Knots<T: RealField, const D: usize, Idx> : Index<Idx> {

    /// An iterator that yields the breakpoints of a [KnotVec].
    type Breaks: Iterator;

    /// An iterator that yields the breakpoints with multiplicity of a [KnotVec].
    type BreaksWithMultiplicity: Iterator;

    /// Returns an iterator over the breaks, i.e. unique knot values.
    fn breaks() -> Self::Breaks;

    /// Returns an iterator over `(multiplicity, break)` pairs.
    fn breaks_with_multiplicity(&self) -> BreaksWithMultiplicity<T>;
}