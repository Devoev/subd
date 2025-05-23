use std::error::Error;
use std::fmt::{Display, Formatter};
use crate::knots::knot_vec::KnotVec;

/// An error returned when a parametric value is outside the range of a [`KnotVec`].
#[derive(Debug)]
pub struct OutsideKnotRangeError;

impl Display for OutsideKnotRangeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "parametric value is outside the span range")
    }
}

impl Error for OutsideKnotRangeError {}

/// An error returned during construction of a [`KnotVec`]
/// if the provided knot values are not sorted.
#[derive(Debug)]
pub struct UnsortedKnotsError;

impl Display for UnsortedKnotsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "provided knot values are not sorted")
    }
}

impl Error for UnsortedKnotsError {}