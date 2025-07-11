use crate::knots::breaks::Breaks;
use crate::knots::knot_vec::KnotVec;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// An error returned when a parametric value is outside the range of a [`KnotVec`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OutsideKnotRangeError;

impl Display for OutsideKnotRangeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "parametric value is outside the span range")
    }
}

impl Error for OutsideKnotRangeError {}

/// An error returned during construction of a [`KnotVec`]
/// if the provided knot values are not sorted.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct UnsortedKnotsError;

impl Display for UnsortedKnotsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "provided knot values are not sorted")
    }
}

impl Error for UnsortedKnotsError {}

/// An error returned when constructing [`Breaks`] from a `Vec`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum FromVecError {
    /// The breakpoints are not sorted.
    UnsortedBreaks,

    /// The breakpoints are not unique.
    DuplicateBreaks
}

impl Display for FromVecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FromVecError::UnsortedBreaks => write!(f, "provided breakpoints are not sorted"),
            FromVecError::DuplicateBreaks => write!(f, "provided breakpoints are not unique"),
        }
    }
}