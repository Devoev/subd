use crate::knots::error::FromVecError;
use crate::knots::knot_vec::KnotVec;
use core::fmt;
use nalgebra::RealField;
use std::fmt::{Display, Formatter};
use std::ops::Index;
use std::slice::Iter;
use std::vec;

/// A vector of unique and increasing *breakpoints* of type [`T`]
#[derive(Clone, Debug)]
pub struct Breaks<T>(pub Vec<T>);

impl<T: RealField> Breaks<T> {
    /// Constructs new [`Breaks<T>`] from the given `breaks`.
    ///
    /// # Errors
    /// Will return an error if the breakpoints are not sorted or not unique.
    ///
    /// # Examples
    /// ```
    /// # use subd::knots::breaks::Breaks;
    /// # use subd::knots::error::FromVecError;
    ///
    /// let breaks_sorted_and_unique = vec![0.0, 0.5, 1.0];
    /// assert!(Breaks::new(breaks_sorted_and_unique).is_ok());
    ///
    /// let breaks_unsorted = vec![0.0, 1.0, 0.5];
    /// assert_eq!(Breaks::new(breaks_unsorted), Err(FromVecError::UnsortedBreaks));
    ///
    /// let breaks_duplicate = vec![0.0, 0.0, 0.5, 1.0, 1.0];
    /// assert_eq!(Breaks::new(breaks_duplicate), Err(FromVecError::DuplicateBreaks));
    /// ```
    pub fn new(mut breaks: Vec<T>) -> Result<Self, FromVecError> {
        if !breaks.is_sorted() { return Err(FromVecError::UnsortedBreaks) };
        let num_breaks = breaks.len();
        breaks.dedup(); // todo: replace this with a function that just checks for the first duplicate
        if num_breaks != breaks.len() { return Err(FromVecError::DuplicateBreaks) };
        Ok(Breaks(breaks))
    }

    /// Constructs new [`Breaks<T>`] from the given `knots` by removing duplicate knot values.
    ///
    /// # Examples
    /// ```
    /// # use subd::knots::breaks::Breaks;
    /// # use subd::knots::knot_vec::KnotVec;
    ///
    /// let knots = KnotVec::new(vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]).unwrap();
    /// let breaks = Breaks::from_knots(knots);
    ///
    /// assert_eq!(breaks.0, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// ```
    pub fn from_knots(knots: KnotVec<T>) -> Self {
        let mut breaks = knots.0;
        breaks.dedup();
        Breaks(breaks)
    }
}

impl<T> Breaks<T> {
    /// Returns the number of breakpoints.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the breakpoint vector is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl <T : RealField> Index<usize> for Breaks<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl <T : RealField + Copy> IntoIterator for Breaks<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl <'a, T : RealField + Copy> IntoIterator for &'a Breaks<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T : RealField> Display for Breaks<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}