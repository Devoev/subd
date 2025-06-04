use crate::knots::error::FromVecError;
use nalgebra::RealField;

/// A vector of unique and increasing *breakpoints* of type [`T`]
#[derive(Clone, Debug)]
pub struct Breaks<T>(pub Vec<T>);

impl<T: RealField + Copy> Breaks<T> {
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
    pub fn new(mut breaks: Vec<T>) -> Result<Breaks<T>, FromVecError> {
        if !breaks.is_sorted() { return Err(FromVecError::UnsortedBreaks) };
        let num_breaks = breaks.len();
        breaks.dedup();
        if num_breaks != breaks.len() { return Err(FromVecError::DuplicateBreaks) };
        Ok(Breaks(breaks))
    }
}