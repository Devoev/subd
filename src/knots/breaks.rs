use crate::knots::error::FromVecError;
use nalgebra::RealField;
use crate::knots::knot_vec::KnotVec;

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
    pub fn from_knots(knots: KnotVec<T>) -> Breaks<T> {
        let mut breaks = knots.0;
        breaks.dedup();
        Breaks(breaks)
    }
}