use core::fmt;
use iter_num_tools::lin_space;
use itertools::{Dedup, DedupWithCount, Itertools};
use nalgebra::RealField;
use std::fmt::{Display, Formatter};
use std::ops::{Index, RangeInclusive};
use std::slice::Iter;
use std::vec;
use crate::knots::error::UnsortedKnotsError;

/// A vector of increasing *knot values* of type [`T`].
#[derive(Debug, Clone)]
pub struct KnotVec<T>(pub Vec<T>);

impl<T: RealField + Copy> KnotVec<T> {
    /// Constructs a new [`KnotVec<T>`] from the given `knots`.
    ///
    /// # Errors
    /// Will return an error if the knot values are not sorted.
    ///
    /// # Examples
    /// ```
    /// # use subd::knots::error::UnsortedKnotsError;
    /// # use subd::knots::knot_vec::KnotVec;
    ///
    /// let knots_sorted = vec![0.0, 0.0, 0.5, 1.0, 1.0];
    /// assert!(KnotVec::new(knots_sorted).is_ok());
    ///
    /// let knots_unsorted = vec![0.0, 1.0, 0.5];
    /// assert_eq!(KnotVec::new(knots_unsorted), Err(UnsortedKnotsError));
    /// ```
    pub fn new(knots: Vec<T>) -> Result<KnotVec<T>, UnsortedKnotsError> {
        match knots.is_sorted() {
            true => Ok(KnotVec(knots)),
            false => Err(UnsortedKnotsError),
        }
    }

    /// Constructs a new [`KnotVec<T>`] assuming that the given `knots` are sorted.
    pub fn from_sorted(knots: Vec<T>) -> Self {
        KnotVec(knots)
    }

    /// Constructs a new [`KnotVec<T>`] by sorting the given `knots`.
    ///
    /// # Examples
    /// ```
    /// # use subd::knots::knot_vec::KnotVec;
    ///
    /// let knots_unsorted = KnotVec::from_unsorted(vec![0.0, 1.0, 0.5, 1.0, 0.0]);
    /// assert_eq!(knots_unsorted.0, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
    /// ```
    pub fn from_unsorted(mut knots: Vec<T>) -> Self {
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        KnotVec(knots)
    }

    /// Constructs a uniform [`KnotVec<T>`] from `0` to `1` with `num` knots.
    ///
    /// # Examples
    /// ```
    /// use subd::knots::knot_vec::KnotVec;
    /// 
    /// let knots = KnotVec::<f64>::uniform(5);
    /// assert_eq!(knots.0, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// ```
    pub fn uniform(num: usize) -> Self {
        KnotVec(lin_space(T::zero()..=T::one(), num).collect_vec())
    }

    /// Returns the first knot.
    pub fn first(&self) -> T {
        self.0[0]
    }

    /// Returns the last knot.
    pub fn last(&self) -> T {
        self.0[self.len() - 1]
    }

    /// Returns the range of this knot vector.
    pub fn range(&self) -> RangeInclusive<T> {
        self.first()..=self.last()
    }
}

/// An iterator that yields the breakpoints of a [`KnotVec`].
pub type BreaksIter<'a, T> = Dedup<Iter<'a, T>>;

/// An iterator that yields the breakpoints with multiplicity of a [`KnotVec`].
pub type BreaksWithMultiplicityIter<'a, T> = DedupWithCount<Iter<'a, T>>;

impl<T : RealField + Copy> KnotVec<T> {
    /// Returns the number of knots.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the knot vector is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns an iterator over the breaks, i.e. unique knot values.
    pub fn breaks_iter(&self) -> BreaksIter<T> {
        self.0.iter().dedup()
    }

    /// Returns an iterator over (multiplicity, break) pairs.
    pub fn breaks_with_multiplicity_iter(&self) -> BreaksWithMultiplicityIter<T> {
        self.0.iter().dedup_with_count()
    }
}

impl <T : RealField> Index<usize> for KnotVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl <T : RealField + Copy> IntoIterator for KnotVec<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl <'a, T : RealField + Copy> IntoIterator for &'a KnotVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T : RealField> Display for KnotVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}