use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use nalgebra::RealField;
use std::fmt::{Debug};
use std::ops::Index;
use std::slice::Iter;
use std::vec;

/// A vector of unique and increasing *breakpoints* of type [`T`],
/// paired with a *multiplicity* value.
#[derive(Clone, Debug)]
pub struct BreaksWithMultiplicity<T>(pub Vec<(usize, T)>);

impl <T: RealField> BreaksWithMultiplicity<T> {
    /// Constructs new [`BreaksWithMultiplicity<T>`] from the given `knots`
    /// by removing duplicate knot values and counting the multiplicity of each knot.
    ///
    /// # Examples
    /// ```
    /// # use subd::knots::breaks_with_multiplicity::BreaksWithMultiplicity;
    /// # use subd::knots::knot_vec::KnotVec;
    ///
    /// let knots = KnotVec::new(
    ///     vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    /// ).unwrap();
    /// let breaks = BreaksWithMultiplicity::from_knots(knots);
    ///
    /// assert_eq!(
    ///     breaks.0,
    ///     vec![(3, 0.0), (1, 0.25), (1, 0.5), (1, 0.75), (3, 1.0)]
    /// );
    /// ```
    pub fn from_knots(knots: KnotVec<T>) -> Self {
        let breaks = knots.breaks_with_multiplicity_iter()
            .map(|(k, zeta)| (k, zeta.clone()))
            .collect();
        BreaksWithMultiplicity(breaks)
    }
}

impl<T> BreaksWithMultiplicity<T> {
    /// Returns the number of breakpoints.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the breakpoint vector is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl <T> Index<usize> for BreaksWithMultiplicity<T> {
    type Output = (usize, T);

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl <T> IntoIterator for BreaksWithMultiplicity<T> {
    type Item = (usize, T);
    type IntoIter = vec::IntoIter<(usize, T)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl <'a, T> IntoIterator for &'a BreaksWithMultiplicity<T> {
    type Item = &'a (usize, T);
    type IntoIter = Iter<'a, (usize, T)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}