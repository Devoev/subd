use core::fmt;
use iter_num_tools::lin_space;
use itertools::{Dedup, DedupWithCount, Itertools};
use nalgebra::RealField;
use std::fmt::{Display, Formatter};
use std::ops::{Index, RangeInclusive};
use std::slice::Iter;
use std::vec;

/// A vector of increasing knot values of type [`T`], backed by a [`Vec<T>`].
#[derive(Debug, Clone)]
pub struct KnotVec<T>(pub(crate) Vec<T>);

impl<T: RealField + Copy> KnotVec<T> {

    /// Constructs a new [`KnotVec<T>`].
    /// If the `knots` are not sorted, [`None`] is returned.
    pub fn new(knots: Vec<T>) -> Option<KnotVec<T>> {
        knots.is_sorted().then_some(KnotVec(knots))
    }

    /// Constructs a uniform [`KnotVec<T>`] with `num` knots.
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
pub type Breaks<'a, T> = Dedup<Iter<'a, T>>;

/// An iterator that yields the breakpoints with multiplicity of a [`KnotVec`].
pub type BreaksWithMultiplicity<'a, T> = DedupWithCount<Iter<'a, T>>;

impl<T : RealField + Copy> KnotVec<T> {

    /// Returns the number of knots.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns an iterator over the breaks, i.e. unique knot values.
    pub fn breaks(&self) -> Breaks<T> {
        self.0.iter().dedup()
    }

    /// Returns an iterator over (multiplicity, break) pairs.
    pub fn breaks_with_multiplicity(&self) -> BreaksWithMultiplicity<T> {
        self.0.iter().dedup_with_count()
    }

    // /// Returns the global mesh size, i.e. `h = max{ h_Q }`.
    // pub fn mesh_size(&self) -> T {
    //     self.elems()
    //         .map(|q| q.elem_size())
    //         .max_by(|a, b| a.partial_cmp(b).unwrap())
    //         .unwrap()
    // }
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