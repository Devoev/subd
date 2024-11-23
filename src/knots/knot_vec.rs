use iter_num_tools::lin_space;
use itertools::{chain, Dedup, DedupWithCount, Itertools};
use nalgebra::RealField;
use std::fmt::{Display, Formatter, Result};
use std::iter::zip;
use std::ops::Index;
use std::slice::Iter;
use std::vec;
use crate::mesh::Mesh;

/// A knot vector of increasing knot values.
#[derive(Debug, Clone)]
pub struct KnotVec<T : RealField>(pub(crate) Vec<T>);

impl<T : RealField + Copy> KnotVec<T> {

    /// Constructs a new `KnotVec`.
    /// If the `knots` are not sorted, `None` is returned.
    pub fn new(knots: Vec<T>) -> Option<Self> {
        if knots.is_sorted() { Some(KnotVec(knots)) }
        else { None }
    }
    
    /// Constructs a new `KnotVec` assuming that `knots` is sorted.
    pub fn from_sorted(knots: Vec<T>) -> Self {
        KnotVec(knots)
    }
    
    /// Constructs a uniform `KnotVec` of size `num`.
    pub fn uniform(num: usize) -> Self {
        lin_space(T::zero()..=T::one(), num).collect()
    }

    /// Constructs an open uniform `KnotVec` of size `n+p+1`.
    /// * `n` Number of basis functions.
    /// * `p` Degree of the basis functions.
    pub fn open(n: usize, p: usize) -> Self {
        chain!(
            std::iter::repeat_n(T::zero(), p),
            lin_space(T::zero()..=T::one(), n-p+1),
            std::iter::repeat_n(T::one(), p)
        ).collect()
    }
}

/// An iterator that yields the breakpoints of a [`KnotVec`].
type Breaks<'a, T> = Dedup<Iter<'a, T>>;

/// An iterator that yields the breakpoints with multiplicity of a [`KnotVec`].
type BreaksWithMultiplicity<'a, T> = DedupWithCount<Iter<'a, T>>;

impl<T : RealField + Copy> KnotVec<T> {

    /// Returns the number of elements in the knot vector, i.e. `n+p+1`.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the first knot.
    pub fn first(&self) -> T {
        self.0[0]
    }

    /// Returns the last knot.
    pub fn last(&self) -> T {
        self.0[self.len() - 1]
    }

    /// Returns an iterator over the breaks, i.e. unique knot values.
    pub fn breaks(&self) -> Breaks<T> {
        self.0.iter().dedup()
    }

    /// Returns an iterator over (multiplicity, break) pairs.
    pub fn breaks_with_multiplicity(&self) -> BreaksWithMultiplicity<T> {
        self.0.iter().dedup_with_count()
    }
}

impl<'a, T: RealField + Copy> Mesh for &'a KnotVec<T> {
    type NodeIter = Breaks<'a, T>;
    type ElemIter = impl Iterator<Item=(usize, usize)>;

    fn num_nodes(self) -> usize {
        self.breaks().count()
    }

    fn nodes(self) -> Self::NodeIter {
        self.breaks()
    }

    fn num_elems(self) -> usize {
        self.num_nodes() - 1
    }

    fn elems(self) -> Self::ElemIter {
        zip(0..self.num_nodes(), 1..=self.num_nodes()) // todo: update return value and type
    }
}

impl <T : RealField> Index<usize> for KnotVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T : RealField + Copy> FromIterator<T> for KnotVec<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        Self::from_sorted(Vec::from_iter(iter))
    }
}

impl <T : RealField + Copy> IntoIterator for KnotVec<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T : RealField> Display for KnotVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?}", self.0)
    }
}