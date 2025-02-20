use crate::mesh::Mesh;
use iter_num_tools::lin_space;
use itertools::{chain, Dedup, DedupWithCount, Itertools};
use nalgebra::RealField;
use std::fmt::{Display, Formatter, Result};
use std::ops::Index;
use std::slice::Iter;
use std::vec;

/// A knot vector of length `n + p + 1`, backed by a [`Vec<T>`].
#[derive(Debug, Clone)]
pub struct KnotVec<T: RealField> {
    /// The vector of knot values.
    pub(crate) vec: Vec<T>,
    
    /// Number of basis functions.
    pub n: usize,
    
    /// Degree of basis functions.
    pub p: usize
}

impl<T : RealField + Copy> KnotVec<T> {

    /// Constructs a new [`KnotVec`].
    /// 
    /// If the `knots` are not sorted 
    /// or the values of `n` and `p` don't match the length of the knots, 
    /// [`None`] is returned.
    pub fn new(knots: Vec<T>, n: usize, p: usize) -> Option<Self> {
        (knots.is_sorted() && knots.len() == n + p + 1)
            .then_some(KnotVec { vec: knots, n, p })
    }
    
    /// Constructs a new `KnotVec` assuming that `knots` is sorted.
    pub fn from_sorted(knots: Vec<T>, n: usize, p: usize) -> Self {
        KnotVec { vec: knots, n, p }
    }
    
    /// Constructs a uniform [`KnotVec`].
    pub fn uniform(n: usize, p: usize) -> Self {
        let vec = lin_space(T::zero()..=T::one(), n + p + 1).collect_vec();
        KnotVec { vec, n, p }
    }
    
    /// Constructs an open [`KnotVec`] with given `internal` knot values.
    /// 
    /// If internal knots are not inside `(0,1)`, [`None`] is returned.
    pub fn open(internal: Vec<T>, n: usize, p: usize) -> Option<Self> {
        let vec = chain!(
            std::iter::repeat_n(T::zero(), p+1),
            internal,
            std::iter::repeat_n(T::one(), p+1)
        ).collect();
        
        Self::new(vec, n, p)
    }

    /// Constructs an open uniform [`KnotVec`].
    pub fn open_uniform(n: usize, p: usize) -> Self {
        let vec = chain!(
            std::iter::repeat_n(T::zero(), p),
            lin_space(T::zero()..=T::one(), n-p+1),
            std::iter::repeat_n(T::one(), p)
        ).collect();
        
        KnotVec { vec, n, p }
    }
}

/// An iterator that yields the breakpoints of a [`KnotVec`].
pub type Breaks<'a, T> = Dedup<Iter<'a, T>>;

/// An iterator that yields the breakpoints with multiplicity of a [`KnotVec`].
pub type BreaksWithMultiplicity<'a, T> = DedupWithCount<Iter<'a, T>>;

impl<T : RealField + Copy> KnotVec<T> {

    /// Returns the number of elements in the knot vector, i.e. `n+p+1`.
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    /// Returns the first knot.
    pub fn first(&self) -> T {
        self.vec[0]
    }

    /// Returns the last knot.
    pub fn last(&self) -> T {
        self.vec[self.len() - 1]
    }

    /// Returns an iterator over the breaks, i.e. unique knot values.
    pub fn breaks(&self) -> Breaks<T> {
        self.vec.iter().dedup()
    }

    /// Returns an iterator over (multiplicity, break) pairs.
    pub fn breaks_with_multiplicity(&self) -> BreaksWithMultiplicity<T> {
        self.vec.iter().dedup_with_count()
    }
    
    /// Returns the global mesh size, i.e. `h = max{ h_Q }`.
    pub fn mesh_size(&self) -> T {
        self.elems()
            .map(|q| q.elem_size())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

impl<'a, T: RealField + Copy> Mesh for &'a KnotVec<T> {
    type NodeIter = Breaks<'a, T>;
    type ElemIter = impl Iterator<Item=ParametricBezierInterval<T>> + Clone;

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
        self.nodes()
            .tuple_windows()
            .map(|(a, b)| ParametricBezierInterval::new(*a, *b))
    }
}

impl <T : RealField> Index<usize> for KnotVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vec[index]
    }
}

impl <T : RealField + Copy> IntoIterator for KnotVec<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.into_iter()
    }
}

impl <'a, T : RealField + Copy> IntoIterator for &'a KnotVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.vec.iter()
    }
}

impl<T : RealField> Display for KnotVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?}", self.vec)
    }
}

// todo: update definition
/// A 1D Bezier element in parametric domain, i.e. the open interval `(a,b)`.
#[derive(Debug, Clone)]
pub struct ParametricBezierInterval<T : RealField> {
    /// Lower bound.
    a: T,
    /// Upper bound.
    b: T
}

impl<T: RealField + Copy> ParametricBezierInterval<T> {

    /// Constructs a new [`ParametricBezierInterval`].
    pub fn new(a: T, b: T) -> Self {
        ParametricBezierInterval { a, b }
    }

    /// Returns the element size, i.e. `diam(Q)`.
    pub fn elem_size(&self) -> T {
        self.b - self.a
    }
}

impl<T: RealField> Display for ParametricBezierInterval<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "({}, {})", self.a, self.b)
    }
}