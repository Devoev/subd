use crate::knots::knot_span::{KnotSpan, MultiKnotSpan};
use crate::knots::knot_vec::{KnotVec, ParametricBezierInterval};
use crate::mesh::Mesh;
use itertools::Itertools;
use nalgebra::{Point, RealField};
use std::fmt::{Display, Formatter};
use std::iter::{zip, Sum};

/// A [`D`]-variate multivariate knot vector.
#[derive(Debug, Clone)]
pub struct MultiKnotVec<T: RealField, const D: usize>(pub(crate) [KnotVec<T>; D]);

impl<T: RealField + Copy, const D: usize> MultiKnotVec<T, D> {

    /// Constructs a new [`MultiKnotVec`] from the given knot vectors.
    pub fn new(knot_vecs: [KnotVec<T>; D]) -> Self {
        MultiKnotVec(knot_vecs)
    }
    
    /// Constructs a open [`MultiKnotVec`] of size `n[i] + p[i] + 1` for each direction.
    pub fn open_uniform(n: [usize; D], p: [usize; D]) -> Self {
        let arr: [KnotVec<T>; D] = zip(n, p)
            .map(|(n, p)| KnotVec::open_uniform(n, p))
            .collect_vec()
            .try_into()
            .unwrap();
        arr.into()
    }
}

impl<T: RealField + Copy, const D : usize> MultiKnotVec<T, D> {
    
    /// Return the number of basis functions per parametric direction.
    pub fn n(&self) -> [usize; D] {
        self.0.iter().map(|knot_vec| knot_vec.n).collect_array().unwrap()
    }
    
    /// Return the degrees of basis functions per parametric direction.
    pub fn p(&self) -> [usize; D] {
        self.0.iter().map(|knot_vec| knot_vec.p).collect_array().unwrap()
    }

    pub fn breaks(&self) -> impl Iterator<Item=Point<T, D>> + '_ {
        self.0.iter()
            .map(|knots| knots.breaks().copied())
            .multi_cartesian_product()
            .map(|vec| Point::from_slice(&vec))
    }

    /// Finds the [`KnotSpan`] containing the given parametric value `t`.
    pub fn find_span(&self, t: [T; D]) -> Result<MultiKnotSpan<T, D>, ()> {
        MultiKnotSpan::find(self, t)
    }
}

impl<T: RealField + Copy, const D : usize> IntoIterator for MultiKnotVec<T, D> {
    type Item = [T; D];
    type IntoIter = impl Iterator<Item=Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
            .map(|knots| knots.into_iter())
            .multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }
}

impl<'a, T: RealField + Copy, const D : usize> IntoIterator for &'a MultiKnotVec<T, D> {
    type Item = [&'a T; D];
    type IntoIter = impl Iterator<Item=Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
            .map(|knots| knots.into_iter())
            .multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }
}

impl<T: RealField + Copy, const D: usize> From<[KnotVec<T>; D]> for MultiKnotVec<T, D> {
    fn from(value: [KnotVec<T>; D]) -> Self {
        MultiKnotVec::new(value)
    }
}

impl<T: RealField, const D: usize> Display for MultiKnotVec<T, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().skip(1)
            .fold(self.0[0].to_string(), |acc, knots| format!("{acc}×{knots}"))
        )
    }
}

// todo: update definition

impl<T: RealField + Copy, const D: usize> Mesh for &MultiKnotVec<T, D> {

    type NodeIter = impl Iterator<Item=Point<T, D>>;
    type ElemIter = impl Iterator<Item=ParametricBezierElement<T, D>>;

    fn num_nodes(self) -> usize {
        self.nodes().count()
    }

    fn nodes(self) -> Self::NodeIter {
        self.breaks()
    }

    fn num_elems(self) -> usize {
        self.elems().count()
    }

    fn elems(self) -> Self::ElemIter {
        self.0.iter()
            .map(|knots| knots.elems())
            .multi_cartesian_product()
            .map(|vec| ParametricBezierElement::new(vec.try_into().unwrap()))
    }
}

/// A Bezier element in `D`-dimensional parametric domain.
#[derive(Debug)]
pub struct ParametricBezierElement<T : RealField, const D : usize>(pub(crate) [ParametricBezierInterval<T>; D]);

impl<T: RealField + Copy, const D : usize> ParametricBezierElement<T, D> {

    /// Constructs a new [`ParametricBezierElement`] from the given knot vectors.
    pub fn new(factors: [ParametricBezierInterval<T>; D]) -> Self {
        ParametricBezierElement(factors)
    }
}

impl<T: RealField + Copy + Sum, const D : usize> ParametricBezierElement<T, D> {
    /// Returns the element size, i.e. `diam(Q)`.
    pub fn elem_size(&self) -> T {
        self.0.iter()
            .map(|I| I.elem_size().powi(2))
            .sum::<T>()
            .sqrt()
    }
}

impl<T: RealField, const D : usize> Display for ParametricBezierElement<T, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().skip(1)
            .fold(self.0[0].to_string(), |acc, knots| format!("{acc}×{knots}"))
        )
    }
}