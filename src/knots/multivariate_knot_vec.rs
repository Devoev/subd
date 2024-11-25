use std::fmt::{Display, Formatter};
use std::iter::zip;
use std::ops::Index;
use itertools::Itertools;
use nalgebra::RealField;
use crate::knots::knot_vec::{Breaks, KnotVec};
use crate::mesh::Mesh;

/// A `D`-dimensional multivariate knot vector.
pub struct MultivariateKnotVec<T : RealField, const D : usize>(pub(crate) [KnotVec<T>; D]);

impl<T: RealField + Copy, const D : usize> MultivariateKnotVec<T, D> {

    /// Constructs a new [`MultivariateKnotVec`] from the given knot vectors.
    pub fn new(factors: [KnotVec<T>; D]) -> Self {
        MultivariateKnotVec(factors)
    }

    /// Constructs a uniform [`MultivariateKnotVec`] of size `nums[i]` for each direction.
    pub fn uniform(nums: [usize; D]) -> Self {
        nums.map(|num| KnotVec::uniform(num)).into()
    }    
    
    /// Constructs a open [`MultivariateKnotVec`] of size `n[i] + p[i] + 1` for each direction.
    pub fn open(n: [usize; D], p: [usize; D]) -> Self {
        let arr: [KnotVec<T>; D] = zip(n, p)
            .map(|(n, p)| KnotVec::open(n, p))
            .collect_vec()
            .try_into()
            .unwrap();
        arr.into()
    }
}

impl<T: RealField + Copy, const D : usize> IntoIterator for MultivariateKnotVec<T, D> {
    type Item = [T; D];
    type IntoIter = impl Iterator<Item=Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
            .map(|knots| knots.into_iter())
            .multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }
}

impl<'a, T: RealField + Copy, const D : usize> IntoIterator for &'a MultivariateKnotVec<T, D> {
    type Item = [&'a T; D];
    type IntoIter = impl Iterator<Item=Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
            .map(|knots| knots.into_iter())
            .multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }
}

impl<T: RealField + Copy, const D: usize> From<[KnotVec<T>; D]> for MultivariateKnotVec<T, D> {
    fn from(value: [KnotVec<T>; D]) -> Self {
        MultivariateKnotVec::new(value)
    }
}

impl<T: RealField, const D: usize> Display for MultivariateKnotVec<T, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().skip(1)
            .fold(self.0[0].to_string(), |acc, knots| format!("{acc}×{knots}"))
        )
    }
}