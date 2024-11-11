use std::fmt::{Display, Formatter, Result};
use std::ops::Index;
use iter_num_tools::lin_space;
use itertools::{chain, Itertools};
use nalgebra::{DimAdd, RealField};

/// A knot vector of increasing knot values.
///
pub struct KnotVec<T : RealField>(Vec<T>);

impl<T : RealField + Copy> KnotVec<T> {

    pub fn new(knots: Vec<T>) -> Option<Self> {
        if knots.is_sorted() { Some(KnotVec(knots)) }
        else { None }
    }

    pub fn from_sorted(knots: Vec<T>) -> Self {
        KnotVec(knots)
    }

    pub fn open(n: usize, p: usize) -> Self {
        chain!(
            std::iter::repeat_n(T::zero(), p),
            lin_space(T::zero()..=T::one(), n-p+1),
            std::iter::repeat_n(T::one(), p)
        ).collect()
    }
}

impl<T : RealField> KnotVec<T> {
    
    pub fn breaks(&self) -> Vec<&T> {
        self.0.iter().dedup().collect()
    }
    
    pub fn breaks_with_multiplicity(&self) -> (Vec<&T>, Vec<usize>) {
        let (m, z): (Vec<usize>, Vec<&T>) = self.0.iter().dedup_with_count().unzip();
        (z, m)
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

impl<T : RealField> Display for KnotVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?}", self.0)
    }
}