use iter_num_tools::lin_space;
use itertools::{chain, Itertools};
use nalgebra::RealField;
use std::fmt::{Display, Formatter, Result};
use std::ops::Index;

/// A knot vector of increasing knot values.
///
pub struct KnotVec<T : RealField>(Vec<T>);

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
    
    /// Constructs a uniform `KnotVec` of size `n+p+1`.
    /// * `n` Number of basis functions.
    /// * `p` Degree of the basis functions.
    pub fn uniform(n: usize, p: usize) -> Self {
        lin_space(T::zero()..=T::one(), n+p+1).collect()
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

impl<T : RealField> KnotVec<T> {

    /// Returns the number of elements in the knot vector, i.e. `n+p+1`.
    pub fn len(&self) -> usize {
        self.0.len()
    }
    
    pub fn breaks(&self) -> Vec<&T> {
        self.0.iter().dedup().collect()
    }
    
    pub fn breaks_with_multiplicity(&self) -> (Vec<&T>, Vec<usize>) {
        let (m, z): (Vec<usize>, Vec<&T>) = self.0.iter().dedup_with_count().unzip();
        (z, m)
    }

    pub fn find_span(&self, t: T) -> usize {
        let idx = self.0.binary_search_by(|xi| xi.partial_cmp(&t).unwrap());
        match idx {
            Ok(i) => { if i == self.len() - 1 { i - 1 } else { i } } // todo: return idx=n not i-1
            Err(i) => { i - 1 }
        }
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