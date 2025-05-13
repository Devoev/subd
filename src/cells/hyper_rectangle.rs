use std::iter::zip;
use std::ops::RangeInclusive;
use itertools::{repeat_n, Itertools};
use crate::cells::cell::{Cell, CellBoundary};
use crate::cells::vertex::VertexTopo;
use nalgebra::{Const, DimNameSub, Point, RealField, SVector, U3};

/// A [`K`]-dimensional hyperrectangle.
/// For vectors, `a` and `b` of length `K` it is defined as the set of all points
/// ```text
/// { x : a[i] <= x[i] <= b[] }
/// ```
/// which is equal to the cartesian product
/// ```text
/// [a[1], b[1]] × ... × [a[K], b[K]]
/// ```
/// of intervals `[a[i], b[i]]`.
#[derive(Debug, Clone, Copy)]
pub struct HyperRectangle<T: RealField, const K: usize> {
    /// Start coordinates.
    pub a: SVector<T, K>,

    /// End coordinates.
    pub b: SVector<T, K>,
}

// todo: update points and ranges methods
impl<T: RealField + Copy, const K: usize> HyperRectangle<T, K> {
    /// Returns an iterator over the corner points of this hyperrectangle.
    pub fn points(&self) -> impl Iterator<Item=Point<T, K>> + '_ {
        repeat_n([&self.a, &self.b], K)
            .multi_cartesian_product()
            .map(|offset| {
                let coords = offset.iter().enumerate().map(|(i, coord)| {
                    coord[i]
                }).collect_array::<K>().unwrap();
                Point::from(coords)
            })
    }

    /// Returns the interval ranges `a[i]..=b[i]` for each parametric direction.
    pub fn ranges(&self) -> [RangeInclusive<T>; K] {
        zip(&self.a, &self.b).map(|(&a, &b)| a..=b).collect_array::<K>().unwrap()
    }
}

// todo: index type is incorrect
//  - nodes and vols should have one multi index
//  - edges and faces should have 3 ?

#[derive(Debug, Clone, Copy)]
pub struct HyperRectangleTopo<const K: usize>(pub [usize; K]);

impl <const K: usize> Cell<Const<K>> for HyperRectangleTopo<K> {
    fn nodes(&self) -> &[VertexTopo] {
        todo!()
    }

    fn is_connected<const M: usize>(&self, other: &Self) -> bool
    where
        Const<K>: DimNameSub<Const<M>>
    {
        todo!()
    }
}

impl CellBoundary<U3> for HyperRectangleTopo<3> {
    const NUM_SUB_CELLS: usize = 6;
    type SubCell = HyperRectangleTopo<2>;
    type Boundary = ();

    fn boundary(&self) -> Self::Boundary {
        todo!("Get")
    }
}