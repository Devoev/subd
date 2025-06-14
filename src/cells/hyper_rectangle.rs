use crate::cells::geo;
use crate::cells::lerp::Lerp;
use crate::cells::topo::{Cell, CellBoundary};
use crate::cells::node::NodeIdx;
use crate::mesh::cartesian::CartMesh;
use itertools::{repeat_n, Itertools};
use nalgebra::{vector, Const, DimName, DimNameSub, Point, RealField, SVector, U1, U3};
use std::iter::zip;
use std::ops::RangeInclusive;

/// A [`K`]-dimensional hyperrectangle.
/// For coordinate vectors `a` and `b` of length `K` it is defined as the set of all points
/// ```text
/// { x : a[i] <= x[i] <= b[i] }
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
    /// Constructs a new [`HyperRectangle`] from the given coordinate vectors `a` and `b`.
    pub fn new(a: SVector<T, K>, b: SVector<T, K>) -> Self {
        HyperRectangle { a, b }
    }
    
    /// Constructs a new [`HyperRectangle`] from the given topology `topo`
    /// and mesh `msh`.
    pub fn from_topo(topo: HyperRectangleTopo<K>, msh: &CartMesh<T, K>) -> Self {
        let idx_a = topo.0;
        let idx_b = idx_a.map(|i| i + 1);
        let a = msh.vertex(idx_a);
        let b = msh.vertex(idx_b);
        HyperRectangle::new(a.coords, b.coords)
    }
    
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
    
    /// Returns the [Self::ranges] as 1D hyper rectangles.
    pub fn intervals(&self) -> [HyperRectangle<T, 1>; K] {
        zip(&self.a, &self.b).map(|(&a, &b)| {
            HyperRectangle::new(vector![a], vector![b])
        }).collect_array().unwrap()
    }
}

impl <T: RealField + Copy, const D: usize> geo::Cell<T, [T; D], D, D> for HyperRectangle<T, D> {
    type GeoMap = Lerp<T, D>;

    fn geo_map(&self) -> Self::GeoMap {
        Lerp::new(self.a, self.b)
    }
}

impl <T: RealField + Copy> geo::Cell<T, T, 1, 1> for HyperRectangle<T, 1> {
    type GeoMap = Lerp<T, 1>;

    fn geo_map(&self) -> Self::GeoMap {
        Lerp::new(self.a, self.b)
    }
}

// todo: index type is incorrect
//  - nodes and vols should have one multi index
//  - edges and faces should have 3 ?

#[derive(Debug, Clone, Copy)]
pub struct HyperRectangleTopo<const K: usize>(pub [usize; K]);

impl <const K: usize> Cell<Const<K>> for HyperRectangleTopo<K> {
    fn nodes(&self) -> &[NodeIdx] {
        todo!()
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        Const<K>: DimNameSub<M>
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