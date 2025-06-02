use crate::cells::chart::Chart;
use crate::cells::geo;
use crate::cells::topo::{Cell, CellBoundary};
use crate::cells::vertex::VertexTopo;
use crate::mesh::cartesian::CartMesh;
use itertools::{repeat_n, Itertools};
use nalgebra::{Const, DimNameSub, Point, Point1, RealField, SVector, U1, U3};
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
}

/// **L**inear int**erp**olation (Lerp) in [`D`] dimensions.
/// Transforms the unit hypercube `[0,1]^D` to a [`HyperRectangle`] by the component-wise mapping
/// ```text
/// x[i] ↦ (1 - x[i]) a[i] + x[i] b[i]
/// ```
/// where `a` and `b` are the start and end coordinates of the rectangle respectively.
pub struct Lerp<T, const D: usize> {
    /// Start coordinates.
    pub a: SVector<T, D>,

    /// End coordinates.
    pub b: SVector<T, D>,
}

impl<T, const D: usize> Lerp<T, D> {
    /// Constructs a new [`Lerp`] from the given coordinate vectors `a` and `b`.
    pub fn new(a: SVector<T, D>, b: SVector<T, D>) -> Self {
        Lerp { a, b }
    }
}

impl <T: RealField + Copy, const D: usize> Chart<T, [T; D], D> for Lerp<T, D> {
    fn eval(&self, x: [T; D]) -> Point<T, D> {
        self.eval(SVector::from(x))
    }
}

impl <T: RealField + Copy, const D: usize> Chart<T, SVector<T, D>, D> for Lerp<T, D> {
    fn eval(&self, x: SVector<T, D>) -> Point<T, D> {
        let ones = SVector::repeat(T::one());
        let p = (ones - x).component_mul(&self.a) + x.component_mul(&self.b);
        Point::from(p)
    }
}

impl <T: RealField + Copy> Chart<T, T, 1> for Lerp<T, 1> {
    fn eval(&self, x: T) -> Point<T, 1> {
        Point1::new((T::one() - x) * self.a.x + x * self.b.x)
    }
}

impl <T: RealField + Copy, const D: usize> geo::Cell<T, [T; D], Const<D>, D> for HyperRectangle<T, D> {
    type GeoMap = Lerp<T, D>;

    fn geo_map(&self) -> Self::GeoMap {
        Lerp::new(self.a, self.b)
    }
}

impl <T: RealField + Copy> geo::Cell<T, T, U1, 1> for HyperRectangle<T, 1> {
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