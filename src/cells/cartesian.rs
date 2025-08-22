use crate::cells::geo;
use crate::cells::lerp::MultiLerp;
use crate::cells::topo::{Cell, CellBoundary};
use crate::cells::node::NodeIdx;
use crate::mesh::cartesian::CartMesh;
use itertools::{repeat_n, Itertools};
use nalgebra::{vector, Const, DimName, DimNameSub, Point, Point1, RealField, SVector, U1, U3};
use std::iter::zip;
use std::ops::RangeInclusive;
use crate::cells::unit_cube::UnitCube;

/// A [`K`]-dimensional cartesian cell aka. hyper-rectangle.
/// For coordinate vectors `a` and `b` of length `K` it is defined as the set of all points
/// ```text
/// { x : a[i] <= x[i] <= b[i] }
/// ```
/// which is equal to the cartesian product
/// ```text
/// [a[1], b[1]] × ... × [a[K], b[K]]
/// ```
/// of intervals `[a[i], b[i]]`.
/// Geometrically it is a `K`-cell restricted by cartesian gridlines.
#[derive(Debug, Clone, Copy)]
pub struct CartCell<T: RealField, const K: usize> {
    /// Start coordinates.
    pub a: Point<T, K>,

    /// End coordinates.
    pub b: Point<T, K>,
}

impl<T: RealField> CartCell<T, 1> {
    /// Constructs a new univariate [`CartCell`],
    /// i.e. the interval `[a,b]`.
    pub fn new_univariate(a: T, b: T) -> Self {
        CartCell::new(Point1::new(a), Point1::new(b))
    }
}

impl<T: RealField, const K: usize> CartCell<T, K> {
    /// Constructs a new [`CartCell`] from the given coordinates `a` and `b`.
    pub fn new(a: Point<T, K>, b: Point<T, K>) -> Self {
        CartCell { a, b }
    }
}

// todo: update points and ranges methods
impl<T: RealField + Copy, const K: usize> CartCell<T, K> {
    /// Constructs a new [`CartCell`] from the given cell index `idx`
    /// and mesh `msh`.
    pub fn from_msh_and_idx(idx: CartCellIdx<K>, msh: &CartMesh<T, K>) -> Self {
        let idx_a = idx.0;
        let idx_b = idx_a.map(|i| i + 1);
        let a = msh.vertex(idx_a);
        let b = msh.vertex(idx_b);
        CartCell::new(a, b)
    }
    
    /// Returns an iterator over the corner points of this cell.
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
        zip(&self.a.coords, &self.b.coords).map(|(&a, &b)| a..=b).collect_array::<K>().unwrap()
    }
    
    /// Returns the [Self::ranges] as 1D intervals.
    pub fn intervals(&self) -> [CartCell<T, 1>; K] {
        zip(&self.a.coords, &self.b.coords)
            .map(|(&a, &b)| CartCell::new_univariate(a, b))
            .collect_array()
            .unwrap()
    }
}

impl <T: RealField + Copy, const D: usize> geo::Cell<T> for CartCell<T, D> {
    type ParametricCell = UnitCube<D>;
    type GeoMap = MultiLerp<T, D>;

    fn ref_cell(&self) -> Self::ParametricCell {
        UnitCube
    }

    fn geo_map(&self) -> Self::GeoMap {
        MultiLerp::new(self.a, self.b)
    }
}

/// Multi-index of a [`CartCell`].
#[derive(Debug, Clone, Copy)]
pub struct CartCellIdx<const K: usize>(pub [usize; K]);

impl <const K: usize> Cell<Const<K>> for CartCellIdx<K> {
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

impl CellBoundary<U3> for CartCellIdx<3> {
    const NUM_SUB_CELLS: usize = 6;
    type SubCell = CartCellIdx<2>;
    type Boundary = ();

    fn boundary(&self) -> Self::Boundary {
        todo!("Get")
    }
}