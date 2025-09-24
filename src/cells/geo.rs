use crate::diffgeo::chart::{Chart, ChartAllocator};
use crate::index::dimensioned::Dimensioned;
use nalgebra::{Const, DefaultAllocator, RealField, Scalar};
use crate::basis::traits::Basis;
use crate::cells::topo;
use crate::mesh::traits::{Mesh, MeshTopology, VertexStorage};

/// A [`ChartAllocator`] for the [`C::GeoMap`] of a cell.
pub trait CellAllocator<T: Scalar, C: Cell<T>>: ChartAllocator<T, C::GeoMap>
    where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}

impl<T: Scalar, C: Cell<T>> CellAllocator<T, C> for DefaultAllocator
    where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}

/// A [`D`]-dimensional cell with geometric information embedded in [`M`]-dimensional space.
pub trait Cell<T: Scalar>
    where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{
    /// Reference cell in the parametric domain for the mapping.
    type ParametricCell;

    /// Parametrization of this cell.
    type GeoMap: Chart<T>;

    /// Returns the reference cell of this cell.
    fn ref_cell(&self) -> Self::ParametricCell;

    /// Returns the parametrization of this cell.
    fn geo_map(&self) -> Self::GeoMap;
    
    // todo: maybe merge Chart and Cell traits?
}

// todo: this could be an alternative to the to_geo_cell method in topo::Cell
//  what is better?
pub trait CellInMesh<T: Scalar>: Cell<T> {
    
    type TopoCell;

    /// Coordinates storage of the associated mesh.
    type Coords: VertexStorage<T, <Self::GeoMap as Chart<T>>::GeometryDim>;

    /// Cell topology of the associated mesh.
    type Cells: MeshTopology;
    
    fn from_msh(cell: &Self::TopoCell, msh: &Mesh<T, M, Self::Coords, Self::Cells>) ->
}

// todo: possibly rename and reorganize these traits and types

/// Coordinate of the [`Cell::GeoMap`] of `C`.
pub type CellCoord<T, C> = <<C as Cell<T>>::GeoMap as Chart<T>>::Coord;

/// Constrains `Self` to have a chart with coordinate [`B::Coord`].
pub trait HasBasisCoord<T: Scalar, B: Basis>: Cell<T, GeoMap: Chart<T, Coord = B::Coord<T>>>
    where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}

impl <T: Scalar, B: Basis, C: Cell<T, GeoMap: Chart<T, Coord = B::Coord<T>>>> HasBasisCoord<T, B> for C
where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}

/// Constrains `Self` to have a chart with both parametric and geometry dimension [`D`].
pub trait HasDim<T: Scalar, const D: usize>: Cell<T, GeoMap: Chart<T, ParametricDim = Const<D>, GeometryDim = Const<D>>>
where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}

impl <T: Scalar, const D: usize, C: Cell<T, GeoMap: Chart<T, ParametricDim = Const<D>, GeometryDim = Const<D>>>> HasDim<T, D> for C
where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}

// todo: maybe remove this trait
/// A [`D`]-dimensional reference cell in the parametric domain.
pub trait RefCell<T: Scalar, X: Dimensioned<T, D>, const D: usize> {
    /// Returns an iterator of `steps` discrete coordinate points in the parametric domain.
    fn coords(steps: usize) -> impl Iterator<Item = X>;
}