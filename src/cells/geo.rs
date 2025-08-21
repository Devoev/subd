use crate::diffgeo::chart::{Chart, ChartAllocator};
use crate::index::dimensioned::Dimensioned;
use nalgebra::{DefaultAllocator, RealField};

/// A [`ChartAllocator`] for the [`C::GeoMap`] of a cell.
pub trait CellAllocator<T: RealField, C: Cell<T>>: ChartAllocator<T, C::GeoMap> 
    where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}

impl<T: RealField, C: Cell<T>> CellAllocator<T, C> for DefaultAllocator
    where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}

/// A [`D`]-dimensional cell with geometric information embedded in [`M`]-dimensional space.
pub trait Cell<T: RealField> 
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

// todo: maybe remove this trait
/// A [`D`]-dimensional reference cell in the parametric domain.
pub trait RefCell<T: RealField, X: Dimensioned<T, D>, const D: usize> {
    /// Returns an iterator of `steps` discrete coordinate points in the parametric domain.
    fn coords(steps: usize) -> impl Iterator<Item = X>;
}