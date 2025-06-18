use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
use nalgebra::RealField;

/// A [`D`]-dimensional cell with geometric information embedded in [`M`]-dimensional space.
pub trait Cell<T: RealField, X: Dimensioned<T, D>, const D: usize, const M: usize> {
    /// Reference cell in the parametric domain for the mapping.
    type RefCell;

    /// Parametrization of this cell.
    type GeoMap: Chart<T, X, D, M>;

    /// Returns the reference cell of this cell.
    fn ref_cell(&self) -> Self::RefCell;

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