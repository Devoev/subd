use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
use nalgebra::RealField;

/// A [`D`]-dimensional cell with geometric information embedded in [`M`]-dimensional space.
pub trait Cell<T: RealField, X: Dimensioned<T, D>, const D: usize, const M: usize> {
    /// Parametrization of this cell.
    type GeoMap: Chart<T, X, D, M>;
    /// Returns the parametrization of this cell.
    fn geo_map(&self) -> Self::GeoMap;
    
    // todo: maybe add methods for jacobian?
}