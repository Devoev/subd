use nalgebra::{DimName, RealField};
use crate::cells::chart::Chart;

/// A [`K`]-dimensional cell with geometric information.
pub trait Cell<T: RealField, X, K: DimName, const M: usize> {
    /// Parametrization of this cell.
    type GeoMap: Chart<T, X, M>;
    /// Returns the parametrization of this cell.
    fn geo_map(&self) -> Self::GeoMap;
    
    // todo: maybe add methods for jacobian?
}