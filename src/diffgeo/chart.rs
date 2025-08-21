use crate::index::dimensioned::Dimensioned;
use nalgebra::{DimName, OMatrix, OPoint, Point, SMatrix, Scalar};

/// A chart mapping points in the physical domain to the parametric domain.
pub trait Chart<T: Scalar> {
    /// Coordinate of a parametric point.
    type Coord;

    /// Dimension of the parametric domain.
    type ParametricDim: DimName;

    /// Dimension of the geometrical domain.
    type GeometryDim: DimName;
    
    /// Evaluates the parametrization of this chart, i.e. the inverse mapping.
    fn eval(&self, x: Self::Coord) -> OPoint<T, Self::GeometryDim>;
    
    /// Evaluates the coordinate representation (Jacobian matrix) 
    /// of the differential aka. pushforward at `x`.
    fn eval_diff(&self, x: Self::Coord) -> OMatrix<T, Self::GeometryDim, Self::ParametricDim>;
}