use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OMatrix, OPoint, Scalar};

// todo: remove T in generic bound by moving it either to a GAT on Coord, or introduce new trait
//  that has GeometryDim and ParametricDim types

/// Allocator for the vector of size [`C::GeometryDim`] of chart evaluations
/// and the [`C::GeometryDim`] ✕ [`C::ParametricDim`] matrix of differential evaluations.
pub trait ChartAllocator<T: Scalar, C: Chart<T>>: Allocator<C::GeometryDim> + Allocator<C::GeometryDim, C::ParametricDim> 
    where  DefaultAllocator: Allocator<C::GeometryDim> + Allocator<C::GeometryDim, C::ParametricDim>
{}

impl<T: Scalar, C: Chart<T>> ChartAllocator<T, C> for DefaultAllocator
    where DefaultAllocator: Allocator<C::GeometryDim> + Allocator<C::GeometryDim, C::ParametricDim>
{}

/// A chart mapping points in the physical domain to the parametric domain.
pub trait Chart<T: Scalar>: Sized
    where DefaultAllocator: ChartAllocator<T, Self>
{
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