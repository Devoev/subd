use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimNameAdd, DimNameSum, Point, RealField, TAffine, Transform, U1};

/// A chart mapping points in the physical domain to the parametric domain.
pub trait Chart<T: RealField, X, const M: usize> {
    /// Evaluates the parametrization of this chart, i.e. the inverse mapping.
    fn eval(&self, x: X) -> Point<T, M>;
}

impl <T: RealField, const D: usize> Chart<T, Point<T, D>, D> for Transform<T, TAffine, D>
    where Const<D>: DimNameAdd<U1>,
          DefaultAllocator: Allocator<DimNameSum<Const<D>, U1>, DimNameSum<Const<D>, U1>>,
{
    fn eval(&self, x: Point<T, D>) -> Point<T, D> {
        self * x
    }
}