use crate::knots::knot_span::KnotSpan;
use nalgebra::{Const, Dim, Dyn, Matrix, Owned, RawStorage, RealField, Storage, ViewStorage};


/// The coordinates of control points, stored column wise as a matrix.
type Coords<T, const M: usize, S> = Matrix<T, Const<M>, Dyn, S>;

/// Control points of a spline in `M` dimensions.
#[derive(Debug, Clone)]
pub struct ControlPoints<T: RealField, const M: usize, S: Storage<T, Const<M>, Dyn>> {
    /// Control point coordinates.
    pub coords: Coords<T, M, S>,
}

/// Owned control points.
pub type OControlPoints<T, const M: usize> = ControlPoints<T, M, Owned<T, Const<M>, Dyn>>;

/// Control points view.
pub type ControlPointsView<'a, T, const M: usize, RStride, CStride> = ControlPoints<T, M, ViewStorage<'a, T, Const<M>, Dyn, RStride, CStride>>;

impl<T: RealField + Copy, const M: usize, S: Storage<T, Const<M>, Dyn>> ControlPoints<T, M, S> {
    /// Constructs new [ControlPoints].
    pub fn new(coords: Coords<T, M, S>) -> Self {
        ControlPoints { coords }
    }

    /// The number of control points, i.e. the number of columns of `self.coords`
    pub fn num(&self) -> usize {
        self.coords.shape().1
    }

    /// Returns all control points belonging to nonzero basis functions in the given `span`.
    pub fn get_nonzero(&self, span: KnotSpan<T>, p: usize) -> ControlPointsView<T, M, S::RStride, S::CStride> {
        match M {
            1 => {
                ControlPoints::new(self.coords.columns_range(span.nonzero_indices(p)))
            }
            _ => todo!("Implement for multivariate splines by multivariate knot spans")
        }
    }
}