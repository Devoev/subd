use crate::knots::knot_span::KnotSpan;
use nalgebra::{Const, Dyn, Matrix, RawStorage, RealField};


/// The coordinates of control points, stored column wise as a matrix.
type Coords<T, const M: usize, S> = Matrix<T, Const<M>, Dyn, S>;

type ViewStorage<'a, T, const M: usize, S> = nalgebra::ViewStorage<'a, T, Const<M>, Dyn, <S as RawStorage<T, Const<M>, Dyn>>::RStride, <S as RawStorage<T, Const<M>, Dyn>>::CStride>;

/// Control points of a spline in `M` dimensions.
struct ControlPoints<T: RealField, const M: usize, S: RawStorage<T, Const<M>, Dyn>> {
    /// Control point coordinates.
    pub coords: Coords<T, M, S>,
}

impl<T: RealField + Copy, const M: usize, S: RawStorage<T, Const<M>, Dyn>> ControlPoints<T, M, S> {
    /// Constructs new [ControlPoints].
    pub fn new(coords: Coords<T, M, S>) -> Self {
        ControlPoints { coords }
    }

    /// The number of control points, i.e. the number of columns of `self.coords`
    pub fn num(&self) -> usize {
        self.coords.shape().1
    }

    /// Returns all control points belonging to nonzero basis functions in the given `span`.
    pub fn get_nonzero(&self, span: KnotSpan<T>, p: usize) -> ControlPoints<T, M, ViewStorage<T, M, S>> {
        match M {
            1 => {
                ControlPoints::new(self.coords.columns_range(span.nonzero_indices(p)))
            }
            _ => todo!()
        }
    }
}