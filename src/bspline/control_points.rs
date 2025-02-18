use crate::knots::knot_span::KnotSpan;
use nalgebra::iter::ColumnIter;
use nalgebra::{Const, Dim, Dyn, Matrix, Owned, Point, RealField, Storage, VectorView, ViewStorage};

/// The coordinates of control points, stored column wise as a matrix.
type Coords<T, const M: usize, C, S> = Matrix<T, Const<M>, C, S>;

/// Control points of a spline in `M` dimensions.
#[derive(Debug, Clone)]
pub struct ControlPoints<T: RealField, const M: usize, C: Dim, S: Storage<T, Const<M>, C>> {
    /// Control point coordinates.
    pub coords: Coords<T, M, C, S>,
}

/// Owned control points.
pub type OControlPoints<T, const M: usize, C> = ControlPoints<T, M, C, Owned<T, Const<M>, C>>;

/// Control points view.
pub type ControlPointsView<'a, T, const M: usize, C, RStride, CStride> = ControlPoints<T, M, C, ViewStorage<'a, T, Const<M>, C, RStride, CStride>>;

impl<T: RealField + Copy, const M: usize, C: Dim, S: Storage<T, Const<M>, C>> ControlPoints<T, M, C, S> {
    
    /// Constructs new [ControlPoints].
    pub fn new(coords: Coords<T, M, C, S>) -> Self {
        ControlPoints { coords }
    }

    /// The number of control points, i.e. the number of columns of `self.coords`
    pub fn num(&self) -> usize {
        self.coords.shape().1
    }

    /// Returns all control points belonging to nonzero basis functions in the given `span`.
    pub fn get_nonzero<const D: usize>(&self, span: [KnotSpan<T>; D], p: usize) -> ControlPointsView<T, M, Dyn, S::RStride, S::CStride> {
        match D {
            1 => {
                ControlPoints::new(self.coords.columns_range(span[0].nonzero_indices(p)))
            }
            2 => {
                
                todo!()
            }
            _ => todo!("Implement for multivariate splines by multivariate knot spans")
        }
    }
}

impl <T: RealField, const M: usize, C: Dim, S: Storage<T, Const<M>, C>> ControlPoints<T, M, C, S> {
    
    /// Iterates through the control points as matrix views.
    pub fn iter(&self) -> ColumnIter<'_, T, Const<M>, C, S> {
        self.coords.column_iter()
    }
    
    /// Iterates through the control points as owned [Point]s.
    pub fn point_iter(&self) -> impl Iterator<Item=Point<T, M>> + '_ {
        self.iter().map(|col| Point::from(col.clone_owned()))
    }
}

impl <'a, T: RealField, const M: usize, C: Dim, S: Storage<T, Const<M>, C>> IntoIterator for &'a ControlPoints<T, M, C, S> {
    
    type Item = VectorView<'a, T, Const<M>, S::RStride, S::CStride>;
    type IntoIter = ColumnIter<'a, T, Const<M>, C, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}