use itertools::Itertools;
use nalgebra::iter::ColumnIter;
use nalgebra::{Const, Dim, Dyn, Matrix, OMatrix, Owned, Point, RealField, Storage, VectorView, ViewStorage};

/// The coordinates of control points, stored column wise as a matrix.
type Coords<T, const M: usize, N, S> = Matrix<T, Const<M>, N, S>;

/// Control points of a spline in [M] dimensions.
/// 
/// # Type parameters
/// - [`T`]: Real scalar type
/// - [`M`]: Size of the embedding euclidian space, i.e. number of coordinates of each point.
/// - [`N`]: Total number of control points.
/// - [`S`]: Underlying storage for the [`Coords`].
#[derive(Debug, Clone)]
pub struct ControlPoints<T: RealField, const M: usize, N: Dim, S: Storage<T, Const<M>, N>> {
    /// Control point coordinates.
    pub coords: Coords<T, M, N, S>,
}

/// Owned control points.
pub type OControlPoints<T, const M: usize, N> = ControlPoints<T, M, N, Owned<T, Const<M>, N>>;

/// Control points view.
pub type ControlPointsView<'a, T, const M: usize, N, RStride, CStride> = ControlPoints<T, M, N, ViewStorage<'a, T, Const<M>, N, RStride, CStride>>;

impl<T: RealField + Copy, const M: usize, N: Dim, S: Storage<T, Const<M>, N>> ControlPoints<T, M, N, S> {
    /// Constructs new [ControlPoints].
    pub fn new(coords: Coords<T, M, N, S>) -> Self {
        ControlPoints { coords }
    }

    /// The number of control points, i.e. the number of columns of `self.coords`
    pub fn num(&self) -> usize {
        self.coords.shape().1
    }

    /// Returns all control points belonging to nonzero basis functions, 
    /// determined by the `nonzero_indices`.
    pub fn get_nonzero(&self, nonzero_indices: impl Iterator<Item=usize>) -> OControlPoints<T, M, Dyn> {
        let columns = nonzero_indices.map(|i| self.coords.column(i)).collect_vec();
        let coords = OMatrix::<T, Const<M>, Dyn>::from_columns(&columns);
        ControlPoints::new(coords)
    }
}

impl <T: RealField, const M: usize, N: Dim, S: Storage<T, Const<M>, N>> ControlPoints<T, M, N, S> {
    /// Iterates through the control points as matrix views.
    pub fn iter(&self) -> ColumnIter<'_, T, Const<M>, N, S> {
        self.coords.column_iter()
    }

    /// Iterates through the control points as owned [points][Point].
    pub fn point_iter(&self) -> impl Iterator<Item=Point<T, M>> + '_ {
        self.iter().map(|col| Point::from(col.clone_owned()))
    }
}

impl <'a, T: RealField, const M: usize, N: Dim, S: Storage<T, Const<M>, N>> IntoIterator for &'a ControlPoints<T, M, N, S> {

    type Item = VectorView<'a, T, Const<M>, S::RStride, S::CStride>;
    type IntoIter = ColumnIter<'a, T, Const<M>, N, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}