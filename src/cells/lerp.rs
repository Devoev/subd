use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::cells::cartesian::CartCell;
use crate::cells::line_segment::LineSegment;
use crate::cells::unit_cube::UnitCube;
use crate::cells::quad::Quad;
use crate::diffgeo::chart::Chart;
use crate::subd::lin_subd::basis::LinBasisQuad;
use nalgebra::{Const, Matrix, Point, RealField, SMatrix, SVector, Scalar, U1, U2};

/// **L**inear int**erp**olation (Lerp) between two points in [`M`] dimensions.
///
/// Linearly transforms the [unit interval](UnitCube) `[0,1]` to a [`LineSegment`]
///   by the mapping
/// ```text
/// t ↦ (1 - t) a + t b = (b - a) t + a
/// ```
/// where `a` and `b` are the start and end coordinates of the line segment respectively.
pub struct Lerp<T: Scalar, const M: usize> {
    /// Start coordinates.
    pub a: Point<T, M>,

    /// End coordinates.
    pub b: Point<T, M>,
}

impl<T: Scalar, const M: usize> Lerp<T, M> {
    /// Constructs a new [`Lerp`] from the given coordinate vectors `a` and `b`.
    pub fn new(a: Point<T, M>, b: Point<T, M>) -> Self {
        Lerp { a, b }
    }
}

impl <T: RealField + Copy, const M: usize> Chart<T> for Lerp<T, M> {
    type Coord = T;
    type ParametricDim = U1;
    type GeometryDim = Const<M>;

    fn eval(&self, x: Self::Coord) -> Point<T, M> {
        self.a.lerp(&self.b, x)
    }

    fn eval_diff(&self, _x: Self::Coord) -> SMatrix<T, M, 1> {
        self.b - self.a
    }
}

/// Volumetric Lerp in [`M`] dimensional Euclidean space.
///
/// Linearly transforms the [unit cube](UnitCube) `[0,1]^M` to a [`CartCell`]
/// by the component-wise mapping
/// ```text
/// x[i] ↦ (1 - x[i]) a[i] + x[i] b[i] .
/// ```
/// where `a` and `b` are the start and end coordinates of the cell respectively.
/// For `x[i] = t` the mapping is the same as the univariate [`Lerp`].
pub struct MultiLerp<T: Scalar, const M: usize> {
    /// Start coordinates.
    pub a: Point<T, M>,

    /// End coordinates.
    pub b: Point<T, M>,
}

impl<T: Scalar, const M: usize> MultiLerp<T, M> {
    /// Constructs a new [`MultiLerp`] from the given coordinate vectors `a` and `b`.
    pub fn new(a: Point<T, M>, b: Point<T, M>) -> Self {
        MultiLerp { a, b }
    }
}

// todo: possibly move below transform methods to generalized Lerp or something else
impl <T: RealField + Copy, const M: usize> MultiLerp<T, M> {
    /// Linearly transforms the unit-hypercube `[-1,1]^M` to `[a,b]`.
    pub fn transform(&self, x: SVector<T, M>) -> Point<T, M> {
        let ones = SVector::repeat(T::one());
        let p = (ones - x).component_mul(&self.a.coords) + x.component_mul(&self.b.coords);
        Point::from(p)
    }

    /// Linearly transforms an arbitrary cartesian cell `ref_elem` to `[a, b]`.
    pub fn transform_cart_cell(&self, ref_elem: CartCell<T, M>, x: Point<T, M>) -> Point<T, M> {
        let s = ref_elem.a; // Start
        let e = ref_elem.b; // End
        let p = self.a + (self.b - self.a).component_mul(&(x - s)).component_div(&(e - s));
        Point::from(p)
    }

    /// Linearly transforms the symmetric normalized hypercube `[-1,1]^M` to `[a,b]`.
    pub fn transform_symmetric(&self, x: SVector<T, M>) -> Point<T, M> {
        let ones = SVector::repeat(T::one());
        let p = self.a + (self.b - self.a).component_mul(&(x + ones)) / T::from_usize(2).unwrap();
        Point::from(p)
    }
    
    /// Returns the constant Jacobian matrix `J = diag(b - a)` of this transformation.
    pub fn jacobian(&self) -> SMatrix<T, M, M> {
        Matrix::from_diagonal(&(self.b - self.a))
    }
}

impl <T: RealField + Copy, const M: usize> Chart<T> for MultiLerp<T, M> {
    type Coord = [T; M];
    type ParametricDim = Const<M>;
    type GeometryDim = Const<M>;

    fn eval(&self, x: [T; M]) -> Point<T, M> {
        self.transform(SVector::from(x))
    }

    fn eval_diff(&self, _x: [T; M]) -> SMatrix<T, M, M> {
        self.jacobian()
    }
}

/// **Bil**inear int**erp**olation (Bi-Lerp) between four points in [`M`] dimensions.
///
/// Linearly transforms the [unit square](UnitCube) `[0,1]²` to the [`Quad`]
/// formed by the four given vertices,
/// by applying univariate Lerps in both parametric direction successively.
pub struct BiLerp<T: Scalar, const M: usize> {
    pub vertices: [Point<T, M>; 4]
}

impl<T: Scalar, const M: usize> BiLerp<T, M> {
    /// Constructs a new [`BiLerp`] from the given array of `vertices`.
    pub fn new(vertices: [Point<T, M>; 4]) -> Self {
        BiLerp { vertices }
    }

    /// Returns the `4✕M` matrix of the vertex coordinates.
    fn coords(&self) -> SMatrix<T, 4, M> {
        Matrix::from_rows(&self.vertices.clone().map(|p| p.coords.transpose()))
    }
}
// todo: replace implementation using lowest order nodal basis interpolation

impl <T: RealField + Copy, const M: usize> Chart<T> for BiLerp<T, M> {
    type Coord = (T, T);
    type ParametricDim = U2;
    type GeometryDim = Const<M>;

    fn eval(&self, x: (T, T)) -> Point<T, M> {
        let b = LinBasisQuad.eval(x);
        let c = self.coords();
        Point::from((b * c).transpose())
    }

    #[allow(clippy::toplevel_ref_arg)]
    fn eval_diff(&self, x: (T, T)) -> SMatrix<T, M, 2> {
        let grad_b = LinBasisQuad.eval_grad(x);
        let c = self.coords();
        (grad_b * c).transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::{matrix, point, vector};
    use rand::random_range;

    #[test]
    fn eval_lerp() {
        // Get random parametric value
        let t = random_range(0.0..=1.0);

        let phi = Lerp::new(point![2.0], point![4.0]);
        assert_abs_diff_eq!(phi.eval(0.0), point![2.0]);
        assert_abs_diff_eq!(phi.eval(0.25), point![2.5]);
        assert_abs_diff_eq!(phi.eval(0.5), point![3.0]);
        assert_abs_diff_eq!(phi.eval(0.75), point![3.5]);
        assert_abs_diff_eq!(phi.eval(1.0), point![4.0]);
        assert_abs_diff_eq!(phi.eval(t), point![(1.0 - t) * 2.0 + t * 4.0]);

        let phi = Lerp::new(point![-1.0, 0.0], point![1.0, 5.0]);
        assert_abs_diff_eq!(phi.eval(0.0), point![-1.0, 0.0]);
        assert_abs_diff_eq!(phi.eval(0.25), point![-0.5, 1.25]);
        assert_abs_diff_eq!(phi.eval(0.5), point![0.0, 2.5]);
        assert_abs_diff_eq!(phi.eval(0.75), point![0.5, 3.75]);
        assert_abs_diff_eq!(phi.eval(1.0), point![1.0, 5.0]);
        assert_abs_diff_eq!(phi.eval(t), point![-(1.0 - t) + t, 5.0 * t]);
    }

    #[test]
    fn eval_diff_lerp() {
        // Get random parametric value
        let t = random_range(0.0..=1.0);

        let phi = Lerp::new(point![2.0], point![4.0]);
        assert_abs_diff_eq!(phi.eval_diff(t), vector![2.0]);

        let phi = Lerp::new(point![-1.0, 0.0], point![1.0, 5.0]);
        assert_abs_diff_eq!(phi.eval_diff(t), vector![2.0, 5.0]);
    }

    #[test]
    fn eval_bi_lerp() {
        // Get random parametric values
        let u = random_range(0.0..=1.0);
        let v = random_range(0.0..=1.0);

        let phi = BiLerp::new([point![0.0, 0.0], point![1.0, 0.0], point![1.0, 1.0], point![0.0, 1.0]]);
        assert_abs_diff_eq!(phi.eval((0.0, 0.0)), point![0.0, 0.0]);
        assert_abs_diff_eq!(phi.eval((1.0, 0.0)), point![1.0, 0.0]);
        assert_abs_diff_eq!(phi.eval((1.0, 1.0)), point![1.0, 1.0]);
        assert_abs_diff_eq!(phi.eval((0.0, 1.0)), point![0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval((u, v)), point![u, v]);
    }

    #[test]
    fn eval_diff_bi_lerp() {
        let phi = BiLerp::new([point![0.0, 0.0], point![1.0, 0.0], point![1.0, 1.0], point![0.0, 1.0]]);
        assert_abs_diff_eq!(phi.eval_diff((0.0, 0.0)), matrix![1.0, 0.0; 0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval_diff((0.5, 0.0)), matrix![1.0, 0.0; 0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval_diff((1.0, 0.0)), matrix![1.0, 0.0; 0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval_diff((1.0, 0.5)), matrix![1.0, 0.0; 0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval_diff((1.0, 1.0)), matrix![1.0, 0.0; 0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval_diff((0.5, 1.0)), matrix![1.0, 0.0; 0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval_diff((0.0, 1.0)), matrix![1.0, 0.0; 0.0, 1.0]);
        assert_abs_diff_eq!(phi.eval_diff((0.0, 0.5)), matrix![1.0, 0.0; 0.0, 1.0]);
    }
}