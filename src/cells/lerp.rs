use crate::cells::cartesian::CartCell;
use crate::cells::line_segment::LineSegment;
use crate::cells::unit_cube::UnitCube;
use crate::diffgeo::chart::Chart;
use nalgebra::{stack, Matrix, Point, RealField, SMatrix, SVector, Scalar};

/// **L**inear int**erp**olation (Lerp) in [`M`] dimensional Euclidean space.
///
/// # Transformation properties
/// There exist 2 versions of a Lerp:
/// - Univariate transformation of the [unit interval](UnitCube) `[0,1]` to a [`LineSegment`]
///   by the mapping
/// ```text
/// t ↦ (1 - t) a + t b = (b - a) t + a
/// ```
/// where `a` and `b` are the start and end coordinates of the rectangle respectively.
/// - [`M`]-variate transformation of the [unit cube](UnitCube) `[0,1]^M` to a [`CartCell`]
/// by the component-wise mapping
/// ```text
/// x[i] ↦ (1 - x[i]) a[i] + x[i] b[i] .
/// ```
/// For `x[i] = t` the mapping is the same as the univariate case.
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

// todo: possibly move below transform methods to generalized Lerp or something else
impl <T: RealField + Copy, const M: usize> Lerp<T, M> {
    /// Linearly transforms an arbitrary hyper rectangle `ref_elem` to `[a, b]`.
    pub fn transform(&self, ref_elem: CartCell<T, M>, x: Point<T, M>) -> Point<T, M> {
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

impl <T: RealField + Copy, const M: usize> Chart<T, [T; M], M, M> for Lerp<T, M> {
    fn eval(&self, x: [T; M]) -> Point<T, M> {
        self.eval(SVector::from(x))
    }

    fn eval_diff(&self, _x: [T; M]) -> SMatrix<T, M, M> {
        self.jacobian()
    }
}

impl <T: RealField + Copy, const M: usize> Chart<T, SVector<T, M>, M, M> for Lerp<T, M> {
    fn eval(&self, x: SVector<T, M>) -> Point<T, M> {
        let ones = SVector::repeat(T::one());
        let p = (ones - x).component_mul(&self.a.coords) + x.component_mul(&self.b.coords);
        Point::from(p)
    }

    fn eval_diff(&self, _x: SVector<T, M>) -> SMatrix<T, M, M> {
        self.jacobian()
    }
}

impl <T: RealField, const M: usize> Chart<T, T, 1, M> for Lerp<T, M> {
    fn eval(&self, x: T) -> Point<T, M> {
        self.a.lerp(&self.b, x)
    }

    fn eval_diff(&self, _x: T) -> SMatrix<T, M, 1> {
        &self.b - &self.a
    }
}

// todo: possibly remove multivariate transformation of Lerp and add MultiLerp?

/// Bilinear interpolation between 4 vertices in [`M`]-dimensional Euclidean space.
pub struct BiLerp<T: Scalar, const M: usize> {
    pub vertices: [Point<T, M>; 4]
}

impl<T: Scalar, const M: usize> BiLerp<T, M> {
    /// Constructs a new [`Lerp`] from the given array of `vertices`.
    pub fn new(vertices: [Point<T, M>; 4]) -> Self {
        BiLerp { vertices }
    }
}

impl <T: RealField + Copy, const M: usize> Chart<T, (T, T), 2, M> for BiLerp<T, M> {
    fn eval(&self, x: (T, T)) -> Point<T, M> {
        let [q11, q12, q22, q21] = self.vertices;
        let (u, v) = x;
        let l1 = Lerp::new(q11, q12);              // q11.lerp(q12, u);
        let l2 = Lerp::new(q21, q22);              // q21.lerp(q22, u);
        let l = Lerp::new(l1.eval(u), l2.eval(u)); // l1.lerp(&l2, v)
        l.eval(v)
    }

    #[allow(clippy::toplevel_ref_arg)]
    fn eval_diff(&self, x: (T, T)) -> SMatrix<T, M, 2> {
        let [q11, q12, q22, q21] = self.vertices;
        let (u, v) = x;
        let l1 = Lerp::new(q11, q12); // let l1 = q11.lerp(q12, u);
        let l2 = Lerp::new(q21, q22); // let l2 = q21.lerp(q22, u);
        let l_du = Lerp::new(         // (q12 - q11).lerp(&(q22 - q21), v),
            Point::from(l1.eval_diff(u)),
            Point::from(l2.eval_diff(u))
        );
        stack![
            l_du.eval(v).coords,
            l2.eval(u) - l2.eval(u) // l2 - l1
        ]
    }
}