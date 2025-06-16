use crate::cells::hyper_rectangle::HyperRectangle;
use crate::diffgeo::chart::Chart;
use nalgebra::{Matrix, Point, Point1, RealField, SMatrix, SVector};

/// **L**inear int**erp**olation (Lerp) in [`D`] dimensions.
/// Transforms the unit hypercube `[0,1]^D` to a [`HyperRectangle`] by the component-wise mapping
/// ```text
/// x[i] ↦ (1 - x[i]) a[i] + x[i] b[i]
/// ```
/// where `a` and `b` are the start and end coordinates of the rectangle respectively.
pub struct Lerp<T, const D: usize> {
    /// Start coordinates.
    pub a: SVector<T, D>,

    /// End coordinates.
    pub b: SVector<T, D>,
}

impl<T, const D: usize> Lerp<T, D> {
    /// Constructs a new [`Lerp`] from the given coordinate vectors `a` and `b`.
    pub fn new(a: SVector<T, D>, b: SVector<T, D>) -> Self {
        Lerp { a, b }
    }
}

// todo: possibly move below transform methods to generalized Lerp or something else
impl <T: RealField + Copy, const D: usize> Lerp<T, D> {
    /// Linearly transforms an arbitrary hyper rectangle `ref_elem` to `[a, b]`.
    pub fn transform(&self, ref_elem: HyperRectangle<T, D>, x: SVector<T, D>) -> Point<T, D> {
        let s = ref_elem.a; // Start
        let e = ref_elem.b; // End
        let p = self.a + (self.b - self.a).component_mul(&(x - s)).component_div(&(e - s));
        Point::from(p)
    }

    /// Linearly transforms the symmetric normalized hypercube `[-1,1]^D` to `[a,b]`.
    pub fn transform_symmetric(&self, x: SVector<T, D>) -> Point<T, D> {
        let ones = SVector::repeat(T::one());
        let p = self.a + (self.b - self.a).component_mul(&(x + ones)) / T::from_usize(2).unwrap();
        Point::from(p)
    }
    
    /// Returns the constant Jacobian matrix `J = diag(b - a)` of this transformation.
    pub fn jacobian(&self) -> SMatrix<T, D, D> {
        Matrix::from_diagonal(&(self.b - self.a))
    }
}

impl <T: RealField + Copy, const D: usize> Chart<T, [T; D], D, D> for Lerp<T, D> {
    fn eval(&self, x: [T; D]) -> Point<T, D> {
        self.eval(SVector::from(x))
    }

    fn eval_diff(&self, _x: [T; D]) -> SMatrix<T, D, D> {
        self.jacobian()
    }
}

impl <T: RealField + Copy, const D: usize> Chart<T, SVector<T, D>, D, D> for Lerp<T, D> {
    fn eval(&self, x: SVector<T, D>) -> Point<T, D> {
        let ones = SVector::repeat(T::one());
        let p = (ones - x).component_mul(&self.a) + x.component_mul(&self.b);
        Point::from(p)
    }

    fn eval_diff(&self, _x: SVector<T, D>) -> SMatrix<T, D, D> {
        self.jacobian()
    }
}

impl <T: RealField + Copy> Chart<T, T, 1, 1> for Lerp<T, 1> {
    fn eval(&self, x: T) -> Point<T, 1> {
        Point1::new((T::one() - x) * self.a.x + x * self.b.x)
    }

    fn eval_diff(&self, _x: T) -> SMatrix<T, 1, 1> {
        self.b - self.a
    }
}