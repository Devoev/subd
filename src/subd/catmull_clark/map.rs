use crate::space::eval_basis::{EvalBasis, EvalGrad};
use crate::diffgeo::chart::Chart;
use nalgebra::{Const, Point, RealField, SMatrix, U2};
use num_traits::ToPrimitive;
use crate::subd::catmull_clark::basis::CatmarkPatchBasis;
use crate::subd::catmull_clark::patch::CatmarkPatch;
// todo: possibly replace with reference to CatmarkPatch or with just CatmarkPatch

/// Parametrization of a [`CatmarkPatch`].
pub struct CatmarkMap<T: RealField, const M: usize>(pub CatmarkPatch<T, M>);

impl <T: RealField + Copy + ToPrimitive, const M: usize> Chart<T> for CatmarkMap<T, M> {
    type Coord = (T, T);
    type ParametricDim = U2;
    type GeometryDim = Const<M>;

    fn eval(&self, x: (T, T)) -> Point<T, M> {
        let b = CatmarkPatchBasis::from(&self.0).eval(x);
        let c = self.0.coords();
        Point::from((b * c).transpose())
    }

    fn eval_diff(&self, x: (T, T)) -> SMatrix<T, M, 2> {
        let grad_b = CatmarkPatchBasis::from(&self.0).eval_grad(x);
        let c = self.0.coords();
        (grad_b * c).transpose()
    }
}