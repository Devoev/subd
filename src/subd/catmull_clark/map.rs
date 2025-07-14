use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::diffgeo::chart::Chart;
use nalgebra::{Point, RealField, SMatrix};
use num_traits::ToPrimitive;
use crate::subd::catmull_clark::patch::CatmarkPatch;
// todo: possibly replace with reference to CatmarkPatch or with just CatmarkPatch

/// Parametrization of a [`CatmarkPatch`].
pub struct CatmarkMap<T: RealField, const M: usize>(pub CatmarkPatch<T, M>);

impl <T: RealField + Copy + ToPrimitive, const M: usize> Chart<T, (T, T), 2, M> for CatmarkMap<T, M> {
    fn eval(&self, x: (T, T)) -> Point<T, M> {
        let b = self.0.basis().eval(x);
        let c = self.0.coords();
        Point::from((b * c).transpose())
    }

    fn eval_diff(&self, x: (T, T)) -> SMatrix<T, M, 2> {
        let grad_b = self.0.basis().eval_grad(x);
        let c = self.0.coords();
        (grad_b * c).transpose()
    }
}