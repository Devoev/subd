use crate::basis::eval::EvalBasis;
use crate::diffgeo::chart::Chart;
use crate::subd::patch::CatmarkPatch;
use nalgebra::{Point, RealField, SMatrix};

// todo: possibly replace with reference to CatmarkPatch

/// Parametrization of a [`CatmarkPatch`].
pub struct CatmarkMap<T: RealField, const M: usize>(pub CatmarkPatch<T, M>);

impl <T: RealField + Copy, const M: usize> Chart<T, (T, T), 2, M> for CatmarkMap<T, M> {
    fn eval(&self, x: (T, T)) -> Point<T, M> {
        let b = self.0.basis().eval(x);
        let c = self.0.coords();
        Point::from((b * c).transpose())
    }

    fn eval_diff(&self, x: (T, T)) -> SMatrix<T, M, 2> {
        todo!("Copy from subd_legacy::surface or re-implement")
    }
}