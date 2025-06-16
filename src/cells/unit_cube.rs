use crate::cells::geo::RefCell;
use iter_num_tools::lin_space;
use itertools::{repeat_n, Itertools};
use nalgebra::RealField;

// todo:
//  1. possibly replace with MultiProd<UnitInterval> and UnitInterval as Unit<Interval>
//  2. store calculated coords directly in struct instead of computation in coords()

/// The [`D`]-dimensional unit hyper-cube `[0,1]^D`.
pub struct UnitCube<const D: usize>;

impl <T: RealField + Copy, const D: usize> RefCell<T, [T; D], D> for UnitCube<D> {
    fn coords(steps: usize) -> impl Iterator<Item=[T; D]> {
        let lin = lin_space(T::zero()..=T::one(), steps);
        repeat_n(lin, D).multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }
}

/// The [`D`]-dimensional symmetric normalized hyper-cube `[-1,1]^D`.
pub struct SymmetricUnitCube<const D: usize>;

impl <T: RealField + Copy, const D: usize> RefCell<T, [T; D], D> for SymmetricUnitCube<D> {
    fn coords(steps: usize) -> impl Iterator<Item=[T; D]> {
        let lin = lin_space(-T::one()..=T::one(), steps);
        repeat_n(lin, D).multi_cartesian_product()
            .map(|vec| vec.try_into().unwrap())
    }
}