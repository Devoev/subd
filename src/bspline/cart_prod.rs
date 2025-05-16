use std::marker::PhantomData;
use nalgebra::{stack, Const, Dyn, MatrixXx2, MatrixXx3, OMatrix, RealField, U2, U3};
use crate::bspline::basis::BsplineBasis;

/// Cartesian product of two scalar bases [`B1`] and [`B2`].
#[derive(Debug, Clone, Copy)]
pub struct Prod<T: RealField, X, B1: BsplineBasis<T, X, 1>, B2: BsplineBasis<T, X, 1>> {
    /// First basis.
    b1: B1,

    /// Second basis.
    b2: B2,

    _phantom_data: PhantomData<(T, X)>,
}


impl <T: RealField, X: Copy, B1: BsplineBasis<T, X, 1>, B2: BsplineBasis<T, X, 1>> Prod<T, X, B1, B2> {
    /// Constructs a new [`Prod`] from the bases `b1` and `b2`.
    pub fn new(b1: B1, b2: B2) -> Self {
        Prod { b1, b2, _phantom_data: Default::default() }
    }
}

#[allow(clippy::toplevel_ref_arg)]
impl<T, X, B1, B2> BsplineBasis<T, X, 2> for Prod<T, X, B1, B2>
    where T: RealField,
          X: Copy,
          B1: BsplineBasis<T, X, 1>,
          B2: BsplineBasis<T, X, 1>
{
    type NonzeroIndices = impl Iterator<Item=usize>;

    fn len(&self) -> usize {
        self.b1.len() + self.b2.len()
    }

    fn eval_nonzero(&self, x: X) -> (MatrixXx2<T>, Self::NonzeroIndices) {
        let (b1, idx1) = self.b1.eval_nonzero(x);
        let (b2, idx2) = self.b2.eval_nonzero(x);
        let b = stack![b1, 0;
                                  0, b2];
        let idx = idx1.chain(idx2); // todo: chain and add offsets to idx2
        (b, idx)
    }
}

/// Cartesian product of three scalar bases [`B1`], [`B2`] and [`B3`].
#[derive(Debug, Clone, Copy)]
pub struct TriProd<T: RealField, X, B1, B2, B3> {
    /// First basis.
    b1: B1,

    /// Second basis.
    b2: B2,

    /// Third basis.
    b3: B3,

    _phantom_data: PhantomData<(T, X)>,
}

#[allow(clippy::toplevel_ref_arg)]
impl <T, X, B1, B2, B3> TriProd<T, X, B1, B2, B3>
    where T: RealField,
          X: Copy,
          B1: BsplineBasis<T, X, 1>,
          B2: BsplineBasis<T, X, 1>,
          B3: BsplineBasis<T, X, 1>
{
    /// Constructs a new [`TriProd`] from the bases `b1`, `b2` and `b3`.
    pub fn new(b1: B1, b2: B2, b3: B3) -> Self {
        TriProd { b1, b2, b3, _phantom_data: Default::default() }
    }
}

#[allow(clippy::toplevel_ref_arg)]
impl<T, X, B1, B2, B3> BsplineBasis<T, X, 3> for TriProd<T, X, B1, B2, B3>
    where T: RealField,
          X: Copy,
          B1: BsplineBasis<T, X, 1>,
          B2: BsplineBasis<T, X, 1>,
          B3: BsplineBasis<T, X, 1>
{
    type NonzeroIndices = impl Iterator<Item=usize>;

    fn len(&self) -> usize {
        self.b1.len() + self.b2.len() + self.b3.len()
    }

    fn eval_nonzero(&self, x: X) -> (MatrixXx3<T>, Self::NonzeroIndices) {
        let (b1, idx1) = self.b1.eval_nonzero(x);
        let (b2, idx2) = self.b2.eval_nonzero(x);
        let (b3, idx3) = self.b3.eval_nonzero(x);

        let b = stack![b1, 0, 0;
                                  0, b2, 0;
                                  0, 0, b3];
        let idx = idx1.chain(idx2).chain(idx3); // todo: chain and add offsets to idx2
        (b, idx)
    }
}