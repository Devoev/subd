use crate::space::eval_basis::{EvalBasis, EvalBasisAllocator};
use crate::space::local::{FindElem, MeshBasis};
use crate::space::basis::BasisFunctions;
use nalgebra::{stack, DefaultAllocator, DimAdd, DimSum, OMatrix, RealField, U1, U2};
use std::iter::once;
use std::marker::PhantomData;

// todo: add proper docs

/// Cartesian product of two scalar bases [`B1`] and [`B2`].
#[derive(Debug, Clone, Copy)]
pub struct Prod<T, B1, B2> {
    /// Pair of first and second basis.
    bases: (B1, B2),

    _phantom_data: PhantomData<T>,
}


impl <T, B1, B2> Prod<T, B1, B2> {
    /// Constructs a new [`Prod`] from the bases `b1` and `b2`.
    pub fn new(bases: (B1, B2)) -> Self {
        Prod { bases, _phantom_data: Default::default() }
    }
}

/// Constrains that [`Self::NumBasis`] and [`Other::NumBasis`] can be added, i.e.
/// `Self::NumBasis: DimAdd<Other::NumBasis>`.
pub trait NumBasisAdd<Other: BasisFunctions>: BasisFunctions<NumBasis: DimAdd<Other::NumBasis>> {}

impl<B1: BasisFunctions, B2: BasisFunctions> NumBasisAdd<B2> for B1
    where B1::NumBasis: DimAdd<B2::NumBasis> {}

impl<T, B1, B2> BasisFunctions for Prod<T, B1, B2>
where B1: BasisFunctions<NumComponents = U1> + NumBasisAdd<B2>,
      B2: BasisFunctions<NumComponents = U1, ParametricDim = B1::ParametricDim, Coord<T> = B1::Coord<T>>,
{
    type NumBasis = DimSum<B1::NumBasis, B2::NumBasis>;
    type NumComponents = U2;
    type ParametricDim = B1::ParametricDim;
    type Coord<_T> = B1::Coord<_T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.bases.0.num_basis_generic().add(self.bases.1.num_basis_generic())
    }
}

#[allow(clippy::toplevel_ref_arg)]
impl <T, B1, B2> EvalBasis<T> for Prod<T, B1, B2>
where T: RealField,
      B1: EvalBasis<T, NumComponents = U1> + NumBasisAdd<B2>,
      B2: EvalBasis<T, NumComponents = U1, ParametricDim = B1::ParametricDim, Coord<T> = B1::Coord<T>>,
      B1::Coord<T>: Copy,
      DefaultAllocator: EvalBasisAllocator<B1> + EvalBasisAllocator<B2> + EvalBasisAllocator<Self>,
{
    fn eval(&self, x: Self::Coord<T>) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let b1 = self.bases.0.eval(x);
        let b2= self.bases.1.eval(x);
        stack![
            b1, 0;
            0, b2
        ]
    }
}

impl <T, B1, B2> MeshBasis<T> for Prod<T, B1, B2>
where T: RealField,
      B1: MeshBasis<T, NumComponents = U1> + NumBasisAdd<B2>,
      B2: MeshBasis<T, NumComponents = U1, ParametricDim = B1::ParametricDim, Coord<T> = B1::Coord<T>>,
      B1::LocalBasis: NumBasisAdd<B2::LocalBasis>,
      B1::Coord<T>: Copy,
      DefaultAllocator: EvalBasisAllocator<B1::LocalBasis> + EvalBasisAllocator<B2::LocalBasis> + EvalBasisAllocator<Prod<T, B1::LocalBasis, B2::LocalBasis>>,
{
    type Cell = (B1::Cell, B2::Cell); // todo: possibly change to Prod<..,..>
    type LocalBasis = Prod<T, B1::LocalBasis, B2::LocalBasis>;
    type GlobalIndices = impl Iterator<Item = usize> + Clone;

    fn local_basis(&self, elem: &Self::Cell) -> Self::LocalBasis {
        let (b1, b2) = &self.bases;
        Prod::new((b1.local_basis(&elem.0), b2.local_basis(&elem.1)))
    }

    fn global_indices(&self, elem: &Self::Cell) -> Self::GlobalIndices {
        // todo: implement this!
        once(0)
    }
}

impl <T, B1, B2> FindElem<T> for Prod<T, B1, B2>
where T: RealField,
      B1: FindElem<T, NumComponents = U1> + NumBasisAdd<B2>,
      B2: FindElem<T, NumComponents = U1, ParametricDim = B1::ParametricDim, Coord<T> = B1::Coord<T>>,
      B1::LocalBasis: NumBasisAdd<B2::LocalBasis>,
      Self::Coord<T>: Copy,
      DefaultAllocator: EvalBasisAllocator<B1::LocalBasis> + EvalBasisAllocator<B2::LocalBasis> + EvalBasisAllocator<Prod<T, B1::LocalBasis, B2::LocalBasis>>,

{
    fn find_elem(&self, x: Self::Coord<T>) -> Self::Cell {
        let (b1, b2) = &self.bases;
        (b1.find_elem(x), b2.find_elem(x))
    }
}

// todo: implement TriProd as well
//
// /// Cartesian product of three scalar bases [`B1`], [`B2`] and [`B3`].
// #[derive(Debug, Clone, Copy)]
// pub struct TriProd<T: RealField, X, B1, B2, B3> {
//     /// First basis.
//     b1: B1,
//
//     /// Second basis.
//     b2: B2,
//
//     /// Third basis.
//     b3: B3,
//
//     _phantom_data: PhantomData<(T, X)>,
// }
//
// #[allow(clippy::toplevel_ref_arg)]
// impl <T, X, B1, B2, B3> TriProd<T, X, B1, B2, B3>
// where T: RealField,
//       X: Copy,
//       B1: BsplineBasis<T, X, 1>,
//       B2: BsplineBasis<T, X, 1>,
//       B3: BsplineBasis<T, X, 1>
// {
//     /// Constructs a new [`TriProd`] from the bases `b1`, `b2` and `b3`.
//     pub fn new(b1: B1, b2: B2, b3: B3) -> Self {
//         TriProd { b1, b2, b3, _phantom_data: Default::default() }
//     }
// }
//
// #[allow(clippy::toplevel_ref_arg)]
// impl<T, X, B1, B2, B3> BsplineBasis<T, X, 3> for TriProd<T, X, B1, B2, B3>
// where T: RealField,
//       X: Copy,
//       B1: BsplineBasis<T, X, 1>,
//       B2: BsplineBasis<T, X, 1>,
//       B3: BsplineBasis<T, X, 1>
// {
//     type NonzeroIndices = impl Iterator<Item=usize>;
//
//     fn num_basis(&self) -> usize {
//         self.b1.num_basis() + self.b2.num_basis() + self.b3.num_basis()
//     }
//
//     fn eval_nonzero(&self, x: X) -> (MatrixXx3<T>, Self::NonzeroIndices) {
//         let (b1, idx1) = self.b1.eval_nonzero(x);
//         let (b2, idx2) = self.b2.eval_nonzero(x);
//         let (b3, idx3) = self.b3.eval_nonzero(x);
//
//         let b = stack![b1, 0, 0;
//                                   0, b2, 0;
//                                   0, 0, b3];
//         let idx = idx1.chain(idx2).chain(idx3); // todo: chain and add offsets to idx2
//         (b, idx)
//     }
// }