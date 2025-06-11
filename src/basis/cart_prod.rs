use crate::basis::eval::EvalBasis;
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use nalgebra::allocator::Allocator;
use nalgebra::{stack, DefaultAllocator, DimAdd, DimSum, OMatrix, RealField, U1, U2};
use std::iter::once;
use std::marker::PhantomData;

// todo: add proper docs

/// Cartesian product of two scalar bases [`B1`] and [`B2`].
#[derive(Debug, Clone, Copy)]
pub struct Prod<T, X, B1, B2> {
    /// Pair of first and second basis.
    bases: (B1, B2),

    _phantom_data: PhantomData<(T, X)>,
}


impl <T, X, B1, B2> Prod<T, X, B1, B2> {
    /// Constructs a new [`Prod`] from the bases `b1` and `b2`.
    pub fn new(bases: (B1, B2)) -> Self {
        Prod { bases, _phantom_data: Default::default() }
    }
}

impl<T, X, B1, B2> Basis for Prod<T, X, B1, B2>
where B1: Basis<NumComponents = U1>,
      B2: Basis<NumComponents = U1>,
      B1::NumBasis: DimAdd<B2::NumBasis>
{
    type NumBasis = DimSum<B1::NumBasis, B2::NumBasis>;
    type NumComponents = U2;

    fn num_basis(&self) -> usize {
        self.bases.0.num_basis() + self.bases.1.num_basis()
    }

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.bases.0.num_basis_generic().add(self.bases.1.num_basis_generic())
    }

    fn num_components(&self) -> usize {
        2
    }

    fn num_components_generic(&self) -> Self::NumComponents {
        U2
    }
}

#[allow(clippy::toplevel_ref_arg)]
impl <T, X, B1, B2> EvalBasis<T, X> for Prod<T, X, B1, B2>
where T: RealField,
      X: Copy,
      B1: EvalBasis<T, X, NumComponents = U1>,
      B2: EvalBasis<T, X, NumComponents = U1>,
      B1::NumBasis: DimAdd<B2::NumBasis>,
      DefaultAllocator: Allocator<B1::NumComponents, B1::NumBasis>,
      DefaultAllocator: Allocator<B2::NumComponents, B2::NumBasis>,
      DefaultAllocator: Allocator<Self::NumComponents, Self::NumBasis>,
{
    fn eval(&self, x: X) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let b1 = self.bases.0.eval(x);
        let b2= self.bases.1.eval(x);
        stack![
            b1, 0;
            0, b2
        ]
    }
}

impl <T, X, B1, B2> LocalBasis<T, X> for Prod<T, X, B1, B2>
where T: RealField,
      X: Copy,
      B1: LocalBasis<T, X, NumComponents = U1>,
      B2: LocalBasis<T, X, NumComponents = U1>,
      B1::NumBasis: DimAdd<B2::NumBasis>,
      B1::ElemBasis: EvalBasis<T, X, NumComponents = U1>,
      B2::ElemBasis: EvalBasis<T, X, NumComponents = U1>,
      <B1::ElemBasis as Basis>::NumBasis: DimAdd<<B2::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<B1::NumComponents, <B1::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<B2::NumComponents, <B2::ElemBasis as Basis>::NumBasis>,
      DefaultAllocator: Allocator<nalgebra::Const<2>, <<B1::ElemBasis as Basis>::NumBasis as DimAdd<<B2::ElemBasis as Basis>::NumBasis>>::Output>
{
    type Elem = (B1::Elem, B2::Elem); // todo: possibly change to Prod<..,..>
    type ElemBasis = Prod<T, X, B1::ElemBasis, B2::ElemBasis>;
    type GlobalIndices = impl Iterator<Item = usize> + Clone;

    fn find_elem(&self, x: X) -> Self::Elem {
        let (b1, b2) = &self.bases;
        (b1.find_elem(x), b2.find_elem(x))
    }

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        let (b1, b2) = &self.bases;
        Prod::new((b1.elem_basis(&elem.0), b2.elem_basis(&elem.1)))
    }

    fn global_indices(&self, local_basis: &Self::ElemBasis) -> Self::GlobalIndices {
        // todo: implement this
        once(0)
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