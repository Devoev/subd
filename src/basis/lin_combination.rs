use crate::basis::error::CoeffsSpaceDimError;
use crate::basis::eval::{EvalBasis, EvalBasisAllocator, EvalGrad, EvalGradAllocator};
use crate::basis::local::{FindElem, LocalBasis, LocalGradBasis};
use crate::basis::space::Space;
use crate::basis::traits::Basis;
use nalgebra::allocator::Allocator;
use nalgebra::{ComplexField, DVector, DefaultAllocator, Dim, Dyn, Matrix, OMatrix, OVector, SVector, Scalar, U1};

/// Linear combination of coefficients with basis functions.
#[derive(Clone, Debug)]
pub struct LinCombination<'a, T: ComplexField, B, const D: usize> {
    /// Coefficients vector.
    pub coeffs: DVector<T>,

    /// Space of basis functions.
    pub space: &'a Space<T::RealField, B, D>,
}

impl <'a, T: ComplexField, B: Basis, const D: usize> LinCombination<'a, T, B, D> {
    /// Constructs a new [`LinCombination`] from the given `coeffs` and `space`.
    /// 
    /// # Errors
    /// Will return an error if the number of rows of `coeffs`
    /// does not match the dimension of `space`.
    pub fn new(coeffs: DVector<T>, space: &'a Space<T::RealField, B, D>) -> Result<Self, CoeffsSpaceDimError> {
        if coeffs.nrows() != space.dim() {
            return Err(CoeffsSpaceDimError { num_coeffs: coeffs.nrows(), dim_space: space.dim() });
        }

        Ok(LinCombination { coeffs, space })
    }
}

// todo: possibly rename

/// Allocator for the vector of [`B::NumComponents`] evaluated function components.
pub trait EvalFunctionAllocator<B: Basis>: Allocator<B::NumComponents> {}

impl <B: Basis> EvalFunctionAllocator<B> for DefaultAllocator
    where DefaultAllocator: Allocator<B::NumComponents>{}

impl <'a, T, B, const D: usize> LinCombination<'a, T, B, D>
    where T: ComplexField,
          B: EvalBasis<T::RealField, NumBasis=Dyn>,
          DefaultAllocator: EvalBasisAllocator<B> + EvalFunctionAllocator<B>
{
    /// Evaluates this linear combination at the parametric point `x`,
    /// by calculating `c[0] * b[0](x) + ... + c[n] * b[n](x)`.
    pub fn eval(&self, x: B::Coord<T::RealField>) -> OVector<T, B::NumComponents> {
        let b= self.space.basis.eval(x);
        // todo: is there another way to calculate this without using .map ?
        //  Maybe add custom cast function?
        b.map(|bi| T::from_real(bi)) * &self.coeffs
    }
}

// todo: make this more generic, e.g. for control points?
//  and why is Allocator<U1/Dyn> really necessary? Seems to be for self.coeffs
/// Allocator for selection of [`B::NumBasis`] (local) coefficients
/// and [`Dyn`] (global) coefficients.
pub trait SelectCoeffsAllocator<B: Basis>: Allocator<B::NumBasis> + Allocator<Dyn> {}

impl <B: Basis> SelectCoeffsAllocator<B> for DefaultAllocator
    where DefaultAllocator: Allocator<Dyn>,
          DefaultAllocator: Allocator<B::NumBasis> {}

// todo: move elsewhere
/// Same as [`Matrix::select_rows`] but generic.
fn select_rows_generic<T: Scalar + Clone, R: Dim, C: Dim, I: Dim>(mat: &OMatrix<T, R, C>, idx: OVector<usize, I>) -> OMatrix<T, I, C>
    where DefaultAllocator: Allocator<R, C>,
          DefaultAllocator: Allocator<I>,
          DefaultAllocator: Allocator<I, C>
{
    let (_, ncols) = mat.shape_generic();
    let (nrows, _) = idx.shape_generic();
    let mat = mat.select_rows(idx.iter());
    Matrix::from_iterator_generic(nrows, ncols, mat.iter().cloned())
}

impl <'a, T, B, const D: usize> LinCombination<'a, T, B, D>
    where T: ComplexField,
          B: LocalBasis<T::RealField>,
          DefaultAllocator: EvalBasisAllocator<B::ElemBasis> + EvalFunctionAllocator<B> + SelectCoeffsAllocator<B::ElemBasis>
{
    /// Evaluates the linear combination on the given `elem` at the parametric point `x`.
    pub fn eval_on_elem(&self, elem: &B::Elem, x: B::Coord<T::RealField>) -> OVector<T, B::NumComponents> {
        let local_space = self.space.local_space(elem); // todo: this can be removed, once the output of 'idx' is changed to a Vector directly
        let (b, idx) = self.space.eval_on_elem_with_idx(elem, x);
        let idx = OVector::from_iterator_generic(local_space.basis.num_basis_generic(), U1, idx);
        let c = select_rows_generic(&self.coeffs, idx);
        b.map(|bi| T::from_real(bi)) * c
    }
}

impl <'a, T, B, const D: usize> LinCombination<'a, T, B, D>
where T: ComplexField,
      B: FindElem<T::RealField>,
      B::Coord<T::RealField>: Clone,
      DefaultAllocator: EvalBasisAllocator<B::ElemBasis> + EvalFunctionAllocator<B> + SelectCoeffsAllocator<B::ElemBasis>
{
    /// Evaluates the linear combination at the parametric point `x`.
    /// This is done by finding the local element in which `x` is, which can potentially be expensive.
    pub fn eval_local(&self, x: B::Coord<T::RealField>) -> OVector<T, B::NumComponents>{
        self.eval_on_elem(&self.space.basis.find_elem(x.clone()), x)
    }
}

impl <'a, T, B, const D: usize> LinCombination<'a, T, B, D>
where T: ComplexField,
      B: LocalGradBasis<T::RealField, D>,
      DefaultAllocator: EvalGradAllocator<B::ElemBasis, D> + SelectCoeffsAllocator<B::ElemBasis>
{
    /// Evaluates the gradient of the linear combination on the given `elem` at the parametric point `x`.
    pub fn eval_grad_on_elem(&self, elem: &B::Elem, x: B::Coord<T::RealField>) -> SVector<T, D> {
        let local_space = self.space.local_space(elem); // todo: this can be removed, once the output of 'idx' is changed to a Vector directly
        let (b, idx) = self.space.eval_grad_on_elem_with_idx(elem, x);
        let idx = OVector::from_iterator_generic(local_space.basis.num_basis_generic(), U1, idx);
        let c = select_rows_generic(&self.coeffs, idx);
        b.map(|bi| T::from_real(bi)) * c
    }
}

impl <'a, T, B, const D: usize> LinCombination<'a, T, B, D>
where T: ComplexField,
      B: FindElem<T::RealField, NumComponents = U1>,
      B::ElemBasis: EvalGrad<T::RealField, D>,
      B::Coord<T::RealField>: Clone,
      DefaultAllocator: EvalGradAllocator<B::ElemBasis, D> + SelectCoeffsAllocator<B::ElemBasis>
{
    /// Evaluates the gradient of the linear combination at the parametric point `x`.
    pub fn eval_grad_local(&self, x: B::Coord<T::RealField>) -> SVector<T, D> {
        self.eval_grad_on_elem(&self.space.basis.find_elem(x.clone()), x)
    }
}