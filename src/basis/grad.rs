use crate::basis::eval::{EvalBasis, EvalGrad, EvalGradAllocator};
use crate::basis::lin_combination::LinCombination;
use crate::basis::local::{LocalBasis, LocalGradBasis};
use crate::basis::space::Space;
use crate::basis::traits::Basis;
use crate::cells::geo::{Cell, CellAllocator, HasBasisCoord, HasDim};
use crate::diffgeo::chart::Chart;
use crate::mesh::traits::{Mesh, MeshTopology, VertexStorage};
use nalgebra::{ComplexField, Const, DefaultAllocator, OMatrix, RealField, U1};
use nalgebra::allocator::Allocator;
use crate::cells::traits::ToElement;

/// Gradient of basis functions `grad B = { grad b[i] : b[i] ∈ B }`.
pub struct GradBasis<B, const D: usize>(B);

impl<B: Basis<NumComponents = U1>, const D: usize> Basis for GradBasis<B, D> {
    type NumBasis = B::NumBasis;
    type NumComponents = Const<D>;
    type Coord<T> = B::Coord<T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.0.num_basis_generic()
    }
}

/// Implement [`EvalBasis`] if `B` implements [`EvalBasis`].
impl <T: RealField, B: EvalGrad<T, D>, const D: usize> EvalBasis<T> for GradBasis<B, D>
    where DefaultAllocator: EvalGradAllocator<B, D>
{
    fn eval(&self, x: Self::Coord<T>) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        self.0.eval_grad(x)
    }
}

/// Implement [`LocalBasis`] if `B` is also a local basis.
impl <T, B, const D: usize> LocalBasis<T> for GradBasis<B, D>
where T: RealField,
      B: LocalGradBasis<T, D>,
      DefaultAllocator: EvalGradAllocator<B::ElemBasis, D>
{
    type Elem = B::Elem;
    type ElemBasis = GradBasis<B::ElemBasis, D>;
    type GlobalIndices = B::GlobalIndices;

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        GradBasis(self.0.elem_basis(elem))
    }

    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices {
        self.0.global_indices(elem)
    }
}

/// Space of gradients of basis functions in `B`.
pub type GradSpace<T, B, const D: usize> = Space<T, GradBasis<B, D>, D>;

impl <T, B, const D: usize> Space<T, B, D>
where T: RealField,
      B: LocalGradBasis<T, D>,
      DefaultAllocator: EvalGradAllocator<B::ElemBasis, D>
{
    /// Returns the gradient of this space.
    pub fn grad(self) -> GradSpace<T, B, D> {
        let basis = self.basis;
        Space::new(GradBasis(basis))
    }
}

impl <'a, T, B, const D: usize> LinCombination<'a, T, B, D>
    where T: ComplexField,
          B: LocalGradBasis<T::RealField, D>,
          DefaultAllocator: EvalGradAllocator<B::ElemBasis, D>
{
    /// Returns the gradient of this linear combination in the space `grad_space`.
    pub fn grad(self, grad_space: &'a GradSpace<T::RealField, B, D>) -> LinCombination<'a, T, GradBasis<B, D>, D> {
        LinCombination::new(self.coeffs, grad_space).unwrap()
    }
}

/// [`GradBasis`] mapped to the physical domain of a single element.
pub struct GradBasisPullbackLocal<C, B, const D: usize> {
    chart: C,
    grad_basis: GradBasis<B, D>,
}

impl<C, B, const D: usize> Basis for GradBasisPullbackLocal<C, B, D>
where
    B: Basis<NumComponents = U1>,
{
    type NumBasis = B::NumBasis;
    type NumComponents = Const<D>;
    type Coord<T> = B::Coord<T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.grad_basis.num_basis_generic()
    }
}

impl<T, C, B, const D: usize> EvalBasis<T> for GradBasisPullbackLocal<C, B, D>
    where T: RealField,
          B: EvalGrad<T, D>,
          B::Coord<T>: Copy,
          C: Chart<T, Coord = B::Coord<T>, ParametricDim = Const<D>, GeometryDim = Const<D>>,
          DefaultAllocator: EvalGradAllocator<B, D>
{
    fn eval(&self, x: Self::Coord<T>) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let d_phi = self.chart.eval_diff(x);
        d_phi.transpose().try_inverse().unwrap() * self.grad_basis.eval(x)
    }
}

// todo: implementation is correct, but refactor all code below
//  should we differentiate between Basis in the parametric and physical domain in general?

/// [`GradBasis`] mapped to the physical domain by inverse pullbacks.
pub struct GradBasisPullback<'a, T, Basis, Coords, Cells, const D: usize> {
    pub msh: &'a Mesh<T, Coords, Cells>,
    pub grad_basis: GradBasis<Basis, D>
}

impl<'a, T, B, Coords, Cells, const D: usize> Basis for GradBasisPullback<'a, T, B, Coords, Cells, D>
where
    B: Basis<NumComponents = U1>,
{
    type NumBasis = B::NumBasis;
    type NumComponents = Const<D>;
    type Coord<_T> = B::Coord<_T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.grad_basis.num_basis_generic()
    }
}

impl <'a, T, B, Coords, Cells, const D: usize> LocalBasis<T> for GradBasisPullback<'a, T, B, Coords, Cells, D>
    where T: RealField,
          B: LocalGradBasis<T, D>,
          Coords: VertexStorage<T>,
          Cells: MeshTopology<Elem = B::Elem>,
          B::Coord<T>: Copy,
          B::Elem: ToElement<T, Coords::GeoDim>,
          <B::Elem as ToElement<T, Coords::GeoDim>>::Elem: HasBasisCoord<T, B> + HasDim<T, D>,
          DefaultAllocator: EvalGradAllocator<B::ElemBasis, D> + CellAllocator<T, <B::Elem as ToElement<T, Coords::GeoDim>>::Elem> + Allocator<Coords::GeoDim>
{
    type Elem = B::Elem;
    type ElemBasis = GradBasisPullbackLocal<<<B::Elem as ToElement<T, Coords::GeoDim>>::Elem as Cell<T>>::GeoMap, B::ElemBasis, D>;
    type GlobalIndices = B::GlobalIndices;

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        let parametric_basis = self.grad_basis.elem_basis(elem);
        let chart = self.msh.geo_elem(elem).geo_map();
        GradBasisPullbackLocal { chart, grad_basis: parametric_basis }
    }

    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices {
        self.grad_basis.global_indices(elem)
    }
}
