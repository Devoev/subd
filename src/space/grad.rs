use crate::space::eval_basis::{EvalBasis, EvalGrad, EvalGradAllocator};
use crate::space::lin_combination::LinCombination;
use crate::space::local::{MeshBasis, MeshElemBasis, MeshGradBasis};
use crate::space::basis::BasisFunctions;
use crate::cells::traits::ToElement;
use crate::diffgeo::chart::{Chart, ChartAllocator};
use crate::element::traits::{ElemAllocator, Element, HasBasisCoord, HasDim};
use crate::mesh::cell_topology::{ElementTopology, VolumetricElementTopology};
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{ElemOfMesh, Mesh, MeshAllocator};
use nalgebra::allocator::Allocator;
use nalgebra::{ComplexField, Const, DefaultAllocator, OMatrix, RealField, U1};
use crate::space::Space;

/// Gradient of basis functions `grad B = { grad b[i] : b[i] ∈ B }`.
pub struct GradBasis<Basis>(Basis);

impl<Basis: BasisFunctions<NumComponents = U1>> BasisFunctions for GradBasis<Basis> {
    type NumBasis = Basis::NumBasis;
    type NumComponents = Basis::ParametricDim;
    type ParametricDim = Basis::ParametricDim;
    type Coord<T> = Basis::Coord<T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.0.num_basis_generic()
    }
}

/// Implement [`EvalBasis`] if `B` implements [`EvalBasis`].
impl <T: RealField, Basis: EvalGrad<T>> EvalBasis<T> for GradBasis<Basis>
    where DefaultAllocator: EvalGradAllocator<Basis>
{
    fn eval(&self, x: Self::Coord<T>) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        self.0.eval_grad(x)
    }
}

/// Implement [`MeshBasis`] if `B` is also a local basis.
impl <T, Basis> MeshBasis<T> for GradBasis<Basis>
where T: RealField,
      Basis: MeshGradBasis<T>,
      DefaultAllocator: EvalGradAllocator<Basis::LocalBasis>
{
    type Cell = Basis::Cell;
    type LocalBasis = GradBasis<Basis::LocalBasis>;
    type GlobalIndices = Basis::GlobalIndices;

    fn local_basis(&self, elem: &Self::Cell) -> Self::LocalBasis {
        GradBasis(self.0.local_basis(elem))
    }

    fn global_indices(&self, elem: &Self::Cell) -> Self::GlobalIndices {
        self.0.global_indices(elem)
    }
}

/// Space of gradients of basis functions in `B`.
pub type GradSpace<T, Basis> = Space<T, GradBasis<Basis>>;

impl <T, Basis> Space<T, Basis>
where T: RealField,
      Basis: MeshGradBasis<T>,
      DefaultAllocator: EvalGradAllocator<Basis::LocalBasis>
{
    /// Returns the gradient of this space.
    pub fn grad(self) -> GradSpace<T, Basis> {
        let basis = self.basis;
        Space::new(GradBasis(basis))
    }
}

impl <'a, T, Basis> LinCombination<'a, T, Basis>
    where T: ComplexField,
          Basis: MeshGradBasis<T::RealField>,
          DefaultAllocator: EvalGradAllocator<Basis::LocalBasis>
{
    /// Returns the gradient of this linear combination in the space `grad_space`.
    pub fn grad(self, grad_space: &'a GradSpace<T::RealField, Basis>) -> LinCombination<'a, T, GradBasis<Basis>> {
        LinCombination::new(self.coeffs, grad_space).unwrap()
    }
}

/// [`GradBasis`] mapped to the physical domain of a single element.
pub struct GradBasisPullbackLocal<C, Basis> {
    chart: C,
    grad_basis: GradBasis<Basis>,
}

impl<C, Basis> BasisFunctions for GradBasisPullbackLocal<C, Basis>
where
    Basis: BasisFunctions<NumComponents = U1>,
{
    type NumBasis = Basis::NumBasis;
    type NumComponents = Basis::ParametricDim;
    type ParametricDim = Basis::ParametricDim;
    type Coord<T> = Basis::Coord<T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.grad_basis.num_basis_generic()
    }
}

impl<T, GeoMap, Basis> EvalBasis<T> for GradBasisPullbackLocal<GeoMap, Basis>
    where T: RealField,
          Basis: EvalGrad<T>,
          Basis::Coord<T>: Copy,
          GeoMap: Chart<T, Coord = Basis::Coord<T>, ParametricDim = Basis::ParametricDim, GeometryDim = Basis::ParametricDim>,
          DefaultAllocator: EvalGradAllocator<Basis> + ChartAllocator<T, GeoMap>
{
    fn eval(&self, x: Self::Coord<T>) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let d_phi = self.chart.eval_diff(x);
        d_phi.transpose().try_inverse().unwrap() * self.grad_basis.eval(x)
    }
}

// todo: implementation is correct, but refactor all code below
//  should we differentiate between Basis in the parametric and physical domain in general?

/// [`GradBasis`] mapped to the physical domain by inverse pullbacks.
pub struct GradBasisPullback<'a, T, Basis, Coords, Cells> {
    pub msh: &'a Mesh<T, Coords, Cells>,
    pub grad_basis: GradBasis<Basis>
}

impl<'a, T, Basis, Coords, Cells> BasisFunctions for GradBasisPullback<'a, T, Basis, Coords, Cells>
where
    Basis: BasisFunctions<NumComponents = U1>,
{
    type NumBasis = Basis::NumBasis;
    type NumComponents = Basis::ParametricDim;
    type ParametricDim = Basis::ParametricDim;
    type Coord<_T> = Basis::Coord<_T>;

    fn num_basis_generic(&self) -> Self::NumBasis {
        self.grad_basis.num_basis_generic()
    }
}

impl <'a, T, Basis, Verts, Cells> MeshBasis<T> for GradBasisPullback<'a, T, Basis, Verts, Cells>
    where T: RealField,
          Verts: VertexStorage<T>,
          Cells: VolumetricElementTopology<T, Verts>,
          Basis: MeshGradBasis<T> + MeshElemBasis<T, Verts, Cells>,
          Basis::Coord<T>: Copy,
          DefaultAllocator: EvalGradAllocator<Basis::LocalBasis> + MeshAllocator<T, Verts, Cells>
{
    type Cell = Basis::Cell;
    type LocalBasis = GradBasisPullbackLocal<<ElemOfMesh<T, Verts, Cells> as Element<T>>::GeoMap, Basis::LocalBasis>;
    type GlobalIndices = Basis::GlobalIndices;

    fn local_basis(&self, cell: &Self::Cell) -> Self::LocalBasis {
        let parametric_basis = self.grad_basis.local_basis(cell);
        let chart = cell.to_element(&self.msh.coords).geo_map();
        GradBasisPullbackLocal { chart, grad_basis: parametric_basis }
    }

    fn global_indices(&self, elem: &Self::Cell) -> Self::GlobalIndices {
        self.grad_basis.global_indices(elem)
    }
}
