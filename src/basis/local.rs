use crate::basis::eval::{EvalBasis, EvalBasisAllocator, EvalGrad, EvalGradAllocator};
use crate::basis::traits::Basis;
use crate::cells::traits::{ElemOfCell, ToElement};
use crate::element::traits::{ElemAllocator, HasBasisCoord};
use crate::mesh::cell_topology::ElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dyn, RealField, Scalar, U1};
// todo: NumBasis from basis super-trait is never used. Can this be removed?

/// Basis functions defined on a mesh.
/// 
/// Each function has local support on a [cell](Self::Cell) and can thus be evaluated
/// by restricting the basis to a local one. 
/// The local basis on a given `cell` can be obtained using the method [`Self::local_basis`].
pub trait MeshBasis<T: Scalar>: Basis<NumBasis = Dyn>
    where DefaultAllocator: EvalBasisAllocator<Self::LocalBasis>
{
    /// Local cell in a mesh.
    type Cell;
    
    /// Restriction of the local basis on a cell.
    type LocalBasis: EvalBasis<T, NumComponents = Self::NumComponents, Coord<T> = Self::Coord<T>>;

    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Iterator over linear global indices.
    type GlobalIndices: Iterator<Item = usize> + Clone;

    /// Returns the [`Self::LocalBasis`] for the given `cell`,
    /// i.e. the restriction of this basis to the cell.
    fn local_basis(&self, cell: &Self::Cell) -> Self::LocalBasis;

    /// Returns an iterator over all global indices of the local basis of `cell`.
    fn global_indices(&self, cell: &Self::Cell) -> Self::GlobalIndices;
}

// todo: from where should the dimension D for geometry and parametric domain come from?
/// Basis on a mesh where each cell implements [`ToElement`].
///
/// The cells are required to match the [`Cells::Cell`] of the element topology `Cells`.
/// For compatibility with the basis functions, the elements must match the [`Basis::Coord<T>`]
/// and the geometric and parametric dimensions.
pub trait MeshElemBasis<T, Verts, Cells>: MeshBasis<T, Cell = Cells::Cell> + Sized
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: ElementTopology<T, Verts>,
      ElemOfCell<T, Cells::Cell, Verts::GeoDim>: HasBasisCoord<T, Self>,
      DefaultAllocator: Allocator<Verts::GeoDim> + EvalBasisAllocator<Self::LocalBasis> + ElemAllocator<T, ElemOfCell<T, Cells::Cell, Verts::GeoDim>> {}

/// Local basis functions with [gradient evaluations](EvalGrad).
pub trait MeshGradBasis<T: RealField, const D: usize>: MeshBasis<T, LocalBasis: EvalGrad<T, D>, NumComponents = U1>
    where DefaultAllocator: EvalGradAllocator<Self::LocalBasis, D> {}

impl <T: RealField, const D: usize, B> MeshGradBasis<T, D> for B
where B: MeshBasis<T, LocalBasis: EvalGrad<T, D>, NumComponents = U1>,
      DefaultAllocator: EvalGradAllocator<Self::LocalBasis, D>
{}

/// Local basis functions that can find the local element by parametric value.
pub trait FindElem<T: Scalar>: MeshBasis<T>
    where DefaultAllocator: EvalBasisAllocator<Self::LocalBasis>
{
    // todo: possibly change to Result<Self::Elem, ...>
    /// Finds the [`Self::Cell`] containing the given parametric value `x`.
    fn find_elem(&self, x: Self::Coord<T>) -> Self::Cell;
}