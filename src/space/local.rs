use crate::cells::traits::ElemOfCell;
use crate::element::traits::{ElemAllocator, ElemCoord, Element};
use crate::mesh::cell_topology::ElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::MeshAllocator;
use crate::space::basis::BasisFunctions;
use crate::space::eval_basis::{EvalBasis, EvalBasisAllocator, EvalGrad, EvalGradAllocator};
use nalgebra::{DefaultAllocator, Dyn, RealField, Scalar, U1};
// todo: NumBasis from basis super-trait is never used. Can this be removed?

/// Basis functions defined on a mesh.
/// 
/// Each function has local support on a [cell](Self::Cell) and can thus be evaluated
/// by restricting the basis to a local one. 
/// The local basis on a given `cell` can be obtained using the method [`Self::local_basis`].
pub trait MeshBasis<T: Scalar>: BasisFunctions<NumBasis = Dyn>
    where DefaultAllocator: EvalBasisAllocator<Self::LocalBasis>
{
    /// Local cell in a mesh.
    type Cell;
    
    /// Restriction of the local basis on a cell.
    type LocalBasis: EvalBasis<T, NumComponents = Self::NumComponents, ParametricDim = Self::ParametricDim, Coord<T> = Self::Coord<T>>;

    // todo: possibly change to IntoIterator or separate trait/ struct all together
    /// Iterator over linear global indices.
    type GlobalIndices: Iterator<Item = usize> + Clone;

    /// Returns the [`Self::LocalBasis`] for the given `cell`,
    /// i.e. the restriction of this basis to the cell.
    fn local_basis(&self, cell: &Self::Cell) -> Self::LocalBasis;

    /// Returns an iterator over all global indices of the local basis of `cell`.
    fn global_indices(&self, cell: &Self::Cell) -> Self::GlobalIndices;
}

/// Basis on a mesh where each cell belongs to an [`ElementTopology`].
///
/// The cells are required to match the [`Cells::Cell`] of the element topology `Cells`.
/// For compatibility with the basis functions, the elements must match the [`BasisFunctions::Coord<T>`]
/// and the geometric dimension must equal [`Verts::GeoDim`].
pub trait MeshElemBasis<T, Verts, Cells>: MeshBasis<
    T,
    Cell = Cells::Cell,
    Coord<T> = ElemCoord<T, ElemOfCell<T, Cells::Cell, Verts::GeoDim>>
> + Sized
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: ElementTopology<T, Verts>,
      DefaultAllocator: EvalBasisAllocator<Self::LocalBasis> + MeshAllocator<T, Verts, Cells> {}

impl <T, Verts, Cells, Basis> MeshElemBasis<T, Verts, Cells> for Basis
where T: Scalar,
      Verts: VertexStorage<T>,
      Cells: ElementTopology<T, Verts>,
      Basis: MeshBasis<
          T,
          Cell = Cells::Cell,
          Coord<T> = ElemCoord<T, ElemOfCell<T, Cells::Cell, Verts::GeoDim>>
      > + Sized,
      DefaultAllocator: EvalBasisAllocator<Self::LocalBasis> + MeshAllocator<T, Verts, Cells> {}

/// Local basis on a single element.
///
/// The [`BasisFunctions::Coord<T>`] must match the coordinates of the `Elem`.
pub trait ElemBasis<T: Scalar, Elem: Element<T>>: BasisFunctions<Coord<T> = ElemCoord<T, Elem>>
    where DefaultAllocator: ElemAllocator<T, Elem> {}

impl <T, Elem, Basis> ElemBasis<T, Elem> for Basis
where T: Scalar,
      Elem: Element<T>,
      Basis: BasisFunctions<Coord<T> = ElemCoord<T, Elem>>,
      DefaultAllocator: ElemAllocator<T, Elem> {}

/// Local basis functions with [gradient evaluations](EvalGrad).
pub trait MeshGradBasis<T: RealField>: MeshBasis<T, LocalBasis: EvalGrad<T>, NumComponents = U1>
    where DefaultAllocator: EvalGradAllocator<Self::LocalBasis> {}

impl <T: RealField, B> MeshGradBasis<T> for B
where B: MeshBasis<T, LocalBasis: EvalGrad<T>, NumComponents = U1>,
      DefaultAllocator: EvalGradAllocator<Self::LocalBasis>
{}

/// Local basis functions that can find the local element by parametric value.
pub trait FindElem<T: Scalar>: MeshBasis<T>
    where DefaultAllocator: EvalBasisAllocator<Self::LocalBasis>
{
    // todo: possibly change to Result<Self::Elem, ...>
    /// Finds the [`Self::Cell`] containing the given parametric value `x`.
    fn find_elem(&self, x: Self::Coord<T>) -> Self::Cell;
}