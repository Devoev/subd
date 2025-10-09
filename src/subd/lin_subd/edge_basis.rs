use crate::space::eval_basis::{EvalBasis, EvalScalarCurl};
use crate::space::local::MeshBasis;
use crate::space::basis::BasisFunctions;
use crate::cells::edge::UndirectedEdge;
use crate::cells::quad::QuadNodes;
use crate::mesh::face_vertex::QuadVertexMesh;
use nalgebra::{matrix, one, zero, Dyn, OMatrix, RealField, RowOVector, RowVector4, U2, U4};

/// Lowest-order edge basis functions on a quadrilateral mesh.
#[derive(Debug, Clone)]
pub struct WhitneyEdgeQuad<'a, T: RealField, const M: usize> {
    /// Quadrilateral mesh.
    pub msh: &'a QuadVertexMesh<T, M>, // todo: is this field really not needed?

    /// Undirected edges of the mesh.
    pub edges: Vec<UndirectedEdge> // todo: possibly move this to the mesh itself
}

impl <'a, T: RealField, const M: usize> WhitneyEdgeQuad<'a, T, M>  {
    /// Constructs a new [`WhitneyEdgeQuad`] on the given `msh`.
    pub fn new(msh: &'a QuadVertexMesh<T, M>) -> Self {
        WhitneyEdgeQuad { msh, edges: msh.edges().map(|edge| edge.into()).collect() }
    }
}

impl<'a, T: RealField, const M: usize> BasisFunctions for WhitneyEdgeQuad<'a, T, M> {
    type NumBasis = Dyn;
    type NumComponents = U2;
    type ParametricDim = U2;
    type Coord<_T> = (_T, _T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.edges.len())
    }
}

impl <'a, T: RealField + Copy, const M: usize> MeshBasis<T> for WhitneyEdgeQuad<'a, T, M> {
    type Cell = QuadNodes;
    type LocalBasis = WhitneyEdgeQuadLocal;
    type GlobalIndices = impl Iterator<Item = usize> + Clone;

    fn local_basis(&self, _elem: &Self::Cell) -> Self::LocalBasis {
        WhitneyEdgeQuadLocal
    }

    fn global_indices(&self, elem: &Self::Cell) -> Self::GlobalIndices {
        let local_edges = elem.undirected_edges();

        let mut edge_idx = [0usize; 4];
        for (i, &edge) in self.edges.iter().enumerate() {
            if local_edges[0] == edge { edge_idx[0] = i }
            if local_edges[1] == edge { edge_idx[1] = i }
            if local_edges[2] == edge { edge_idx[2] = i }
            if local_edges[3] == edge { edge_idx[3] = i }
        }
        edge_idx.into_iter()
    }
}

/// Lowest order edge basis functions on a quadrilateral.
#[derive(Copy, Clone, Debug)]
pub struct WhitneyEdgeQuadLocal;

impl BasisFunctions for WhitneyEdgeQuadLocal {
    type NumBasis = U4;
    type NumComponents = U2;
    type ParametricDim = U2;
    type Coord<T> = (T, T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        U4
    }
}

impl <T: RealField + Copy> EvalBasis<T> for WhitneyEdgeQuadLocal {
    fn eval(&self, (u, v): (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        // e12 = [1 - y; 0];
        // e23 = [0; x];
        // e34 = [y; 0];
        // e41 = [0; 1 - x];
        matrix![
            -v + one(), zero(), v,     zero();    // u-components
            zero(),     u,      zero(), -u + one(); // v-components
        ]
    }
}

impl <T: RealField + Copy> EvalScalarCurl<T> for WhitneyEdgeQuadLocal {
    fn eval_scalar_curl(&self, _x: (T, T)) -> RowOVector<T, Self::NumBasis> {
        RowVector4::from_element(one())
    }
}