use std::collections::BTreeSet;
use itertools::iproduct;
use nalgebra::{Const, DMatrix, DVector, DimNameDiff, DimNameSub, RealField, Scalar, U1};
use nalgebra_sparse::CsrMatrix;
use num_traits::Zero;
use crate::cells::topo::CellBoundary;
use crate::mesh::elem_vertex::ElemVertexMesh;
use crate::mesh::traits::MeshTopology;

/// Homogeneous Dirichlet boundary conditions on nodes.
pub struct DirichletBcHom {
    /// The number of total nodes.
    num_nodes: usize,

    /// Degrees-of-freedom indices.
    idx_dof: BTreeSet<usize>,
}

impl DirichletBcHom {
    /// Constructs a new [`DirichletBcHom`] from the given elem-to-vertex `msh`,
    /// where `idx_dof` are all interior nodes.
    pub fn from_mesh<T: RealField, C: CellBoundary<Const<K>>, const K: usize, const M: usize>(msh: &ElemVertexMesh<T, C, K, M>) -> Self
        where Const<K>: DimNameSub<U1> + DimNameSub<DimNameDiff<Const<K>, U1>>
    {
        let num_nodes = msh.num_nodes();
        let idx = (0..num_nodes).collect::<BTreeSet<_>>();
        let idx_bc = msh.boundary_nodes().map(|n| n.0).collect::<BTreeSet<_>>();
        let idx_dof = idx.difference(&idx_bc).copied().collect::<BTreeSet<_>>();
        DirichletBcHom { num_nodes, idx_dof }
    }

    /// Deflates the system `ax = b` by removing rows and columns of BC indices.
    /// Returns the reduced system matrix and right hand side vector.
    pub fn deflate<T: Scalar + Zero>(&self, a: CsrMatrix<T>, b: DVector<T>) -> (CsrMatrix<T>, DVector<T>) {
        let num_dof = self.idx_dof.len();
        let b_dof = DVector::from_iterator(num_dof, self.idx_dof.iter().map(|&i| b[i].clone()));

        // todo: this is VERY inefficient and expensive because
        //  - The call to `get_entry` requires a binary search
        //  - A dense matrix is created, not a sparse one
        let a_dof_dof = DMatrix::from_iterator(num_dof, num_dof, iproduct!(self.idx_dof.iter(), self.idx_dof.iter())
            .map(|(&i, &j)| a.get_entry(i, j).unwrap().into_value())
        );

        (CsrMatrix::from(&a_dof_dof), b_dof)
    }

    /// Inflates a vector `u` by inserting the values from `u_dof` at the DOF indices.
    pub fn inflate<T: Scalar + Zero + Copy>(&self, u_dof: DVector<T>) -> DVector<T> {
        let mut u = DVector::zeros(self.num_nodes);
        for (i_local, &i) in self.idx_dof.iter().enumerate() {
            u[i] = u_dof[i_local];
        }
        u
    }
}