use crate::bspline::space::BsplineSpace;
use crate::cells::node::Node;
use crate::cells::traits::{CellBoundary, CellConnectivity, OrderedCell, OrientedCell};
use crate::mesh::elem_vertex::ElemVertexMesh;
use itertools::iproduct;
use nalgebra::{DMatrix, DVector, RealField, Scalar, U2};
use nalgebra_sparse::CsrMatrix;
use num_traits::Zero;
use std::collections::{BTreeSet, HashSet};
use std::hash::Hash;

/// Homogeneous Dirichlet boundary conditions on nodes.
pub struct DirichletBcHom {
    /// The number of total nodes.
    num_nodes: usize,

    /// Degrees-of-freedom indices.
    idx_dof: BTreeSet<usize>,
}

impl DirichletBcHom {
    /// Constructs a new `DirichletBcHom` from the given set of boundary nodes `idx_bc`.
    pub fn new(num_nodes: usize, idx_bc: BTreeSet<Node>) -> Self {
        let idx: BTreeSet<Node> = (0..num_nodes).collect();
        let idx_dof: BTreeSet<Node> = idx.difference(&idx_bc).copied().collect();
        DirichletBcHom { num_nodes, idx_dof }
    }
    
    
    /// Constructs a new `DirichletBcHom` from the given elem-to-vertex `msh`,
    /// where `idx_dof` are all interior nodes.
    pub fn new_on_mesh<T: RealField, F: CellBoundary<Dim = U2, Node = Node>, const M: usize>(msh: &ElemVertexMesh<T, F, M>) -> Self
    where F::Node: Eq + Hash,
          F::SubCell: OrderedCell + OrientedCell + CellConnectivity + Clone + Eq + Hash
    {
        let num_nodes = msh.num_nodes();
        let idx = (0..num_nodes).collect::<BTreeSet<Node>>();
        let idx_bc = msh.boundary_nodes().collect::<BTreeSet<_>>();
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
    
    /// Returns the number of DOFs.
    pub fn num_dof(&self) -> usize {
        self.idx_dof.len()
    }
}