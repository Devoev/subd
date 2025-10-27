use crate::cells::node::Node;
use crate::cells::traits::{CellBoundary, CellConnectivity, OrderedCell, OrientedCell};
use crate::mesh::elem_vertex::ElemVertexMesh;
use itertools::iproduct;
use nalgebra::{DMatrix, DVector, Point, Scalar, U2};
use nalgebra_sparse::CsrMatrix;
use num_traits::Zero;
use std::collections::BTreeSet;
use std::hash::Hash;

/// Homogeneous Dirichlet boundary conditions on nodes.
pub struct DirichletBc<T> {
    /// The number of total nodes.
    num_nodes: usize,

    /// Degrees-of-freedom indices.
    idx_dof: BTreeSet<usize>,

    // todo: actually use u_bc in inflate and deflate functions
    /// Vector of solution coefficients on the boundary.
    u_bc: DVector<T>
}

impl <T: Scalar + Zero> DirichletBc<T> {
    /// Constructs a new [`DirichletBc`] from the given set of boundary nodes `idx_bc`
    /// and solution coefficients at the boundary `u_bc`.
    pub fn new(num_nodes: usize, idx_bc: BTreeSet<Node>, u_bc: DVector<T>) -> Self {
        let idx: BTreeSet<Node> = (0..num_nodes).collect();
        let idx_dof: BTreeSet<Node> = idx.difference(&idx_bc).copied().collect();
        DirichletBc { num_nodes, idx_dof, u_bc }
    }

    /// Constructs a new [`DirichletBc`] with *homogeneous* (= zero) boundary data.
    pub fn new_homogeneous(num_nodes: usize, idx_bc: BTreeSet<Node>) -> Self {
        let u_bc = DVector::zeros(idx_bc.len());
        DirichletBc::new(num_nodes, idx_bc, u_bc)
    }

    /// Constructs a new [`DirichletBc`] on the given elem-to-vertex `msh`,
    /// where `idx_dof` are all interior nodes.
    pub fn new_on_mesh<F, const M: usize>(msh: &ElemVertexMesh<T, F, M>, u_bc: DVector<T>) -> Self
    where F: CellBoundary<Dim = U2, Node = Node>,
          F::Node: Eq + Hash,
          F::SubCell: OrderedCell + OrientedCell + CellConnectivity + Clone + Eq + Hash
    {
        let num_nodes = msh.num_nodes();
        let idx = (0..num_nodes).collect::<BTreeSet<Node>>();
        let idx_bc = msh.boundary_nodes().collect::<BTreeSet<_>>();
        let idx_dof = idx.difference(&idx_bc).copied().collect::<BTreeSet<_>>();
        DirichletBc::new(num_nodes, idx_dof, u_bc)
    }

    /// Constructs a new [`DirichletBc`] with *homogeneous* boundary data on the given `msh`.
    pub fn new_homogeneous_on_mesh<F, const M: usize>(msh: &ElemVertexMesh<T, F, M>) -> Self
    where F: CellBoundary<Dim = U2, Node = Node>,
          F::Node: Eq + Hash,
          F::SubCell: OrderedCell + OrientedCell + CellConnectivity + Clone + Eq + Hash
    {
        let num_nodes = msh.num_nodes();
        let idx = (0..num_nodes).collect::<BTreeSet<Node>>();
        let idx_bc = msh.boundary_nodes().collect::<BTreeSet<_>>();
        let idx_dof = idx.difference(&idx_bc).copied().collect::<BTreeSet<_>>();
        DirichletBc::new_homogeneous(num_nodes, idx_dof)
    }

    /// Constructs a new [`DirichletBc`] on the given `msh`.
    ///
    /// The boundary data is given by the function `g`
    /// and the coefficients of the solution are calculated by evaluation of `g` at the boundary nodes.
    /// This only holds true for **interpolating** nodal basis function.
    pub fn new_interpolating<F: CellBoundary<Dim = U2, Node = Node>, const M: usize>(msh: &ElemVertexMesh<T, F, M>, idx_bc: BTreeSet<Node>, g: impl Fn(&Point<T,M>) -> T) -> Self
    where F::Node: Eq + Hash,
          F::SubCell: OrderedCell + OrientedCell + CellConnectivity + Clone + Eq + Hash
    {
        // Find bc and dof indices
        let num_nodes = msh.num_nodes();
        let idx = (0..num_nodes).collect::<BTreeSet<Node>>();
        let idx_bc = msh.boundary_nodes().collect::<BTreeSet<_>>();
        let idx_dof = idx.difference(&idx_bc).copied().collect::<BTreeSet<_>>();

        // Evaluate boundary function `g` at boundary nodes
        let u_bc = DVector::from_iterator(
            idx_bc.len(),
            idx_bc.iter().map(|&node| g(&msh.coords[node]))
        );

        DirichletBc::new(num_nodes, idx_dof, u_bc)
    }
    
    /// Returns the number of DOFs.
    pub fn num_dof(&self) -> usize {
        self.idx_dof.len()
    }
}

impl <T: Scalar + Zero + Copy> DirichletBc<T> {
    /// Deflates the system `ax = b` by removing rows and columns of BC indices.
    /// Returns the reduced system matrix and right hand side vector.
    pub fn deflate(&self, a: CsrMatrix<T>, b: DVector<T>) -> (CsrMatrix<T>, DVector<T>) {
        let num_dof = self.idx_dof.len();
        let b_dof = DVector::from_iterator(num_dof, self.idx_dof.iter().map(|&i| b[i]));

        // todo: this is VERY inefficient and expensive because
        //  - The call to `get_entry` requires a binary search
        //  - A dense matrix is created, not a sparse one
        let a_dof_dof = DMatrix::from_iterator(num_dof, num_dof, iproduct!(self.idx_dof.iter(), self.idx_dof.iter())
            .map(|(&i, &j)| a.get_entry(i, j).unwrap().into_value())
        );

        // todo: possibly implement like this? but this doesnt work...
        // let a_dof_dof = a.filter(|i, j, _| self.idx_dof.contains(&i) && self.idx_dof.contains(&j));

        (CsrMatrix::from(&a_dof_dof), b_dof)
    }

    /// Inflates a vector `u` by inserting the values from `u_dof` at the DOF indices.
    pub fn inflate(&self, u_dof: DVector<T>) -> DVector<T> {
        let mut u = DVector::zeros(self.num_nodes);
        for (i_local, &i) in self.idx_dof.iter().enumerate() {
            u[i] = u_dof[i_local];
        }
        u
    }
}