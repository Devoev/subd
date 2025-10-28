use crate::cells::node::Node;
use crate::cells::traits::{CellBoundary, CellConnectivity, OrderedCell, OrientedCell};
use crate::mesh::elem_vertex::ElemVertexMesh;
use nalgebra::{ComplexField, DVector, Point, Scalar, U2};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use num_traits::Zero;
use std::collections::BTreeSet;
use std::hash::Hash;

/// Homogeneous Dirichlet boundary conditions on nodes.
pub struct DirichletBc<T> {
    /// Boundary-condition indices.
    idx_bc: BTreeSet<usize>,

    /// Degrees-of-freedom indices.
    idx_dof: BTreeSet<usize>,

    /// Vector of solution coefficients on the boundary.
    u_bc: DVector<T>
}

impl <T: Scalar + Zero> DirichletBc<T> {
    /// Constructs a new [`DirichletBc`] from the given set of boundary nodes `idx_bc`
    /// and solution coefficients at the boundary `u_bc`.
    pub fn new(num_nodes: usize, idx_bc: BTreeSet<Node>, u_bc: DVector<T>) -> Self {
        let idx: BTreeSet<Node> = (0..num_nodes).collect();
        let idx_dof: BTreeSet<Node> = idx.difference(&idx_bc).copied().collect();
        DirichletBc { idx_bc, idx_dof, u_bc }
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
        let idx_bc = msh.boundary_nodes().collect();
        DirichletBc::new(msh.num_nodes(), idx_bc, u_bc)
    }

    /// Constructs a new [`DirichletBc`] with *homogeneous* boundary data on the given `msh`.
    pub fn new_homogeneous_on_mesh<F, const M: usize>(msh: &ElemVertexMesh<T, F, M>) -> Self
    where F: CellBoundary<Dim = U2, Node = Node>,
          F::Node: Eq + Hash,
          F::SubCell: OrderedCell + OrientedCell + CellConnectivity + Clone + Eq + Hash
    {
        let idx_bc = msh.boundary_nodes().collect();
        DirichletBc::new_homogeneous(msh.num_nodes(), idx_bc)
    }

    /// Constructs a new [`DirichletBc`] on the given `msh`.
    ///
    /// The boundary data is given by the function `g`
    /// and the coefficients of the solution are calculated by evaluation of `g` at the boundary nodes.
    /// This only holds true for **interpolating** nodal basis function.
    pub fn new_interpolating<F: CellBoundary<Dim = U2, Node = Node>, const M: usize>(msh: &ElemVertexMesh<T, F, M>, g: impl Fn(&Point<T,M>) -> T) -> Self
    where F::Node: Eq + Hash,
          F::SubCell: OrderedCell + OrientedCell + CellConnectivity + Clone + Eq + Hash
    {
        // Find bc indices
        let idx_bc = msh.boundary_nodes().collect::<BTreeSet<_>>();

        // Evaluate boundary function `g` at boundary nodes
        let u_bc = DVector::from_iterator(
            idx_bc.len(),
            idx_bc.iter().map(|&node| g(&msh.coords[node]))
        );

        DirichletBc::new(msh.num_nodes(), idx_bc, u_bc)
    }
    
    /// Returns the number of DOFs.
    pub fn num_dof(&self) -> usize {
        self.idx_dof.len()
    }

    /// Returns the number of BC indices.
    pub fn num_bc(&self) -> usize {
        self.idx_bc.len()
    }

    /// Returns the total number of indices.
    pub fn num_indices(&self) -> usize {
        self.num_dof() + self.num_bc()
    }
}

impl <T: ComplexField + Copy> DirichletBc<T> {
    /// Deflates the system `ax = b` by removing rows and columns of BC indices.
    /// Returns the reduced system matrix and right hand side vector.
    pub fn deflate(&self, a: CsrMatrix<T>, b: DVector<T>) -> (CsrMatrix<T>, DVector<T>) {
        // Save num DOFs/BCs
        let num_dof = self.num_dof();
        let num_bc = self.num_bc();

        // Deflate right hand side
        // let b_dof = b.remove_rows_at(&idx_bc); // todo: is this better?
        let b_dof = b.select_rows(self.idx_dof.iter());

        // Build new COO matrix on DOF/BC indices
        let mut a_dof_dof = CooMatrix::new(num_dof, num_dof);
        for (i, &i_dof) in self.idx_dof.iter().enumerate() {
            for (j, &j_dof) in self.idx_dof.iter().enumerate() {
                // todo: the call to index_entry / get_entry is very inefficient.
                a_dof_dof.push(i, j, a.index_entry(i_dof, j_dof).into_value());
            }
        }

        let mut a_dof_bc = CooMatrix::new(num_dof, num_bc);
        for (i, &i_dof) in self.idx_dof.iter().enumerate() {
            for (j, &j_bc) in self.idx_bc.iter().enumerate() {
                // todo: the call to index_entry / get_entry is very inefficient.
                a_dof_bc.push(i, j, a.index_entry(i_dof, j_bc).into_value());
            }
        }

        // Conversion to CSR
        let a_dof_dof = CsrMatrix::from(&a_dof_dof);
        let a_dof_bc = CsrMatrix::from(&a_dof_bc);
        (a_dof_dof, b_dof - a_dof_bc * &self.u_bc)
    }

    /// Inflates a vector `u` by inserting the values from `u_dof` at the DOF indices.
    pub fn inflate(&self, u_dof: DVector<T>) -> DVector<T> {
        // Create empty coefficients vector
        let mut u = DVector::zeros(self.num_indices());

        // Fill BC values
        for (i_local, &i) in self.idx_bc.iter().enumerate() {
            u[i] = self.u_bc[i_local];
        }

        // Fill DOF values
        for (i_local, &i) in self.idx_dof.iter().enumerate() {
            u[i] = u_dof[i_local];
        }
        u
    }
}