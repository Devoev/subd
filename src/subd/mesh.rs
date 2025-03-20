use crate::subd::face::{edges_of_face, is_adjacent, sort_face};
use crate::subd::patch::{ExtendedPatch, Patch};
use itertools::{chain, Itertools};
use nalgebra::{
    center, matrix, vector, DMatrix, DVector, MatrixXx2
    , Point2, RealField, RowDVector, SMatrix, SVector, Vector2,
};
use std::collections::HashMap;
use std::iter::once;
use std::ops::{Deref, DerefMut};

pub type Node = usize;
pub type Edge = [Node; 2];
pub type Face = [Node; 4];

#[derive(Debug, Clone, Default)]
pub struct LogicalMesh {
    pub faces: Vec<Face>,
}

#[derive(Debug, Clone, Default)]
pub struct QuadMesh<T: RealField> {
    pub nodes: Vec<Point2<T>>,

    pub logical_mesh: LogicalMesh,
}

impl<T: RealField> Deref for QuadMesh<T> {
    type Target = LogicalMesh;

    fn deref(&self) -> &Self::Target {
        &self.logical_mesh
    }
}

impl<T: RealField> DerefMut for QuadMesh<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.logical_mesh
    }
}

impl<T: RealField + Copy> QuadMesh<T> {
    /// Returns the point of the given `node` index.
    pub fn node(&self, node: Node) -> Point2<T> {
        self.nodes[node]
    }

    /// Returns the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the nodes of the given `edge`.
    pub fn nodes_of_edge(&self, edge: &Edge) -> [Point2<T>; 2] {
        edge.map(|n| self.node(n))
    }

    /// Returns the nodes of the given `face`.
    pub fn nodes_of_face(&self, face: Face) -> [Point2<T>; 4] {
        face.map(|node| self.node(node))
    }

    /// Returns all faces who have the given `node` as a vertex.
    pub fn faces_of_node(&self, node: Node) -> impl Iterator<Item = (usize, &Face)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, face)| face.contains(&node))
    }

    /// Calculates the valence of the given `node`.
    pub fn valence(&self, node: Node) -> usize {
        self.faces_of_node(node).count()
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: Face) -> Option<Node> {
        face.into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Returns `true` if the face is regular.
    pub fn is_regular(&self, face: Face) -> bool {
        face.iter().all(|node| self.valence(*node) == 4)
    }

    /// Computes the centroid of the given `face`.
    pub fn centroid(&self, face: Face) -> Point2<T> {
        let corners = self.nodes_of_face(face);
        let centroid =
            corners.iter().map(|p| p.coords).sum::<Vector2<T>>() / T::from_f64(4.0).unwrap();
        Point2::from(centroid)
    }

    /// Returns all adjacent faces to `face`.
    pub fn adjacent_faces(&self, face: Face) -> impl Iterator<Item = (usize, &Face)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, f)| is_adjacent(f, &face))
    }

    /// Returns the one-ring around the given `node`.
    pub fn face_ring(&self, node: Node) -> Vec<&Face> {
        let mut faces = self.faces_of_node(node).map(|(_, f)| f).collect_vec();
        let mut sorted_faces = vec![faces.pop().unwrap()];
        while !faces.is_empty() {
            let prev = sorted_faces.last().unwrap();
            let (idx, next) = faces
                .iter()
                .find_position(|f| is_adjacent(f, prev))
                .unwrap();
            sorted_faces.push(next);
            faces.remove(idx);
        }
        sorted_faces
    }

    /// Returns the one-ring of nodes around the given `node`.
    pub fn node_ring(&self, node: Node) -> Vec<Node> {
        let mut faces = self.face_ring(node).into_iter().cloned().collect_vec();

        faces
            .iter_mut()
            .flat_map(|face| {
                let idx = face.iter().position(|v| *v == node).unwrap();
                face.rotate_left(idx);
                face[1..=2].iter().copied()
            })
            .collect_vec()
    }

    /// Finds the patch of the regular or irregular `face`.
    pub fn find_patch(&self, face: Face) -> Patch<T> {
        // todo: describe how the starting node is selected or change
        Patch::find(self, face, self.irregular_node_of_face(face).unwrap_or(face[0]))
    }

    /// Finds an extended patch of the irregular `face`.
    pub fn find_patch_ext(&self, face: Face) -> ExtendedPatch<T> {
        ExtendedPatch::find(self, face)
    }

    /// Evaluates the basis patch at the parametric point `(u,v)`.
    pub fn eval_basis_patch(u: T, v: T) -> SVector<T, 16> {
        let mat = matrix![
            -1.0, 3.0, -3.0, 1.0;
            3.0, -6.0, 3.0, 0.0;
            -3.0, 0.0, 3.0, 0.0;
            1.0, 4.0, 1.0, 0.0;
        ]
        .cast::<T>()
            / T::from_i32(6).unwrap();

        let u_pow = vector![u.powi(3), u.powi(2), u, T::one()];
        let v_pow = vector![v.powi(3), v.powi(2), v, T::one()];

        let bu = mat * u_pow;
        let bv = mat * v_pow;
        bu.kronecker(&bv)
    }

    /// Evaluates the patch of `face` at the parametric point `(u,v)`.
    pub fn eval_patch(&self, face: Face, u: T, v: T) -> SVector<T, 2> {
        // Convert patch control points to matrix
        let patch_nodes = self.find_patch(face).nodes_regular();
        let rows = patch_nodes
            .iter()
            .map(|node| self.node(*node).coords)
            .collect_vec();
        let coords = SMatrix::<T, 2, 16>::from_columns(&rows);

        // Evaluate basis functions at (u,v)
        let b = QuadMesh::eval_basis_patch(u, v);

        // Compute surface point
        coords * b
    }

    /// Linearly subdivides this mesh.
    pub fn lin_subd(&mut self) {
        let mut edge_midpoints = HashMap::<Edge, Node>::new();
        let mut faces = Vec::<Face>::new();

        // Refine every mesh face
        for i in 0..self.faces.len() {
            let face = self.faces[i];

            // Get edges of initial face
            let edges = edges_of_face(face);

            // Calculate and add new mid-edge points
            let midpoints = edges.map(|mut edge| {
                // Sort edge, to avoid duplicates
                edge.sort();

                // Computes the midpoint of the edge.
                match edge_midpoints.get(&edge) {
                    Some(node) => *node,
                    None => {
                        let [a, b] = self.nodes_of_edge(&edge);
                        let node = self.num_nodes();
                        self.nodes.push(center(&a, &b));
                        edge_midpoints.insert(edge, node);
                        node
                    }
                }
            });

            // Calculate new center point
            let center = self.centroid(face);
            let m = self.num_nodes();
            self.nodes.push(center);

            // Add subdivided faces
            let [a, b, c, d] = face;
            let [ab, bc, cd, da] = midpoints;

            faces.push([a, ab, m, da]);
            faces.push([ab, b, bc, m]);
            faces.push([m, bc, c, cd]);
            faces.push([da, m, cd, d]);
        }

        // Update faces
        self.faces = faces
    }

    /// Computes the dual of this mesh.
    pub fn dual(&mut self) {
        // Get dual nodes by calculating centroids of faces
        let dual_nodes = self
            .faces
            .iter()
            .map(|face| self.centroid(*face))
            .collect_vec();

        // Get dual faces by finding all dual nodes/ primal faces
        let dual_faces: Vec<Face> = (0..self.num_nodes())
            .flat_map(|node| {
                let adjacent_faces = self.faces_of_node(node).map(|(i, _)| i as Node);
                let dual_face = adjacent_faces.collect_array()?;
                let vertices = dual_face.map(|v| dual_nodes[v]);
                Some(sort_face(dual_face, vertices, self.node(node)))
            })
            .collect_vec();

        self.nodes = dual_nodes;
        self.faces = dual_faces;
    }

    /// Refine this mesh by repeated averaging algorithm.
    pub fn repeated_averaging(&mut self, levels: usize, p: usize) {
        for _ in 0..levels {
            self.lin_subd();
            for _ in 0..p - 1 {
                self.dual();
            }
        }
    }

    /// Applies the Catmull-Clark scheme to this mesh.
    pub fn catmull_clark(&mut self) {
        let mut edge_points = HashMap::<Edge, Node>::new();
        let mut face_points = HashMap::<Face, Node>::new();
        let mut nodes = Vec::<Point2<T>>::new();

        for node in 0..self.num_nodes() {
            let coords = self.catmull_clark_step(node);
            for view in coords.row_iter() {
                let point = Point2::from(view.into_owned().transpose());
            }
        }
    }

    /// Performs one subdivision of the given `node`.
    fn catmull_clark_step(&self, node: Node) -> MatrixXx2<T> {
        let n = self.valence(node);
        let ring = self.node_ring(node);
        let edge_nodes = [ring[0], ring[2], ring[4], ring[6], ring[8]];
        let face_nodes = [ring[1], ring[3], ring[5], ring[7], ring[9]];

        let rows = chain!(face_nodes, edge_nodes, once(node))
            .map(|idx| self.node(idx).coords.transpose())
            .collect_vec();

        let coords = MatrixXx2::from_rows(&rows);
        let s = QuadMesh::<T>::build_catmull_clark_matrix(n);
        s * coords
    }

    /// Builds the `2n+1 ✕ 2n+1` subdivision matrix.
    pub fn build_catmull_clark_matrix(n: usize) -> DMatrix<T> {
        let weight = 1.0 / 16.0;
        let n_inv_squared = 1.0 / (n as f64).powi(2);

        // Assemble sub matrices
        // Faces to faces
        let ff = DMatrix::<f64>::from_diagonal_element(n, n, 4.0);

        // Edges to faces
        let mut ef = DMatrix::<f64>::from_element(n, n, 4.0);
        ef.fill_lower_triangle(0.0, 1);
        ef.fill_upper_triangle(0.0, 2);
        ef[(n - 1, 0)] = 4.0;

        // Vertex to faces
        let vf = DVector::from_element(n, 4.0);

        // Faces to edges
        let mut fe = DMatrix::<f64>::from_element(n, n, 1.0);
        fe.fill_lower_triangle(0.0, 2);
        fe.fill_upper_triangle(0.0, 1);
        fe[(0, n - 1)] = 1.0;

        // Edges to edges
        let mut ee = DMatrix::<f64>::from_element(n, n, 6.0);
        ee.fill_lower_triangle(1.0, 1);
        ee.fill_lower_triangle(0.0, 2);
        ee[(n - 1, 0)] = 1.0;
        ee.fill_upper_triangle_with_lower_triangle();

        // Vertex to edges
        let ve = DVector::from_element(n, 6.0);

        // Faces to vertex
        let fv = RowDVector::from_element(n, 4.0 * n_inv_squared);

        // Edges to vertex
        let ev = RowDVector::from_element(n, 24.0 * n_inv_squared);

        // Vertex to vertex
        let vv = (16.0 * (n as f64) - 28.0) / (n as f64);

        // Assemble total matrix
        let mut s = DMatrix::<f64>::zeros(2 * n + 1, 2 * n + 1);
        s.view_mut((0, 0), (n, n)).copy_from(&ff);
        s.view_mut((0, n), (n, n)).copy_from(&ef);
        s.view_mut((0, 2 * n), (n, 1)).copy_from(&vf);
        s.view_mut((n, 0), (n, n)).copy_from(&fe);
        s.view_mut((n, n), (n, n)).copy_from(&ee);
        s.view_mut((n, 2 * n), (n, 1)).copy_from(&ve);
        s.view_mut((2 * n, 0), (1, n)).copy_from(&fv);
        s.view_mut((2 * n, n), (1, n)).copy_from(&ev);
        s[(2 * n, 2 * n)] = vv;

        (s * weight).cast()
    }
}
