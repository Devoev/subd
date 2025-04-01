use crate::subd::basis;
use crate::subd::edge::{next_edge, reverse_edge, sort_edge_by_face};
use crate::subd::face::{edges_of_face, sort_by_node, sort_by_origin};
use crate::subd::mesh::{Face, Node, QuadMesh};
use itertools::{izip, Itertools};
use nalgebra::{Dyn, Matrix2, OMatrix, Point2, RealField, SMatrix, U2};
use num_traits::ToPrimitive;
use std::collections::{HashMap, HashSet};
use std::iter::once;
use gauss_quad::GaussLegendre;

/// A patch of a quadrilateral mesh.
/// The faces are sorted in clockwise order, i.e.
/// ```text
/// +---+---+---+
/// | 1 | 2 | 3 |
/// +---+---+---+
/// | 0 | f | 4 |
/// +---+---+---+
/// | 7 | 6 | 5 |
/// +---+---+---+
/// ```
#[derive(Debug, Clone)]
// pub struct Patch<'a, T: RealField> {
//     /// The mesh reference.
//     msh: &'a QuadMesh<T>,
//
//     /// The faces of this mesh.
//     pub faces: Vec<Face>,
//
//     /// The center face `f`.
//     pub center: Face
// }

pub enum Patch<'a, T: RealField> {
    Regular {
        msh: &'a QuadMesh<T>,
        faces: Vec<Face>,
        center: Face
    },
    Irregular {
        msh: &'a QuadMesh<T>,
        faces: Vec<Face>,
        center: Face
    },
    BoundaryRegular {
        msh: &'a QuadMesh<T>,
        faces: Vec<Face>,
        center: Face
    },
    BoundaryRegularCorner {
        msh: &'a QuadMesh<T>,
        faces: Vec<Face>,
        center: Face
    }
}

impl<'a, T: RealField + Copy> Patch<'a, T> {
    /// Finds the patch with the `center` face.
    ///
    /// The node `start` is the node between the faces `f`, `0`, `7` and `6`,
    /// i.e. the node of the center face closest to the parametric origin.
    pub fn find(msh: &'a QuadMesh<T>, center: Face, start: Node) -> Self {
        // todo: update/ optimize this code

        // Find all faces in the 1-ring neighborhood
        let mut faces_neighborhood = msh.faces.iter().filter(|other| {
            let other_set = HashSet::from(**other);
            let face_set = HashSet::from(center);
            let count = face_set.intersection(&other_set).count();
            count == 1 || count == 2
        }).collect_vec();

        // Find starting point, i.e. face with an edge containing the start node
        let (mut idx, mut found_face, mut found_edge) = faces_neighborhood.iter().enumerate()
            .find_map(|(i, other)| {
                let edges = edges_of_face(**other);
                // Find edge that is included in face and starts with start
                let edge_irr = edges.iter().find(|edge| edge[0] == start && center.contains(&edge[1]));
                edge_irr.map(|edge| (i, *other, *edge))
            }).unwrap();

        let mut faces_sorted = vec![*found_face];

        while faces_neighborhood.len() > 1 {
            // Remove already visited face
            faces_neighborhood.swap_remove(idx);

            // Find next face
            let next_e = next_edge(found_edge, *found_face);
            let next_next_e = next_edge(next_e, *found_face);
            let next_inv = reverse_edge(next_e);
            let next_next_inv = reverse_edge(next_next_e);

            (idx, found_face, found_edge) = faces_neighborhood.iter().enumerate()
                .find_map(|(i, other)| {
                    let edges = edges_of_face(**other);
                    let found = edges.iter().find(|edge| **edge == next_inv || **edge == next_next_inv);
                    found.map(|edge| (i, *other, *edge))
                }).unwrap();

            // Save found face
            faces_sorted.push(*found_face);
        }

        // Check the type of patch
        let n = msh.valence(start);
        if !msh.is_boundary(center) {
            if n == 4 {
                Patch::Regular { msh, faces: faces_sorted, center }
            } else {
                Patch::Irregular { msh, faces: faces_sorted, center }
            }
        } else if n == 3 && faces_sorted.len() == 5 {
            Patch::BoundaryRegular { msh, faces: faces_sorted, center }
        } else if n == 2 || n == 3 {
            Patch::BoundaryRegularCorner { msh, faces: faces_sorted, center }
        } else {
            panic!("Irregular boundary patches (e.g. concave corners) are not implemented yet!")
        }
    }
}

impl<'a, T: RealField + Copy> Patch<'a, T> {

    /// Returns the [`QuadMesh`] belonging to this patch.
    fn msh(&self) -> &'a QuadMesh<T> {
        match self {
            Patch::Regular { msh, .. } => msh,
            Patch::Irregular { msh, .. } => msh,
            Patch::BoundaryRegular { msh, .. } => msh,
            Patch::BoundaryRegularCorner { msh, .. } => msh,
        }
    }

    /// Returns the center [`Face`] of this patch.
    pub fn center(&self) -> Face {
        match self {
            Patch::Regular { msh: _, center, .. } => *center,
            Patch::Irregular { msh: _, center, .. } => *center,
            Patch::BoundaryRegular { msh: _, center, .. } => *center,
            Patch::BoundaryRegularCorner { msh: _, center, .. } => *center
        }
    }

    /// Returns the vector of faces of this patch.
    pub fn faces(&self) -> &Vec<Face> {
        match self {
            Patch::Regular { msh: _, center: _, faces } => faces,
            Patch::Irregular { msh: _, center: _, faces } => faces,
            Patch::BoundaryRegular { msh: _, center: _, faces } => faces,
            Patch::BoundaryRegularCorner { msh: _, center: _, faces } => faces,
        }
    }

    /// Finds and returns the irregular node index and its valence.
    /// Returns `None` if there is no irregular node.
    pub fn irregular_node(&self) -> Option<(usize, usize)> {
        let node_irr = self.msh().irregular_node_of_face(self.center())?;
        let n = self.msh().valence(node_irr);
        Some((node_irr, n))
    }

    /// Sorts the faces of this **regular** patch, such that the origin is given by `uv_origin`.
    /// This is done by successively applying [`sort_by_origin`] to each patch face.
    fn sort_faces_regular(&self, uv_origin: Node) -> Self {
        // fixme: this assumes that face 7 is the lower left one (in uv space) and includes uv_origin.
        //  Is this always true? Maybe fix this, by ALWAYS sorting in the construction process?
        // Sort face 7
        let &last = self.faces().last().unwrap();
        let f7 = sort_by_origin(last, uv_origin);
        let n5 = f7[2];

        // Sort center face
        let sorted_center = sort_by_origin(self.center(), n5);
        let [_, n6, n10, n9] = sorted_center;

        // Sort other faces
        let anchor_to_idx = [
            (n5, 1), (n9, 1), (n10, 1), (n10, 0), (n6, 0), (n6, 3), (n6, 2)
        ];
        let sorted_faces = anchor_to_idx.iter().enumerate()
            .map(|(face_id, (node, idx))| sort_by_node(self.faces()[face_id], *node, *idx))
            .chain(once(f7))
            .collect();

        Patch::Regular { msh: self.msh(), faces: sorted_faces, center: sorted_center }
    }

    /// Sorts the faces of this **planar boundary** patch, such that the origin is given by `uv_origin`.
    /// This is done by successively applying [`sort_by_origin`] to each patch face.
    fn sort_faces_boundary_planar(&self, uv_origin: Node) -> Self {
        // Sort face 0
        let &first = self.faces().first().unwrap();
        let f0 = sort_by_origin(first, uv_origin);
        let n1 = f0[1];
        
        // Sort center face
        let sorted_center = sort_by_origin(self.center(), n1);
        let [_, _, n6, n5] = sorted_center;

        // Sort other faces
        let anchor_to_idx = [
            (n5, 1), (n6, 1), (n6, 0), (n6, 3)
        ];
        let sorted_faces = once(f0).chain(
            anchor_to_idx.iter().enumerate()
                .map(|(face_id, (node, idx))| sort_by_node(self.faces()[face_id + 1], *node, *idx))
        ).collect();
        
        Patch::BoundaryRegular { msh: self.msh(), faces: sorted_faces, center: sorted_center }
    }

    /// Sorts the faces of this **convex boundary** patch, such that the origin is given by `uv_origin`.
    /// This is done by successively applying [`sort_by_origin`] to each patch face.
    fn sort_faces_boundary_convex(&self, uv_origin: Node) -> Self {
        // Sort center face
        let sorted_center = sort_by_origin(self.center(), uv_origin);
        let [_, _, n4, _] = sorted_center;

        // Sort other faces
        let anchor_to_idx = [
            (n4, 1), (n4, 0), (n4, 3)
        ];

        let sorted_faces = anchor_to_idx.iter().enumerate()
            .map(|(face_id, (node, idx))| sort_by_node(self.faces()[face_id], *node, *idx))
            .collect();

        Patch::BoundaryRegular { msh: self.msh(), faces: sorted_faces, center: sorted_center }
    }

    /// Returns the nodes of this regular patch in lexicographical order, i.e.
    /// ```text
    /// 12 -- 13 -- 14 -- 15
    ///  |     |     |     |
    ///  8 --- 9 -- 10 -- 11
    ///  |     |     |     |
    ///  4 --- 5 --- 6 --- 7
    ///  |     |     |     |
    ///  0 --- 1 --- 2 --- 3
    /// ```
    fn nodes_regular(&self) -> [Node; 16] {
        // todo: move this to the creation process of a patch
        // Find uv origin and sort all faces
        let uv_opposite = self.faces()[7].iter().position(|n| self.center().contains(n)).unwrap();
        let uv_origin = (uv_opposite + 2) % 4;
        let sorted = self.sort_faces_regular(self.faces()[7][uv_origin]);

        let pick = [
            (7, 0), (7, 1), (6, 1), (5, 1),
            (7, 3), (7, 2), (6, 2), (5, 2),
            (1, 0), (1, 1), (2, 1), (3, 1),
            (1, 3), (1, 2), (2, 2), (3, 2),
            
        ];
        pick.map(|(face, node)| sorted.faces()[face][node])
    }

    /// Returns the nodes of this planar boundary patch in lexicographical order, i.e.
    /// ```text
    ///  |     |     |     |
    ///  8 --- 9 -- 10 -- 11
    ///  |     |     |     |
    ///  4 --- 5 --- 6 --- 7
    ///  |     |  p  |     |
    ///  0 --- 1 --- 2 --- 3
    /// ```
    fn nodes_boundary_planar(&self) -> [Node; 12] {
        // Find uv origin and sort all faces
        let n5 = self.faces()[1].iter().find(|n| self.center().contains(n)).unwrap();
        let n1_idx = self.faces()[0].iter().position(|n| self.center().contains(n) && n != n5).unwrap();
        let uv_origin = (n1_idx + 3) % 4;
        let sorted = self.sort_faces_boundary_planar(self.faces()[0][uv_origin]);

        let pick = [
            (0, 0), (0, 1), (4, 0), (4, 1),
            (1, 0), (1, 1), (2, 1), (3, 1),
            (1, 3), (1, 2), (2, 2), (3, 2),

        ];
        pick.map(|(face, node)| sorted.faces()[face][node])
    }

    /// Returns the nodes of this convex corner boundary patch in lexicographical order, i.e.
    /// ```text
    ///  |     |     |
    ///  6 --- 7 --- 8 ---
    ///  |     |     |
    ///  3 --- 4 --- 5 ---
    ///  |  p  |     |
    ///  0 --- 1 --- 2 ---
    /// ```
    fn nodes_boundary_convex(&self) -> [Node; 9] {
        // Find uv origin and sort all faces
        let n0 = self.center().into_iter().find(|&n| self.msh().valence(n) == 2).unwrap();
        let sorted = self.sort_faces_boundary_convex(n0);

        let pick = [
            (2, 0), (2, 1),
            (0, 0), (1, 0), (1, 1),
            (0, 3), (1, 3), (1, 2)
        ];

        once(n0).chain(
            pick.into_iter().map(|(face, node)| sorted.faces()[face][node])
        ).collect_array().unwrap()
    }

    /// Returns the nodes of this irregular patch in the following order
    /// ```text
    /// 2N+7--2N+6--2N+5--2N+1
    ///   |     |     |     |
    ///   2 --- 3 --- 4 --2N+2
    ///   |     |     |     |
    ///   1 --- 0 --- 5 --2N+3
    ///  ╱    ╱ |     |     |
    /// 2N   ╱  7 --- 6 --2N+4
    ///  ╲  ╱  ╱
    ///   ○ - 8
    /// ```
    /// where `N` is the valence of the irregular node (`0` in the graphic).
    fn nodes_irregular(&self) -> Vec<Node> {
        // Get valence of irregular node
        let (node_irr, n) = self.irregular_node().expect("Patch must be irregular!");

        // Get faces at irregular node
        let mut inner_faces = vec![self.faces()[0], self.center()];
        inner_faces.extend_from_slice(&self.faces()[6..n+4]);

        // Get nodes of inner faces by setting their uv origin to the irregular node
        let nodes_it = inner_faces.iter().flat_map(|&face| {
            let sorted = sort_by_origin(face, node_irr);
            once(sorted[3]).chain(once(sorted[2]))
        });

        let mut inner_nodes = vec![node_irr];
        inner_nodes.extend(nodes_it);

        // Get faces away from irregular node
        let outer_faces = &self.faces()[1..=5].iter().enumerate().map(|(i, &face)| {
           sort_by_origin(face, inner_nodes[i + 2])
        }).collect_vec();

        let pick = [
            (2, 2), (2, 1), (3, 1), (4, 1),
            (1, 2), (0, 2), (0, 3)
        ];
        let outer_nodes = pick.map(|(face, node)| outer_faces[face][node]);

        // Combine both
        inner_nodes.extend_from_slice(&outer_nodes);
        inner_nodes
    }

    /// Returns a vector over the nodes of this patch.
    pub fn nodes(&self) -> Vec<Node> {
        match self {
            Patch::Regular { .. } => self.nodes_regular().into_iter().collect(),
            Patch::Irregular { .. } => self.nodes_irregular().into_iter().collect(),
            Patch::BoundaryRegular { .. } => self.nodes_boundary_planar().into_iter().collect(),
            Patch::BoundaryRegularCorner { .. } => self.nodes_boundary_convex().into_iter().collect(),
        }
    }

    /// Returns a matrix `(c1,...,cN)` over the coordinates of control points of this patch.
    pub fn coords(&self) -> OMatrix<T, U2, Dyn> {
        let points = self.nodes()
            .into_iter()
            .map(|n| self.msh().node(n).coords)
            .collect_vec();
        OMatrix::from_columns(&points)
    }
}

impl <'a, T: RealField + Copy + ToPrimitive> Patch<'a, T> {

    /// Evaluates this patch at the parametric point `(u,v)`.
    pub fn eval(&self, u: T, v: T) -> Point2<T> {
        match self {
            Patch::Regular { .. } => self.eval_regular(u, v),
            Patch::Irregular { .. } => self.eval_irregular(u, v),
            Patch::BoundaryRegular { .. } => self.eval_boundary_planar(u, v),
            Patch::BoundaryRegularCorner { .. } => self.eval_boundary_convex(u, v),
        }
    }
    
    /// Evaluates the jacobian of this patch at the parametric point `(u,v)`.
    pub fn eval_jacobian(&self, u: T, v: T) -> Matrix2<T> {
        if !matches!(self, Patch::Regular { .. }) { panic!("Only regular patches are implemented for now!") }
        
        let b_du = basis::eval_regular_du(u, v);
        let b_dv = basis::eval_regular_dv(u, v);
        let c = self.coords();
        
        Matrix2::from_columns(&[c.clone() * b_du, c * b_dv])
    }

    /// Numerically calculates the area of this patch using Gaussian quadrature.
    pub fn calc_area(&self) -> f64 {
        let quad = GaussLegendre::new(2).unwrap();
        let integrand = |u, v| {
            let d_phi = self.eval_jacobian(T::from_f64(u).unwrap(), T::from_f64(v).unwrap());
            d_phi.determinant().abs().powi(2).to_f64().unwrap() // fixme: why is powi(2) required??
        };
        quad.integrate(0.0, 1.0, |v| quad.integrate(0.0, 1.0, |u| integrand(u, v)))
    }

    /// Evaluates this regular patch at the parametric point `(u,v)`.
    fn eval_regular(&self, u: T, v: T) -> Point2<T> {
        // Evaluate basis functions and patch
        let b = basis::eval_regular(u, v);
        Point2::from(self.coords() * b)
    }

    /// Evaluates this planar boundary patch at the parametric point `(u,v)`.
    fn eval_boundary_planar(&self, u: T, v: T) -> Point2<T> {
        // Evaluate basis functions and patch
        let b = basis::eval_boundary(u, v, false, true);
        Point2::from(self.coords() * b)
    }

    /// Evaluates this convex boundary patch at the parametric point `(u,v)`.
    fn eval_boundary_convex(&self, u: T, v: T) -> Point2<T> {
        // Evaluate basis functions and patch
        let b = basis::eval_boundary(u, v, true, true);
        Point2::from(self.coords() * b)
    }

    /// Evaluates this irregular patch at the parametric point `(u,v)`.
    fn eval_irregular(&self, u: T, v: T) -> Point2<T> {
        // Get valence of irregular node
        let (_, n) = self.irregular_node().expect("Patch must be irregular!");

        // Evaluate basis functions and patch
        let b = basis::eval_irregular(u, v, n);
        Point2::from(self.coords() * b)
    }
}

/// An extended patch of a quadrilateral mesh.
/// Consist of 3 regular patches and one irregular patch.
pub struct ExtendedPatch<'a, T: RealField> {
    /// The irregular patch.
    patch_irr: Patch<'a, T>,
    /// The 3 regular patches surrounding the irregular one.
    patches_reg: [Patch<'a, T>; 3]
}

impl <'a, T: RealField + Copy> ExtendedPatch<'a, T> {

    /// Finds the extended patch with the center `face`.
    pub fn find(msh: &'a QuadMesh<T>, face: Face) -> Self {
        // Find the irregular node
        let node_irr = msh.irregular_node_of_face(face).expect("Face must be irregular!");

        let [_, a, b, c] = sort_by_origin(face, node_irr);
        let mut patch_faces = [Face::default(); 3];

        for face in &msh.faces {
            if face.contains(&a) && face.contains(&b) { patch_faces[0] = *face }
            else if face.contains(&b) && face.contains(&c) { patch_faces[2] = *face }
            else if face.contains(&b) { patch_faces[1] = *face }
        }

        // Create irregular patch and find nodes
        let patch_irr = Patch::find(msh, face, node_irr);
        let nodes_irr = patch_irr.nodes_irregular();

        // Node indices for the orientation of regular patches
        let starts = [5, 4, 3];
        let origins = [7, 0, 1];

        // Create and sort regular patches
        let patches_reg = izip!(patch_faces, starts, origins)
            .map(|(face, start_idx, origin_idx)| {
                let patch = Patch::find(msh, face, nodes_irr[start_idx]);
                patch.sort_faces_regular(nodes_irr[origin_idx])
            })
            .collect_array().unwrap();

        ExtendedPatch {
            patch_irr,
            patches_reg
        }
    }

    /// Returns the nodes of this extended patch in the following order
    /// ```text
    /// 2N+16-2N+15-2N+14-2N+13-2N+8
    ///   |     |     |     |     |
    /// 2N+7--2N+6--2N+5--2N+1--2N+9
    ///   |     |     |     |     |
    ///   2 --- 3 --- 4 --2N+2-2N+10
    ///   |     |     |     |     |
    ///   1 --- 0 --- 5 --2N+3-2N+11
    ///  ╱    ╱ |     |     |     |
    /// 2N   ╱  7 --- 6 --2N+4-2N+12
    ///  ╲  ╱  ╱
    ///   ○ - 8
    /// ```
    /// where `N` is the valence of the irregular node (`0` in the graphic).
    pub fn nodes(&self) -> Vec<Node> {
        let mut nodes = self.patch_irr.nodes_irregular();
        let nodes1 = &self.patches_reg[0].nodes_regular();
        let nodes2 = &self.patches_reg[1].nodes_regular();
        let nodes3 = &self.patches_reg[2].nodes_regular();

        nodes.extend_from_slice(&[nodes2[15], nodes2[11], nodes2[7], nodes2[3]]);
        nodes.push(nodes1[3]);
        nodes.extend_from_slice(&[nodes3[15], nodes3[14], nodes3[13], nodes3[12]]);

        nodes
    }
}