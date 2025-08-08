use crate::cells;
use crate::cells::geo;
use crate::cells::line_segment::DirectedEdge;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadBndTopo, QuadNodes};
use crate::cells::unit_cube::UnitCube;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::subd::catmull_clark::basis::CatmarkPatchBasis;
use crate::subd::catmull_clark::map::CatmarkMap;
use crate::subd::catmull_clark::mesh::CatmarkMesh;
use itertools::Itertools;
use nalgebra::{Const, DimName, DimNameSub, Dyn, OMatrix, Point, RealField, U2};
use num_traits::ToPrimitive;

/// A Catmull-Clark surface patch.
#[derive(Debug, Clone)]
pub enum CatmarkPatch<T: RealField, const M: usize> {
    /// The regular interior case. See [`CatmarkPatchNodes::Regular`].
    Regular([Point<T, M>; 16]),

    /// The regular boundary case. See [`CatmarkPatchNodes::Boundary`].
    Boundary([Point<T, M>; 12]),

    /// The regular corner case. See [`CatmarkPatchNodes::Corner`].
    Corner([Point<T, M>; 9]),

    /// The irregular interior case. See [`CatmarkPatchNodes::Irregular`].
    Irregular(Vec<Point<T, M>>, usize)
}

impl<T: RealField + Copy, const M: usize> CatmarkPatch<T, M> {
    /// Constructs a new [`CatmarkPatch`] from the given `msh` and `patch_topo`.
    pub fn from_msh(msh: &CatmarkMesh<T, M>, patch_topo: &CatmarkPatchNodes) -> Self {
        let coords = patch_topo
            .as_slice()
            .iter()
            .map(|node| *msh.coords(*node));
        match patch_topo {
            CatmarkPatchNodes::Regular(_) => CatmarkPatch::Regular(coords.collect_array().unwrap()),
            CatmarkPatchNodes::Boundary(_) => CatmarkPatch::Boundary(coords.collect_array().unwrap()),
            CatmarkPatchNodes::Corner(_) => CatmarkPatch::Corner(coords.collect_array().unwrap()),
            CatmarkPatchNodes::Irregular(_, n) => CatmarkPatch::Irregular(coords.collect_vec(), *n)
        }
    }

    /// Returns a slice containing the control points.
    pub fn as_slice(&self) -> &[Point<T, M>] {
        match self {
            CatmarkPatch::Regular(val) => val.as_slice(),
            CatmarkPatch::Boundary(val) => val.as_slice(),
            CatmarkPatch::Corner(val) => val.as_slice(),
            CatmarkPatch::Irregular(val, _) => val.as_slice(),
        }
    }

    /// Returns the bicubic Catmull-Clark basis functions corresponding to the patch.
    pub fn basis(&self) -> CatmarkPatchBasis {
        match self {
            CatmarkPatch::Regular(_) => CatmarkPatchBasis::Regular,
            CatmarkPatch::Boundary(_) => CatmarkPatchBasis::Boundary,
            CatmarkPatch::Corner(_) => CatmarkPatchBasis::Corner,
            CatmarkPatch::Irregular(_, n) => CatmarkPatchBasis::Irregular(*n)
        }
    }

    /// Returns a matrix `(c1,...,cN)` over the coordinates of control points of this patch.
    pub fn coords(&self) -> OMatrix<T, Dyn, Const<M>> {
        let points = self.as_slice()
            .iter()
            .map(|&point| point.coords.transpose())
            .collect_vec();
        OMatrix::from_rows(&points)
    }

    /// Returns the quadrilateral in the center of this patch.
    pub fn center_quad(&self) -> Quad<T, M> {
        match self {
            CatmarkPatch::Regular(val) => {
                Quad::new([val[5], val[6], val[10], val[9]])
            }
            CatmarkPatch::Boundary(val) => {
                Quad::new([val[1], val[2], val[6], val[5]])
            }
            CatmarkPatch::Corner(val) => {
                Quad::new([val[0], val[1], val[4], val[3]])
            }
            CatmarkPatch::Irregular(val, _) => {
                Quad::new([val[0], val[5], val[4], val[3]])
            }
        }
    }
}

impl <T: RealField + Copy + ToPrimitive, const M: usize> geo::Cell<T, (T, T), 2, M> for CatmarkPatch<T, M> {
    type RefCell = UnitCube<2>;
    type GeoMap = CatmarkMap<T, M>;

    fn ref_cell(&self) -> Self::RefCell {
        UnitCube
    }

    fn geo_map(&self) -> Self::GeoMap {
        CatmarkMap(self.clone()) // todo: possibly replace clone if reference is introduced in CatmarkMap
    }
}

// todo: the implementations below should be updated!

/// Patch-to-nodes topology of a Catmull-Clark surface patch.
#[derive(Clone, Debug)]
pub enum CatmarkPatchNodes {
    /// The regular interior case of valence `n=4`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///  12 -- 13 -- 14 -- 15
    ///   |     |     |     |
    ///   8 --- 9 -- 10 -- 11
    ///   |     |  p  |     |
    ///   4 --- 5 --- 6 --- 7
    ///   |     |     |     |
    ///   0 --- 1 --- 2 --- 3
    /// ```
    /// where `p` is the center face of the patch.
    Regular([NodeIdx; 16]),

    /// The regular boundary case of valence `n=3`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///  |     |     |     |
    ///  8 --- 9 -- 10 -- 11
    ///  |     |     |     |
    ///  4 --- 5 --- 6 --- 7
    ///  |     |  p  |     |
    ///  0 --- 1 --- 2 --- 3
    /// ```
    /// where `p` is the center face of the patch.
    Boundary([NodeIdx; 12]),

    /// The regular corner case of valence `n=2`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///  |     |     |
    ///  6 --- 7 --- 8 ---
    ///  |     |     |
    ///  3 --- 4 --- 5 ---
    ///  |  p  |     |
    ///  0 --- 1 --- 2 ---
    /// ```
    /// where `p` is the center face of the patch.
    Corner([NodeIdx; 9]),

    /// The irregular interior case of valence `n≠4`.
    /// The nodes are ordered in the following order
    /// ```text
    /// 2N+7--2N+6--2N+5--2N+1
    ///   |     |     |     |
    ///   2 --- 3 --- 4 --2N+2
    ///   |     |  p  |     |
    ///   1 --- 0 --- 5 --2N+3
    ///  ╱    ╱ |     |     |
    /// 2N   ╱  7 --- 6 --2N+4
    ///  ╲  ╱  ╱
    ///   ○ - 8
    /// ```
    /// where `p` is the center face of the patch and node `0` is the irregular node.
    Irregular(Vec<NodeIdx>, usize)

    // todo: add IrregularBoundary/Corner case of valence = 4
}

impl CatmarkPatchNodes {
    /// Finds the [`CatmarkPatchNodes`] in the given quad-vertex topology `msh`.
    /// The center face `p` is given by `quad`.
    pub fn find<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>, quad: &QuadNodes) -> Self {
        match CatmarkPatchFaces::find(msh, *quad) {
            CatmarkPatchFaces::Regular(faces) => {
                let pick = [
                    (0, 0), (0, 1), (1, 1), (2, 1),
                    (0, 3), (0, 2), (1, 2), (2, 2),
                    (6, 0), (6, 1), (7, 1), (8, 1),
                    (6, 3), (6, 2), (7, 2), (8, 2),
                ];
                let nodes = pick.map(|(face, node)| faces[face].0[node]);
                CatmarkPatchNodes::Regular(nodes)
            }
            CatmarkPatchFaces::Boundary(faces) => {
                let pick = [
                    (0, 0), (0, 1), (1, 1), (2, 1),
                    (3, 0), (3, 1), (4, 1), (5, 1),
                    (3, 3), (3, 2), (4, 2), (5, 2),
                ];
                let nodes = pick.map(|(face, node)| faces[face].0[node]);
                CatmarkPatchNodes::Boundary(nodes)
            }
            CatmarkPatchFaces::Corner(faces) => {
                let pick = [
                    (0, 0), (0, 1), (1, 1),
                    (2, 0), (2, 1), (3, 1),
                    (2, 3), (2, 2), (3, 2),
                ];
                let nodes = pick.map(|(face, node)| faces[face].0[node]);
                CatmarkPatchNodes::Corner(nodes)
            }
            CatmarkPatchFaces::Irregular(faces) => {
                // Get faces at irregular node
                let node_irr = faces[0].0[0];
                let mut nodes = vec![node_irr];

                let n = faces.len() - 5;
                let faces_irr = &faces[..n];
                let faces_reg = &faces[n..];

                // Get nodes around irregular faces
                let nodes_irr = faces_irr.iter()
                    .flat_map(|&face| [face.0[3], face.0[2]]);
                nodes.extend(nodes_irr);

                // Get faces away from irregular node
                let pick = [
                    (0, 2), (0, 1), (1, 1), (2, 1),
                    (3, 2), (4, 2), (4, 3)
                ];
                let nodes_reg = pick.map(|(face, node)| faces_reg[face].0[node]);

                // Combine both
                nodes.extend_from_slice(&nodes_reg);
                CatmarkPatchNodes::Irregular(nodes, n)
            }
        }
    }

    /// Returns a slice containing the nodes.
    pub fn as_slice(&self) -> &[NodeIdx] {
        match self {
            CatmarkPatchNodes::Regular(val) => val.as_slice(),
            CatmarkPatchNodes::Boundary(val) => val.as_slice(),
            CatmarkPatchNodes::Corner(val) => val.as_slice(),
            CatmarkPatchNodes::Irregular(val, _) => val.as_slice(),
        }
    }

    /// Returns the quadrilateral in the center of this patch.
    pub fn center_quad(&self) -> QuadNodes {
        match self {
            CatmarkPatchNodes::Regular(val) => {
                QuadNodes([val[5], val[6], val[10], val[9]])
            }
            CatmarkPatchNodes::Boundary(val) => {
                QuadNodes([val[1], val[2], val[6], val[5]])
            }
            CatmarkPatchNodes::Corner(val) => {
                QuadNodes([val[0], val[1], val[4], val[3]])
            }
            CatmarkPatchNodes::Irregular(val, _) => {
                QuadNodes([val[0], val[5], val[4], val[3]])
            }
        }
    }
}

impl cells::topo::Cell<U2> for CatmarkPatchNodes {
    fn nodes(&self) -> &[NodeIdx] {
        self.as_slice()
    }

    // todo: possibly change this
    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U2: DimNameSub<M>
    {
        self.center_quad().is_connected(&other.center_quad(), dim)
    }
    
    // todo: does it make sense to override the method here? or change signature of trait

    /// Returns `true` if the center face of this patch contains the given node.
    fn contains_node(&self, node: NodeIdx) -> bool {
        self.center_quad().contains_node(node)
    }
}

impl cells::topo::CellBoundary<U2> for CatmarkPatchNodes  {
    const NUM_SUB_CELLS: usize = 4;
    type SubCell = DirectedEdge;
    type Boundary = QuadBndTopo;

    // todo: possibly change this
    fn boundary(&self) -> QuadBndTopo {
        self.center_quad().boundary()
    }
}

/// Patch-to-faces topology of a Catmull-Clark surface patch.
#[derive(Clone, Debug)]
pub enum CatmarkPatchFaces {
    /// The regular interior case of valence `n=4`.
    /// The faces are ordered in lexicographical order
    /// ```text
    /// +---+---+---+
    /// | 6 | 7 | 8 |
    /// +---+---+---+
    /// | 3 | p | 5 |
    /// +---+---+---+
    /// | 0 | 1 | 2 |
    /// +---+---+---+
    /// ```
    /// where `p` is the center face of the patch.
    /// Each face is sorted such that the first node is the lower left one.
    Regular([QuadNodes; 9]),

    /// The regular boundary case of valence `n=3`.
    /// The faces are ordered in lexicographical order
    /// ```text
    /// |   |   |   |
    /// +---+---+---+
    /// | 3 | 4 | 5 |
    /// +---+---+---+
    /// | 0 | p | 2 |
    /// +---+---+---+
    /// ```
    /// where `p` is the center face of the patch.
    /// Each face is sorted such that the first node is the lower left one.
    Boundary([QuadNodes; 6]),

    /// The regular corner case of valence `n=2`.
    /// The faces are ordered in lexicographical order
    /// ```text
    /// |   |   |
    /// +---+---+---
    /// | 2 | 3 |
    /// +---+---+---
    /// | p | 1 |
    /// +---+---+---
    /// ```
    /// where `p` is the center face of the patch.
    /// Each face is sorted such that the first node is the lower left one.
    Corner([QuadNodes; 4]),

    /// The irregular interior case of valence `n≠4`.
    /// The faces are ordered in the following order
    /// ```text
    ///   +-----+-----+-----+
    ///   | n+4 | n+3 |  n  |
    ///   +-----+-----+-----+
    ///   |  0  |  p  | n+1 |
    ///   +-----+-----+-----+
    ///  /     /|  2  | n+2 |
    /// + n-1 / +-----+-----+
    ///  \   / /
    ///   ○---+
    /// ```
    /// where `p` is the center face of the patch. The orientation of each face is as
    /// - Face `p`: Sorted such that the irregular node is the lower left one.
    /// - Faces `0..n-1`: Sorted such that the first node is the irregular one.
    /// - Faces `n..n+4`: Sorted such that the first node is the lower left one.
    Irregular(Vec<QuadNodes>)
}

impl CatmarkPatchFaces {
    // todo: remove T and M arguments. Can we replace QuadVertexMesh argument=?
    /// Finds the faces of the `msh` making up the patch of the `center` quadrilateral.
    pub fn find<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>, center: QuadNodes) -> CatmarkPatchFaces {
        // Find all faces in the 1-ring neighborhood
        let faces = msh.elems.iter()
            .filter(|other| other.is_touching(center))
            .collect_vec();

        if msh.is_regular_face(center) || msh.is_boundary_elem(&center) {
            match faces.len() {
                8 => {
                    let faces: [QuadNodes; 9] = Self::traverse_faces_regular(center, faces).try_into().unwrap();
                    CatmarkPatchFaces::Regular(faces)
                },
                5 => {
                    let node_irr = center.0.into_iter().enumerate()
                        .find_map(|(idx, node)| {
                            let next_idx = (idx + 1) % 4;
                            let next_node = center.0[next_idx];
                            (msh.valence(node) == 4 && msh.valence(next_node) != 4).then_some(next_node)
                        }).unwrap();
                    let center_face = center.sorted_by_origin(node_irr);
                    let faces: [QuadNodes; 6] = Self::traverse_faces_regular(center_face, faces).try_into().unwrap();
                    CatmarkPatchFaces::Boundary(faces)
                },
                3 => {
                    let node_irr = center.0.into_iter()
                        .find(|&n| msh.valence(n) == 2)
                        .unwrap();
                    let center_face = center.sorted_by_origin(node_irr);
                    let faces: [QuadNodes; 4] = Self::traverse_faces_regular(center_face, faces).try_into().unwrap();
                    CatmarkPatchFaces::Corner(faces)
                },
                _ => panic!("Possibly add more options for `faces.len()` (is {})", faces.len())
            }
        } else {
            let node_irr = msh.irregular_node_of_face(center).unwrap();
            let center_face = center.sorted_by_origin(node_irr);
            let faces = Self::traverse_faces_irregular(center_face, faces);
            CatmarkPatchFaces::Irregular(faces)
        }
    }

    /// Traverses the given `faces` of a **regular** patch in lexicographical order around the already **sorted** `center_face`
    /// and returns them in a vector.
    ///
    /// The traversal order is compatible with [`FaceConnectivity::Regular`], [`FaceConnectivity::Boundary`] and [`FaceConnectivity::Corner`].
    fn traverse_faces_regular(center: QuadNodes, faces: Vec<&QuadNodes>) -> Vec<QuadNodes> {
        let center_edges = center.edges();
        let mut faces_sorted: Vec<Option<QuadNodes>> = vec![None; 9];
        faces_sorted[4] = Some(center);

        for &face in faces {
            // Check if faces are connected by an edge
            let edges = face.edges();
            let edge_res = center_edges.iter().find_position(|&&edge| edges.contains(&edge.reversed()));
            if let Some((i, edge)) = edge_res {
                match i {
                    0 => faces_sorted[1] = Some(face.sorted_by_node(edge.end(), 2)),
                    1 => faces_sorted[5] = Some(face.sorted_by_node(edge.end(), 3)),
                    2 => faces_sorted[7] = Some(face.sorted_by_node(edge.end(), 0)),
                    3 => faces_sorted[3] = Some(face.sorted_by_node(edge.end(), 1)),
                    _ => unreachable!()
                }
                continue
            }

            // Check if faces are connected by a node
            let node_res = center.0.iter().find_position(|&&node| face.nodes().contains(&node));
            if let Some((i, &node)) = node_res {
                match i {
                    0 => faces_sorted[0] = Some(face.sorted_by_node(node, 2)),
                    1 => faces_sorted[2] = Some(face.sorted_by_node(node, 3)),
                    2 => faces_sorted[8] = Some(face.sorted_by_node(node, 0)),
                    3 => faces_sorted[6] = Some(face.sorted_by_node(node, 1)),
                    _ => unreachable!()
                }
                continue
            }

            panic!("Face {face:?} is not connected to the center face {center:?}");
        }

        faces_sorted.into_iter().flatten().collect()
    }

    /// Traverses the given `faces` of an **irregular** patch around the already **sorted** `center_face`
    /// and returns them in a vector.
    ///
    /// The traversal order is compatible with [`FaceConnectivity::Irregular`].
    fn traverse_faces_irregular(center: QuadNodes, mut faces: Vec<&QuadNodes>) -> Vec<QuadNodes> {

        // Find irregular faces 0..n-1
        let mut faces_irregular = vec![center];
        let node_irr = center.nodes()[0];

        let mut found_face = center;
        while faces.len() > 5 {
            // Find next face
            let next_edge = found_face.edges()[0];
            let face_idx = faces.iter()
                .position(|&&face| face.edges().contains(&next_edge.reversed()))
                .expect("No face contains given edge. Check faces argument.");

            found_face = faces.swap_remove(face_idx).sorted_by_origin(node_irr);
            faces_irregular.push(found_face);
        }

        faces_irregular.rotate_right(1);

        // Find regular faces n..n+4
        let mut faces_regular = vec![QuadNodes([NodeIdx(0); 4]); 5]; // todo: replace with better default value

        // Find faces connected to edges 1 and 2
        for (edge_dix, &edge) in center.edges()[1..=2].iter().enumerate() {
            let face_idx = faces.iter()
                .position(|&&face| face.edges().contains(&edge.reversed()))
                .expect("No face contains given edge. Check faces argument.");
            let face = *faces.swap_remove(face_idx);
            match edge_dix {
                0 => faces_regular[1] = face.sorted_by_node(edge.start(), 0),
                1 => faces_regular[3] = face.sorted_by_node(edge.start(), 1),
                _ => unreachable!("Iterating over only 2 edges!")
            };
        }

        // Find faces connected to nodes 1, 2, 3
        for (node_idx, node) in center.nodes()[1..=3].iter().enumerate() {
            let face_idx = faces.iter()
                .position(|&&face| face.nodes().contains(node))
                .expect("No face contains given node. Check faces argument.");
            let face = *faces.swap_remove(face_idx);
            match node_idx {
                0 => faces_regular[2] = face.sorted_by_node(*node, 3),
                1 => faces_regular[0] = face.sorted_by_node(*node, 0),
                2 => faces_regular[4] = face.sorted_by_node(*node, 1),
                _ => unreachable!("Iterating over only 3 nodes!")
            }
        }

        // Combine faces
        faces_irregular.extend_from_slice(&faces_regular);
        faces_irregular
    }

    /// Returns a slice containing the faces.
    pub fn as_slice(&self) -> &[QuadNodes] {
        match self {
            CatmarkPatchFaces::Regular(val) => val.as_slice(),
            CatmarkPatchFaces::Boundary(val) => val.as_slice(),
            CatmarkPatchFaces::Corner(val) => val.as_slice(),
            CatmarkPatchFaces::Irregular(val) => val.as_slice(),
        }
    }
}