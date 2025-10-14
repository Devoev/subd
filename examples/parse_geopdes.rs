//! This example parses a 2D NURBS representation of an electron gun
//! from the file `egun_geo.txt` in a GeoPDEs format using [`parse_geopdes_nurbs`].

use approx::abs_diff_eq;
use itertools::Itertools;
use nalgebra::{point, Point2};
use subd::cells::quad::QuadNodes;
use subd::element::quad::Quad;
use subd::io::geopdes::parse_geopdes_nurbs;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::plot::plot_faces;

fn main() {
    // Parse Electron gun geometry from GeoPDEs file
    let path = "examples/egun_geo.txt";
    let (patches, interfaces, boundaries) = parse_geopdes_nurbs(path).expect("`egun_geo.txt` should exist in the examples/ folder");

    println!("----- Patches -----");
    for patch in &patches {
        let (name, degrees, nums, knots, control_points, weights) = patch;
        println!("Patch '{name}' of degrees {degrees:?} with knot vectors {knots:?}, weights = {weights:?} and control points = {control_points}");
    }

    println!("----- Interfaces -----");
    for interface in &interfaces {
        let (name, side1, side2, orientation) = interface;
        println!("Interface '{name}' with sides {side1:?} and {side2:?} and {orientation:?}");
    }

    println!("----- Boundaries -----");
    for boundary in &boundaries {
        let (name, sides) = boundary;
        println!("Boundary '{name}' with sides {sides:?}");
    }

    // Build quads from each patch
    // Do this, by just selecting the four corner vertices
    let quads = patches.iter()
        .map(|patch| {
            let c = &patch.4;
            let [px, py] = patch.1[0..2] else { panic!("Getting exactly two degrees") };

            let vertex_coords = if px == 1 && py == 1 {
                [c.column(0), c.column(1), c.column(3), c.column(2)]
            } else if px == 2 && py == 1 {
                [c.column(0), c.column(2), c.column(5), c.column(3)]
            } else if px == 1 && py == 2 {
                [c.column(0), c.column(1), c.column(5), c.column(4)]
            } else if px == 2 && py == 2 {
                [c.column(0), c.column(2), c.column(8), c.column(6)]
            } else {
                panic!("Degrees >= 2 are not supported yet");
            };

            let vertices = vertex_coords
                .map(|col| point![col[0], col[1]]);

            Quad::new(vertices)
        })
        .collect_vec();

    // Init vertex and face vectors for mesh
    let mut vertices = Vec::<Point2<f64>>::new();
    let mut faces = Vec::<QuadNodes>::new();

    // Converts the lexicographical index to the index used by `QuadNodes`
    let side_to_edge = |side: usize| {
        match side {
            1 => 3,
            2 => 1,
            3 => 0,
            4 => 2,
            _ => panic!("Side must be between 1 and 4")
        }
    };

    // Iterate over every quad to collect all 4 edges
    for (quad_idx, quad) in quads.iter().enumerate() {
        let quad_interfaces = interfaces.iter()
            .enumerate()
            .filter_map(|(idx, (_, (patch1, side1), (patch2, side2), orientation))| {
                if patch1 - 1 == quad_idx {
                    Some((side_to_edge(*side1), orientation[0], idx))
                } else if patch2 - 1 == quad_idx {
                    Some((side_to_edge(*side2), orientation[0], idx))
                } else {
                    None
                }
            });

        let quad_boundaries = boundaries.iter()
            .flat_map(|(_, sides)| {
                sides.iter()
                    .filter_map(|(patch, side)| {
                        if patch - 1 == quad_idx {
                            Some(side_to_edge(*side))
                        } else {
                            None
                        }
                    })
                    .collect_vec()
            });

        // All 4 interfaces corresponding to edges of `quad`
        let sides = quad_interfaces
            .map(|side| side.0) // omit interface and orientation
            .chain(quad_boundaries);

        let edges = quad.edges();
        let mut nodes = [None; 4];
        let tol = 1e-8;

        // Iterate over all 4 sides of the quad
        for side in sides {
            // Get vertex of edge
            let vertex = edges[side].vertices[0];

            // Search if `vertex` has already been added as a node
            if let Some(node) = vertices.iter().position(|&v| abs_diff_eq!(v, vertex, epsilon = tol)) {
                // If node does already exist, use it to set the node in `nodes`
                nodes[side] = Some(node);
            } else {
                // If node doesnt exist yet, append coordinates to `vertices` vector
                nodes[side] = Some(vertices.len());
                vertices.push(vertex)
            }
        }

        faces.push(QuadNodes(nodes.map(|n| n.expect("Node should be set"))))
    }

    // Build mesh
    let msh = QuadVertexMesh::new(vertices, faces);
    let msh = msh.lin_subd().unpack();
    plot_faces(&msh, msh.cell_iter().copied()).show();
}