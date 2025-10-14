//! This example parses a 2D NURBS representation of an electron gun
//! from the file `egun_geo.txt` in a GeoPDEs format using [`parse_geopdes_nurbs`].

use itertools::Itertools;
use nalgebra::{point, Point2};
use std::collections::HashMap;
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

    // todo: to convert the above format to a quad-vertex mesh, do the following
    //  - Iterate over every patch/ quad and find all interfaces connected to it
    //  - For every interface check if the nodes (interface_idx, node_idx_1/2) has already been processed
    //  - If not, add the corresponding vertex to a HashSet with key (interface_idx, node_idx_1/2)
    //  - Otherwise, re-use the vertex at key (interface_idx, node_idx_1/2)

    // Build quads from each patch
    let quads = patches.iter()
        .filter_map(|patch| {
            let control_points = &patch.4;
            if control_points.ncols() != 4 { return None }  // fixme: this is only to form a quad. Approximate other patches as well
            let vertices = control_points.column_iter()
                .map(|col| point![col[0], col[1]])
                .take(4)
                .collect_array::<4>()
                .unwrap();

            let vertices = [vertices[0], vertices[1], vertices[3], vertices[2]];
            Some(Quad::new(vertices))
        })
        .collect_vec();

    // For each interface save the global index of the starting (or ending) node.
    let mut interface_to_nodes = HashMap::<usize, usize>::new();
    let mut vertices = Vec::<Point2<f64>>::new();
    let mut faces = Vec::<QuadNodes>::new();

    let side_to_edge = |side: usize| {
        match side {
            1 => 3,
            2 => 1,
            3 => 0,
            4 => 2,
            _ => panic!("Side must be between 1 and 4")
        }
    };

    for quad_idx in 0..quads.len() {
        let gamma = interfaces.iter()
            .enumerate()
            .filter_map(|(idx, (_, (patch1, side1), (patch2, side2), orientation))| {
                if patch1 - 1 == quad_idx {
                    Some((side_to_edge(*side1), orientation[0], idx))
                } else if patch2 - 1 == quad_idx {
                    Some((side_to_edge(*side2), orientation[0], idx))
                } else {
                    None
                }
            })
            .collect_vec();

        let bnd = boundaries.iter()
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
            })
            .collect_vec();

        let quad = quads[quad_idx];
        let edges = quad.edges();
        let mut nodes = [0; 4];
        // println!("SIDES OF FACE {quad:?}");
        // Interfaces
        for (side, orientation, gamma_idx) in gamma {
            // Save node index of interface
            // todo: consider orientation and test if already present
            interface_to_nodes.insert(gamma_idx, vertices.len());

            // Save nodes
            nodes[side] = vertices.len();

            // Save vertex coordinates
            let vertex = edges[side].vertices[0];
            // dbg!("(Interface)", side, vertex);
            vertices.push(vertex)
        }

        // Boundaries
        for side in bnd {
            // Save nodes
            nodes[side] = vertices.len();

            // Save vertex coordinates
            let vertex = edges[side].vertices[0];
            // dbg!("(Boundary)", side, vertex);
            vertices.push(vertex)
        }

        faces.push(QuadNodes(nodes))
    }

    // dbg!(&vertices);
    // dbg!(&faces);
    // dbg!(&quads);
    let msh = QuadVertexMesh::new(vertices, faces);
    plot_faces(&msh, msh.cell_iter().copied()).show();

    // // Collect vertices of quads
    // let mut quad_to_nodes = HashMap::<usize, QuadNodes>::new();
    //
    // for (_, (patch1, side1), (patch2, side2), orientation) in interfaces {
    //     // Get geometrical quads and interfaces
    //     let quad1 = &quads[patch1-1];
    //     let side1 = &quad1.edges()[side1-1];
    //     let quad2 = &quads[patch2-1];
    //     let side2 = &quad2.edges()[side2-1];
    //
    //     if let Some(nodes) = quad_to_nodes.get_mut(&(patch1 - 1)) {
    //         // todo: update two nodes corresponding to the interface sides
    //     } else {
    //         // todo: create new QuadNodes and update with interface data
    //     }
    //
    //     dbg!((side1, side2, orientation));
    // }
}