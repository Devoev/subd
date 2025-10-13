//! This example parses a 2D NURBS representation of an electron gun
//! from the file `egun_geo.txt` in a GeoPDEs format using [`parse_geopdes_nurbs`].

use std::collections::HashMap;
use itertools::Itertools;
use nalgebra::{point, Point, Point2};
use subd::cells::quad::QuadNodes;
use subd::element::quad::Quad;
use subd::io::geopdes::parse_geopdes_nurbs;

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
        .map(|patch| {
            let control_points = &patch.4;
            let vertices = control_points.column_iter()
                .map(|col| point![col[0], col[1]])
                .take(4) // fixme: this is only to form a quad
                .collect_array()
                .unwrap();

            Quad::new(vertices)
        })
        .collect_vec();

    // Collect vertices of quads
    let mut quad_to_nodes = HashMap::<usize, QuadNodes>::new();

    for (_, (patch1, side1), (patch2, side2), orientation) in interfaces {
        // Get geometrical quads and interfaces
        let quad1 = &quads[patch1-1];
        let side1 = &quad1.edges()[side1-1];
        let quad2 = &quads[patch2-1];
        let side2 = &quad2.edges()[side2-1];

        if let Some(nodes) = quad_to_nodes.get_mut(&(patch1 - 1)) {
            // todo: update two nodes corresponding to the interface sides
        } else {
            // todo: create new QuadNodes and update with interface data
        }

        dbg!((side1, side2, orientation));
    }
}