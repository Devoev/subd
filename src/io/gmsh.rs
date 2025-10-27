use std::{fs, io};
use std::fmt::Debug;
use std::path::Path;
use std::str::FromStr;
use itertools::Itertools;
use nalgebra::{Point, Scalar};
use crate::cells::quad::QuadNodes;
use crate::mesh::face_vertex::QuadVertexMesh;

/// Parses a quadrilateral mesh from a file `path`
/// in the [Gmsh](https://gmsh.info/) mesh format `4.1`.
///
/// The format is described [here](https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format).
pub fn parse_gmsh_quad_mesh<T: Scalar + FromStr, const M: usize>(path: impl AsRef<Path>) -> io::Result<QuadVertexMesh<T, M>>
    where T::Err: Debug
{
    let str = fs::read_to_string(path)?;
    let lines = str.lines().collect_vec();

    // $Nodes
    // numEntityBlocks(size_t) numNodes(size_t)
    //  minNodeTag(size_t) maxNodeTag(size_t)
    // entityDim(int) entityTag(int) parametric(int; 0 or 1)
    //  numNodesInBlock(size_t)
    //  nodeTag(size_t)
    //     ...
    //     x(double) y(double) z(double)
    //     < u(double; if parametric and entityDim >= 1) >
    // < v(double; if parametric and entityDim >= 2) >
    // < w(double; if parametric and entityDim == 3) >
    // ...
    //     ...
    // $EndNodes

    // PARSE NODES
    // Header
    let nodes_start_idx = lines.iter()
        .position(|line| line.starts_with("$Nodes"))
        .expect("$Nodes must be defined");

    let header = lines[nodes_start_idx + 1];
    let [num_entity_bocks, num_nodes, min_node, max_node] = header.split_whitespace()
        .map(|str| str.parse::<usize>().expect("First line of $Nodes must only contain integers"))
        .collect_array()
        .expect("First line must contain exactly four integers");

    // Parse each node block
    let mut verts = Vec::<Point<T, M>>::with_capacity(num_nodes);
    let mut nodes_iter = lines.iter().skip(nodes_start_idx + 2);

    while let Some(header) = nodes_iter.next() {
        // Break if end is reached
        if *header == "$EndNodes" { break }

        // Parse block header
        let [dim, entity_tag, parametric, num_nodes] = header
            .split_whitespace()
            .map(|str| str.parse::<usize>().expect("First line of a node block must only contain integers"))
            .collect_array()
            .expect("First line must contain exactly four integers");

        // Collect vertices
        let block_verts = (&mut nodes_iter)
            .skip(num_nodes)
            .take(num_nodes)
            .map(|line| {
                let coords = line.split_whitespace()
                    .map(|num| num.parse::<T>().expect("Vertex coordinates must be of type `T`"))
                    .next_array::<M>() // todo: this is used instead of collect_array, to just remove the z-component. Change?
                    .unwrap_or_else(|| panic!("Vertices must be of M = {M} dimensions"));

                Point::from(coords)
            });
        verts.extend(block_verts);
    }

    // $Elements
    // numEntityBlocks(size_t) numElements(size_t)
    //  minElementTag(size_t) maxElementTag(size_t)
    // entityDim(int) entityTag(int) elementType(int; see below)
    //  numElementsInBlock(size_t)
    //  elementTag(size_t) nodeTag(size_t) ...
    // ...
    //     ...
    // $EndElements

    // PARSE ELEMENTS
    // Header
    let elems_start_idx = lines.iter()
        .position(|line| line.starts_with("$Elements"))
        .expect("$Elements must be defined");

    let header = lines[elems_start_idx + 1];
    let [num_entity_bocks, num_elems, min_elem, max_elem] = header.split_whitespace()
        .map(|str| str.parse::<usize>().expect("First line of $Nodes must only contain integers"))
        .collect_array()
        .expect("First line must contain exactly four integers");

    // Parse each element block
    let mut quads = Vec::<QuadNodes>::with_capacity(num_elems);
    let mut elems_iter = lines.iter().skip(elems_start_idx + 2);

    while let Some(header) = elems_iter.next() {
        // Break if end is reached
        if *header == "$EndElements" { break }

        // Parse block header
        let [dim, entity_tag, elem_type, num_elems] = header
            .split_whitespace()
            .map(|str| str.parse::<usize>().expect("First line of an element block must only contain integers"))
            .collect_array()
            .expect("First line of an element block must contain exactly four integers");

        if elem_type == 3 { // For quad elements (`elem_type` = 3), collect nodes
            let block_quads = (&mut elems_iter)
                .take(num_elems)
                .map(|line| {
                    let [_quad_idx, n1, n2, n3, n4] = line.split_whitespace()
                        .map(|idx| idx.parse::<usize>().expect("Element block must contain only integers") - 1)
                        .collect_array()
                        .expect("Quad must be have exactly four nodes");

                    QuadNodes([n1, n2, n3, n4])
                });

            quads.extend(block_quads);
        } else { // Else skip element block
            elems_iter.nth(num_elems-1);
        }
    }

    Ok(QuadVertexMesh::new(verts, quads))
}