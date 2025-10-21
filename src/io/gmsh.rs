use std::{fs, io};
use std::fmt::Debug;
use std::path::Path;
use std::str::FromStr;
use itertools::Itertools;
use nalgebra::{Point, Scalar};
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

    // Parse nodes
    let nodes_block_start_idx = lines.iter()
        .position(|line| line.starts_with("$Nodes"))
        .expect("$Nodes must be defined");

    let header = lines[nodes_block_start_idx + 1];
    let [num_entity_bocks, num_nodes, min_node, max_node] = header.split_whitespace()
        .map(|str| str.parse::<usize>().expect("First line of $Nodes must only contain integers"))
        .collect_array()
        .expect("First line must contain exactly four integers");

    dbg!(num_entity_bocks, num_nodes, min_node, max_node);
    let mut verts = Vec::<Point<T, M>>::with_capacity(num_nodes);

    let mut nodes_iter = lines.iter().skip(nodes_block_start_idx + 2);

    // todo: repeat the process below until `nodes_iter` is exhausted
    //  can this be done using nodes_iter.len() != 0?

    let [dim, entity_tag, parametric, num_nodes] = nodes_iter.next()
        .expect("")
        .split_whitespace()
        .map(|str| str.parse::<usize>().expect("First line of $Nodes must only contain integers"))
        .collect_array()
        .expect("First line must contain exactly four integers");

    let vertices = nodes_iter
        .skip(num_nodes)
        .take(num_nodes)
        .map(|line| {
            let coords = line.split_whitespace()
                .map(|num| num.parse::<T>().expect("Vertex coordinates must be of type `T`"))
                .next_array::<M>() // todo: this is used instead of collect_array, to just remove the z-component. Change?
                .unwrap_or_else(|| panic!("Vertices must be of M = {dim} dimensions", dim = M));

            Point::from(coords)
        });

    verts.extend(vertices);

    dbg!(verts);

    todo!()
}