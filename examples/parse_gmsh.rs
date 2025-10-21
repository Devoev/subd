//! This example parses a 2D quadrilateral mesh of an electron gun
//! from the file `egun.msh` in a Gmsh format using [`parse_gmsh_quad_mesh`].

use subd::io::gmsh::parse_gmsh_quad_mesh;
use subd::mesh::face_vertex::QuadVertexMesh;

fn main() {
    // Parse Electron gun mesh from Gmsh file
    let path = "examples/egun.msh";
    let msh: QuadVertexMesh<f64, 2> = parse_gmsh_quad_mesh(path).expect("`egun.msh` should exist in the examples/ folder");
    dbg!(msh);
}