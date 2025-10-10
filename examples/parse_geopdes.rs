//! This example parses a 2D NURBS representation of an electron gun
//! from the file `egun_geo.txt` in a GeoPDEs format using [`parse_geopdes_nurbs`].

use subd::io::geopdes::parse_geopdes_nurbs;

fn main() {
    let path = "examples/egun_geo.txt";
    let patches = parse_geopdes_nurbs(path).unwrap();

    for patch in patches {
        let (name, degrees, nums, knots, control_points, weights) = patch;
        println!("Patch '{name}' of degrees {degrees:?} with knot vectors {knots:?}, weights = {weights:?} and control points = {control_points}");
    }
}