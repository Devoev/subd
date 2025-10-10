use crate::knots::knot_vec::KnotVec;
use itertools::Itertools;
use nalgebra::DMatrix;
use std::path::Path;
use std::{fs, io};

/// Single NURBS Patch.
type Patch = (String, Vec<usize>, Vec<usize>, Vec<KnotVec<f64>>, DMatrix<f64>, Vec<f64>);

/// Parses a NURBS multipatch geometry from a file `path`
/// in the [GeoPDEs](https://rafavzqz.github.io/geopdes/) format `v.2.1`.
///
/// The format is detailed [here](https://github.com/rafavzqz/geopdes/blob/master/geopdes/doc/geo_specs_mp_v21.txt).
pub fn parse_geopdes_nurbs(path: impl AsRef<Path>) -> io::Result<Vec<Patch>> {
    let str = fs::read_to_string(path)?;
    let mut lines = str.lines()
        .filter(|line| !line.starts_with("#")); // filter out comments starting with '#'

    // Parse parameters from first line
    let [parametric_dim, geo_dim, num_patches, num_interfaces, num_subdomains] = lines.next()
        .expect("File must contain more than one line")
        .split_whitespace()
        .map(|str| str.parse::<usize>().expect("First line must only contain integers"))
        .collect_array()
        .expect("First line must contain exactly five integers");

    // Parse patches
    let mut patches = Vec::<Patch>::with_capacity(num_patches);
    let len_patch = 4 + parametric_dim + geo_dim;
    for mut chunk_lines in &lines.by_ref().take(num_patches * len_patch).chunks(len_patch) {
        // Parse name of the patch
        let name = chunk_lines.next()
            .expect("Patch must have a name")
            .to_string();

        // Parse degrees in parametric space
        let degrees = chunk_lines.next()
            .expect("Degrees for the mapping must be defined")
            .split_whitespace()
            .map(|num| num.parse::<usize>().expect("Degrees must be a integers"))
            .collect_vec();

        // Parse number of control points
        let num_control_points = chunk_lines.next()
            .expect("Numbers of control points for the mapping must be defined")
            .split_whitespace()
            .map(|num| num.parse::<usize>().expect("Numbers of control points must be a integers"))
            .collect_vec();

        let num_control_points_total = num_control_points.iter().product();

        // Parse knot vectors
        let knots = chunk_lines.by_ref()
            .take(parametric_dim)
            .map(|line| {
                let knots = line.split_whitespace()
                    .map(|num| num.parse::<f64>().expect("Knots must be floats"))
                    .collect_vec();
                KnotVec::new(knots).expect("Knot vectors must be sorted")
            })
            .collect_vec();

        // Parse control points
        let control_points_iter = chunk_lines.by_ref()
            .take(parametric_dim)
            .flat_map(|line| {
                line.split_whitespace()
                    .map(|num| num.parse::<f64>().expect("Control points must be floats"))
            });
        let control_points = DMatrix::from_row_iterator(geo_dim, num_control_points_total, control_points_iter);

        // Parse weights
        let weights = chunk_lines.next()
            .expect("Weights for the mapping must be defined")
            .split_whitespace()
            .map(|num| num.parse::<f64>().expect("Weights must be floats"))
            .collect_vec();

        // Collect patch
        patches.push((name, degrees, num_control_points, knots, control_points, weights))
    }

    // todo: parse interfaces and boundary as well

    Ok(patches)
}