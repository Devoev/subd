use crate::diffgeo::chart::{Chart, ChartAllocator};
use crate::quadrature::pullback::DimMinSelf;
use crate::space::basis::BasisFunctions;
use nalgebra::{DefaultAllocator, DimName, Scalar};

/// A [`ChartAllocator`] for the [`C::GeoMap`] of an element.
pub trait ElemAllocator<T: Scalar, C: Element<T>>: ChartAllocator<T, C::GeoMap>
    where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}

impl<T: Scalar, C: Element<T>> ElemAllocator<T, C> for DefaultAllocator
    where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}

/// A geometrical element embedded into Euclidean space.
pub trait Element<T: Scalar>: Sized
    where DefaultAllocator: ElemAllocator<T, Self>
{
    /// Reference element in the parametric domain for the mapping.
    type ParametricElement;

    /// Parametrization of this element.
    type GeoMap: Chart<T>;

    /// Returns the parametric element of this `self`.
    fn parametric_element(&self) -> Self::ParametricElement;

    /// Returns the parametrization of `self`.
    fn geo_map(&self) -> Self::GeoMap;
    
    // todo: maybe merge Chart and Cell traits?
}

/// Coordinate of the [`Element::GeoMap`] of `Elem`.
pub type ElemCoord<T, Elem> = <<Elem as Element<T>>::GeoMap as Chart<T>>::Coord;

/// The geometrical dimension of the `Elem`.
pub type ElemDim<T, Elem> = <<Elem as Element<T>>::GeoMap as Chart<T>>::GeometryDim;

/// Volumetric element where the [parametric dimension][Chart::ParametricDim]
/// equals the [geometric dimension][Chart::GeometryDim].
pub trait VolumeElement<T: Scalar>: Element<T, GeoMap: Chart<
    T,
    GeometryDim: DimMinSelf, // todo: this bound is required for Jacobian determinant. Add this in the future
    ParametricDim = <Self::GeoMap as Chart<T>>::GeometryDim
>>
    where DefaultAllocator: ElemAllocator<T, Self> {}

impl <T, Elem> VolumeElement<T> for Elem 
    where T: Scalar, 
          Elem: Element<T, GeoMap: Chart<
              T,
              GeometryDim: DimMinSelf,
              ParametricDim = <Self::GeoMap as Chart<T>>::GeometryDim
          >>,
          DefaultAllocator: ElemAllocator<T, Self>
{}

/// Constrains `Self` to have a chart with coordinate [`B::Coord`].
pub trait HasBasisCoord<T: Scalar, B: BasisFunctions>: Element<T, GeoMap: Chart<T, Coord = B::Coord<T>>>
    where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}

impl <T: Scalar, B: BasisFunctions, Elem: Element<T, GeoMap: Chart<T, Coord = B::Coord<T>>>> HasBasisCoord<T, B> for Elem
where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}

/// Constrains `Self` to have a chart with both parametric and geometry dimension [`D`].
pub trait HasDim<T: Scalar, D: DimName>: Element<T, GeoMap: Chart<T, ParametricDim = D, GeometryDim = D>>
where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}

impl <T: Scalar, D: DimName, Elem: Element<T, GeoMap: Chart<T, ParametricDim = D, GeometryDim = D>>> HasDim<T, D> for Elem
where DefaultAllocator: ChartAllocator<T, Self::GeoMap>
{}