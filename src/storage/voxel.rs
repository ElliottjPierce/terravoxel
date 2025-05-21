//! This module defines what a voxel is, etc.

use core::fmt::{Debug, Formatter};

use arrayvec::ArrayVec;
use bevy_math::{IVec3, UVec3};

/// Represents a distinct layer of voxel data. Ex: Void, Air, Liquid, Solid, etc.
///
/// Only 7 bits may represent this value,
/// and the highest layer [`VOID`](Self::VOID) doesn't correspond to a surface.
/// That gives 128 layers and 127 surfaces.
/// This is represented internally as a [`u8`] where the highest bit is always off.
///
/// Layers that are greater than others will appear on top of lesser ones. Ex: Water will appear on top of solids.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VoxelLayer(u8);

impl VoxelLayer {
    /// The highest layer; this layer can not represent a surface.
    /// Use this for air, space, etc.
    pub const VOID: Self = Self(0x7f);
    /// The highest layer that forms a surface against [`Self::VOID`].
    pub const HIGHEST_SURFACE: Self = Self(Self::VOID.0 - 1);

    /// Constructs the layer from a `u32` if the layer is within range.
    pub const fn from_u32(layer: u32) -> Option<VoxelLayer> {
        if layer > 0x7f {
            None
        } else {
            Some(Self(layer as u8))
        }
    }

    /// Gets the layer as a `u32` id/index.
    pub const fn index(self) -> u32 {
        self.0 as u32
    }
}

impl Debug for VoxelLayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        if *self == Self::VOID {
            f.write_str("Void")
        } else {
            write!(f, "LayerId={}", self.index())
        }
    }
}

/// Represents a distinct voxel material per layer. Ex: Stone, Dirt, Grass.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VoxelMaterial(u16);

impl VoxelMaterial {
    /// Constructs the material from a `u32` id if the id is within range.
    pub const fn from_u32(material: u32) -> Option<VoxelMaterial> {
        if material > 0xFFFF {
            None
        } else {
            Some(Self(material as u16))
        }
    }

    /// Gets the material as a `u32` id/index.
    pub const fn index(self) -> u32 {
        self.0 as u32
    }
}

impl Debug for VoxelMaterial {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "MaterialId={}", self.index())
    }
}

/// Represents how full a voxel is with a quantized [`u8`].
/// This "fullness" is mainly about how close this voxel is to the nearest higher layer surface.
///
/// The voxel is always at least half full; otherwise it would have a different material.
/// So this fullness value from 0 to 1, really represents a range from 0.5 to 1 (Ex: 0.5 means 75% full).
/// For example, a grass voxel near air that is 25% grass and 75% air gets an air material and a fullness of 0.5.
///
/// Note that when determining where a surface is between two voxels,
/// it is at the position of the lower layer voxel,
/// lerped to the position of the higher layer voxel,
/// by the fullness of the lower layer voxel.
///
/// As an example, imagine an air voxel followed by two dirt voxels.
/// The air voxel is 100% full since there is no layer above it.
/// The first dirt voxel may be 75% full, and the next one 100% since it is deeper than the first.
/// As the first dirt voxel is dug out, it's fullness drops to 70%, 65%, etc, and eventually hits 50%, where the surface now goes directly though the center of the first dirt voxel.
/// As it is dips bellow 50%, it switches immediately to a 100% full air voxel.
/// But because the dirt voxel beneath it is 100% full, the surface's location has barely moved.
/// As the second dirt voxel is dug more, the fullness decreases, moving from the center of the first dirt voxel, to the center of the second one.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VoxelFullness(u8);

impl VoxelFullness {
    /// Gets the fullness from 0 to 1, 0 meaning 50% full and 1 meaning 100% full.
    /// You can think of this as the distance to the nearest higher layer surface scaled to the diameter of the voxel, and clamped to (0, 1).
    #[inline]
    pub const fn fullness_unorm(self) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 8 value bits    15 bits as precision padding.
        #[expect(
            clippy::unusual_byte_groupings,
            reason = "This shows what the bits mean."
        )]
        const BASE_VALUE: u32 = 0b_0_01111111_00000000_011111111111111;
        let result_in_1_to_2 = BASE_VALUE | ((self.0 as u32) << 15);
        f32::from_bits(result_in_1_to_2) - 1.0
    }

    /// Gets the fullness as a percent between 0 and 1, 0 meaning 0% full and 1 meaning 100% full.
    pub const fn fullness_percent(self) -> f32 {
        self.fullness_unorm() * 0.5 + 0.5
    }

    /// Gets the fullness between 0 and 255, 0 meaning 50% full and 255 meaning 100% full.
    #[inline]
    pub const fn fullness_quantized(self) -> u8 {
        self.0
    }

    /// Creates the [`VoxelFullness`] from its [quantized](Self::fullness_quantized) value.
    #[inline]
    pub const fn from_quantized(fullness: u8) -> Self {
        Self(fullness)
    }
}

impl Debug for VoxelFullness {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} ({})",
            self.fullness_quantized(),
            self.fullness_percent()
        )
    }
}

/// Voxel data specifies what a voxel is in a few ways:
/// Every voxel has a material (ex: grass), fullness (ex: 75%), and a layer (ex: solid).
///
/// [`VoxelData`] may refer to the data for a particular voxel,
/// the average data for a group of voxels, or the exact data for a group of voxels.
/// When it represents an exact voxel, it represents the data of the center of that voxel.
///
/// Internally, this is stored, in order of most to least significant, as:
///
/// - 1 skipped bit, (for flags elsewhere but *always* off here)
/// - 7 layer bits, (so it can be accessed with one bit shift)
/// - 16 material bits, (since it's the least accessed)
/// - 8 fullness bits. (since it can be accessed with one bit and)
#[repr(transparent)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct VoxelData(u32);

impl VoxelData {
    pub(crate) const PLACEHOLDER: Self = Self(0);
    pub(crate) const RESERVED_BIT: u32 = 1 << 31;
    const LAYER_SHIFT: u32 = 24;
    const MAT_SHIFT: u32 = 8;
    const FULLNESS_BITS: u32 = 0xFF;

    /// Gets the [`VoxelFullness`] of the voxel.
    #[inline]
    pub const fn fullness(self) -> VoxelFullness {
        VoxelFullness(self.0 as u8)
    }

    /// Gets the [`VoxelLayer`] of the voxel.
    #[inline]
    pub const fn layer(self) -> VoxelLayer {
        VoxelLayer((self.0 >> Self::LAYER_SHIFT) as u8)
    }

    /// Gets the [`VoxelMaterial`] of the voxel.
    #[inline]
    pub const fn material(self) -> VoxelMaterial {
        VoxelMaterial((self.0 >> Self::MAT_SHIFT) as u16)
    }

    /// Gets the raw bits of this voxel data.
    #[inline]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Constructs a [`VoxelData`] from its bits.
    /// If the bits are not valid, this returns `None`.
    #[inline]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & Self::RESERVED_BIT > 0 {
            None
        } else {
            Some(Self(bits))
        }
    }

    /// Constructs a [`VoxelData`] from these bits.
    /// This does not validate the bits.
    /// Passing incorrect bits does not produce UB, but is a nasty logic error.
    #[inline]
    pub const fn from_bits_unchecked(bits: u32) -> Self {
        Self(bits)
    }

    /// Constructs a [`VoxelData`] from its bits.
    /// If the bits are invalid, this simply makes them valid.
    #[inline]
    pub const fn force_from_bits(bits: u32) -> Self {
        Self(bits & !Self::RESERVED_BIT)
    }

    /// Constructs a [`VoxelData`] from its parts. Keep in mind that the material is per layer.
    #[inline]
    pub const fn new(layer: VoxelLayer, mat: VoxelMaterial, fullness: VoxelFullness) -> Self {
        Self::new_by_bits(layer.0 as u32, mat.0 as u32, fullness.0 as u32)
    }

    /// Constructs a [`VoxelData`] from its parts, where each part is in raw `u32` format, within the right number of bits and in the least significant bits.
    #[inline]
    const fn new_by_bits(layer_bits: u32, mat_bits: u32, fullness_bits: u32) -> Self {
        Self((layer_bits << Self::LAYER_SHIFT) | (mat_bits << Self::MAT_SHIFT) | fullness_bits)
    }

    /// Constructs a [`VoxelData`] given its layer, material within that layer, inverse radius (reciprocal), and distance to the nearest surface of the upper layer.
    /// The distance and radius are assumed to be positive.
    pub const fn from_surface_distance(
        distance_to_upper_surface: f32,
        inverse_diameter_of_voxel: f32,
        layer: VoxelLayer,
        material_within_layer: VoxelMaterial,
    ) -> Self {
        let fullness = (distance_to_upper_surface * inverse_diameter_of_voxel).min(1.0);
        let fullness_bits = (fullness * 255.0) as u32;
        Self::new_by_bits(
            layer.0 as u32,
            material_within_layer.0 as u32,
            fullness_bits,
        )
    }

    /// Creates an approximate [`VoxelData`] for this cluster of data.
    fn approximate(cluster: [Self; 8]) -> Self {
        let mut highest_layer = VoxelLayer(0);
        let mut next_highest_layer = VoxelLayer(0);

        let mut num_highest_layer = 0;
        let mut num_next_highest_layer = 0;

        let mut highest_layer_fullness = 0;
        let mut next_highest_layer_fullness = 0;

        let mut highest_layer_mats = ArrayVec::<(VoxelMaterial, u8), 8>::new_const();
        let mut next_highest_layer_mats = ArrayVec::<(VoxelMaterial, u8), 8>::new_const();
        fn include_mat(mat: VoxelMaterial, mats: &mut ArrayVec<(VoxelMaterial, u8), 8>) {
            for (m, num) in mats.iter_mut() {
                if *m == mat {
                    *num += 1;
                    return;
                }
            }
            // SAFETY: The capacity of the cluster is known.
            unsafe {
                mats.push_unchecked((mat, 1));
            }
        }

        for c in cluster {
            let layer = c.layer();
            if layer > highest_layer {
                next_highest_layer = highest_layer;
                highest_layer = layer;
                num_next_highest_layer = num_highest_layer;
                num_highest_layer = 0;
                next_highest_layer_mats = highest_layer_mats.take();
                next_highest_layer_fullness = highest_layer_fullness;
                highest_layer_fullness = 0;
            } else if layer > next_highest_layer && layer < highest_layer {
                next_highest_layer = layer;
                num_next_highest_layer = 0;
                next_highest_layer_fullness = 0;
                next_highest_layer_mats.clear();
            }

            if layer == highest_layer {
                num_highest_layer += 1;
                highest_layer_fullness += c.0 & Self::FULLNESS_BITS;
                include_mat(c.material(), &mut highest_layer_mats);
            } else if layer == next_highest_layer {
                num_next_highest_layer += 1;
                next_highest_layer_fullness += c.0 & Self::FULLNESS_BITS;
                include_mat(c.material(), &mut next_highest_layer_mats);
            };
        }

        // The 2 is arbitrary but if we don't do this, a tiny air pocket could consume the whole approximation, all the way up to the root.
        // But in general, we want to use the average of the highest layer. Ex: in an ocean, distant chunks should not be sand, but water.
        // If we only used the most common layer, there might be spots of land that didn't actually exist.
        let (layer, total_fullness, layer_num, mats) = if num_highest_layer > 2 {
            (
                highest_layer,
                highest_layer_fullness,
                num_highest_layer,
                highest_layer_mats,
            )
        } else {
            (
                next_highest_layer,
                next_highest_layer_fullness,
                num_next_highest_layer,
                next_highest_layer_mats,
            )
        };

        let mat = mats.into_iter().max_by_key(|i| i.1);
        // SAFETY: This can't be empty because ultimately we know the cluster is not empty.
        let mat = unsafe { mat.unwrap_unchecked().0 };
        // This can't overflow beyond a u8 since it is an average.
        let fullness = total_fullness / layer_num;

        Self::new_by_bits(layer.0 as u32, mat.0 as u32, fullness)
    }
}

impl Debug for VoxelData {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "VoxelData {{ layer: {0:?}, material: {1:?}, fullness: {2:?}, }} (bits: {3})",
            self.layer(),
            self.material(),
            self.fullness(),
            self.0
        )
    }
}

/// Represents the location of a voxel.
///
/// Internally, it is more efficient to use unsigned coordinates, but externally, signed coordinates are more convenient.
/// This type bridges that gap.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct VoxelLocation(UVec3);

impl VoxelLocation {
    const MAPPER: u32 = 1 << 31;

    /// Gets the [`VoxelLocation`] of the voxel at `loc`.
    pub fn of_location(loc: IVec3) -> Self {
        Self(loc.as_uvec3() ^ Self::MAPPER)
    }

    /// Returns the location of the voxel as a [`IVec3`]
    #[inline]
    pub fn location(self) -> IVec3 {
        (self.0 ^ Self::MAPPER).as_ivec3()
    }

    /// Returns the location of the voxel mapped to a [`UVec3`]
    #[inline]
    pub fn location_mapped(self) -> UVec3 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_bits() {
        let layer = 123;
        let mat = 19873;
        let fullness = 127;

        let voxel = VoxelData::new(
            VoxelLayer::from_u32(layer).unwrap(),
            VoxelMaterial::from_u32(mat).unwrap(),
            VoxelFullness(fullness),
        );
        let voxel_bits = voxel.to_bits();
        assert_eq!(voxel, VoxelData::from_bits(voxel_bits).unwrap());

        assert_eq!(voxel.fullness().fullness_quantized(), fullness);
        assert_eq!(voxel.material().index(), mat);
        assert_eq!(voxel.layer().index(), layer);
    }

    #[test]
    fn reasonable_estimate() {
        let voxels = [
            VoxelData::new_by_bits(1, 2, 200),
            VoxelData::new_by_bits(4, 0, 125),
            VoxelData::new_by_bits(4, 1, 50),
            VoxelData::new_by_bits(4, 1, 200),
            VoxelData::new_by_bits(4, 1, 100),
            VoxelData::new_by_bits(4, 2, 75),
            VoxelData::new_by_bits(6, 1, 200),
            VoxelData::new_by_bits(6, 1, 200),
        ];
        let estimate = VoxelData::approximate(voxels);
        assert_eq!(estimate, VoxelData::new_by_bits(4, 1, 110));

        let voxels = [
            VoxelData::new_by_bits(4, 0, 255),
            VoxelData::new_by_bits(1, 2, 255),
            VoxelData::new_by_bits(4, 1, 255),
            VoxelData::new_by_bits(6, 1, 255),
            VoxelData::new_by_bits(4, 1, 255),
            VoxelData::new_by_bits(4, 1, 255),
            VoxelData::new_by_bits(4, 2, 255),
            VoxelData::new_by_bits(6, 1, 255),
        ];
        let estimate = VoxelData::approximate(voxels);
        assert_eq!(estimate, VoxelData::new_by_bits(4, 1, 255));
    }
}
