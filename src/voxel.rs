//! This module defines what a voxel is, etc.

/// Represents a distinct layer of voxel data. Ex: Void, Air, Liquid, Solid, etc.
///
/// Layers that are greater than others will appear on top of lesser ones. Ex: Water will appear on top of solids.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VoxelLayer(u8);

/// Represents a distinct voxel material per layer. Ex: Stone, Dirt, Grass.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VoxelMaterial(u16);

/// Voxel data specifies what a voxel is in a few ways:
/// Every voxel has a material (ex: grass), fullness (ex: 75%), and a layer (ex: solid).
///
/// [`VoxelData`] may refer to the data for a particular voxel,
/// the average data for a group of voxels, or the exact data for a group of voxels.
/// When it represents an exact voxel, it represents the data of the center of that voxel.
///
/// Internally, this is stored as 8 fullness bits, 16 material, and 8 layer bits in order of most to least significant.
#[repr(transparent)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct VoxelData(u32);

impl VoxelData {
    /// Gets the fullness of this voxel.
    ///
    /// The voxel is always at least half full; otherwise it would have a different material.
    /// So this fullness value from 0 to 1, really represents a range from 0.5 to one (Ex: 0.5 means 75% full).
    /// For example, a grass voxel near air that is 25% grass and 75% air gets an air material and a fullness of 0.5.
    pub fn fullness(self) -> f32 {
        /// The base value bits for the floats we make.
        /// Positive sign, exponent of 0    , 8 value bits    15 bits as precision padding.
        #[expect(
            clippy::unusual_byte_groupings,
            reason = "This shows what the bits mean."
        )]
        const BASE_VALUE: u32 = 0b_0_01111111_00000000_011111111111111;
        let fullness_bits = self.0 >> 24;
        let result_in_1_to_2 = BASE_VALUE | (fullness_bits << 15);
        f32::from_bits(result_in_1_to_2) - 1.0
    }

    /// Gets the [`VoxelLayer`] of the voxel.
    pub fn layer(self) -> VoxelLayer {
        VoxelLayer(self.0 as u8)
    }

    /// Gets the [`VoxelMaterial`] of the voxel.
    pub fn material(self) -> VoxelMaterial {
        VoxelMaterial((self.0 >> 8) as u16)
    }

    /// Gets the raw bits of this voxel data.
    pub fn to_bits(self) -> u32 {
        self.0
    }

    /// Constructs a [`VoxelData`] from its bits.
    /// All bits are valid.
    pub fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Constructs a [`VoxelData`] given its layer, material within that layer, inverse radius (reciprocal), and distance to the nearest surface of the upper layer.
    /// The distance and radius are assumed to be positive.
    pub fn from_surface_distance(
        distance_to_upper_surface: f32,
        inverse_radius_of_voxel: f32,
        layer: VoxelLayer,
        material_within_layer: VoxelMaterial,
    ) -> Self {
        let fullness = (distance_to_upper_surface * inverse_radius_of_voxel).min(1.0);
        let fullness_bits = (fullness * 255.0) as u32;
        let layer_bits = layer.0 as u32;
        let mat_bits = material_within_layer.0 as u32;
        Self((fullness_bits << 24) | (mat_bits << 8) | layer_bits)
    }
}
