//! This module defines what a voxel is, etc.

use arrayvec::ArrayVec;

/// Represents a distinct layer of voxel data. Ex: Void, Air, Liquid, Solid, etc.
///
/// Layers that are greater than others will appear on top of lesser ones. Ex: Water will appear on top of solids.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VoxelLayer(u8);

impl VoxelLayer {
    /// The highest layer; this layer can not represent a surface.
    /// Use this for air, space, etc.
    pub const VOID: Self = Self(u8::MAX);
    /// The highest layer that forms a surface against [`Self::VOID`].
    pub const HIGHEST_SURFACE: Self = Self(u8::MAX - 1);
}

/// Represents a distinct voxel material per layer. Ex: Stone, Dirt, Grass.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct VoxelMaterial(u16);

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
    pub fn fullness_unorm(self) -> f32 {
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
    pub fn fullness_percent(self) -> f32 {
        self.fullness_unorm() * 0.5 + 0.5
    }

    /// Gets the fullness between 0 and 255, 0 meaning 50% full and 255 meaning 100% full.
    pub fn fullness_quantized(self) -> u8 {
        self.0
    }
}

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
    /// Gets the [`VoxelFullness`] of the voxel.
    pub fn fullness(self) -> VoxelFullness {
        VoxelFullness((self.0 >> 24) as u8)
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
                mats.push_unchecked((mat, 0));
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
            } else if layer > next_highest_layer {
                next_highest_layer = layer;
                num_next_highest_layer = 0;
                next_highest_layer_mats.clear();
            }

            if layer == highest_layer {
                num_highest_layer += 1;
                highest_layer_fullness += c.0 >> 24;
                include_mat(c.material(), &mut highest_layer_mats);
            } else if layer == next_highest_layer {
                num_next_highest_layer += 1;
                next_highest_layer_fullness += c.0 >> 24;
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
        let fullness = total_fullness / layer_num;

        Self((fullness << 24) | ((mat.0 as u32) << 8) | layer.0 as u32)
    }
}
