//! Contains logic for what a terrain may look like.

use bevy_math::IVec3;

use crate::voxel::VoxelData;

/// Represents a type that can dictate the shape of a terrain.
pub trait TerrainGenerator {
    /// Samples the terrain at `location` in a radius of `inverse_radius` to get an approximation of the [`VoxelData`] within that radius.
    fn sample_voxel_group(&self, location: IVec3, inverse_radius: f32) -> VoxelData;
}
