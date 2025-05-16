//! Contains the logic for specifying bulk [`VoxelData`].

use bevy_math::{IVec3, UVec3};

use crate::voxel::VoxelData;

/// To save memory we store [`VoxelData`] in a sparse tree per chunk.
/// This is a node in that tree.
///
/// Each node may or may not have children and may or may not be exact.
///
/// | children? |is exact| is not exact |
/// |---|---|---|
/// | has children | This is the average/representative of some children data. | This is a guess of this data, but children may be more precise. |
/// | has no children | This is the data of all would-be children. | This is the best available guess of all would-be children; more samples should be made as needed. |
///
#[derive(PartialEq, Eq, Clone, Copy)]
struct ChunkNode {
    /// Represents some [`VoxelData`] if the high bit is on, this is only an approximation/guess.
    /// Otherwise, this is exact.
    voxel_data: u32,
    /// Stores the index to the ldb child of this node (1 of 8).
    /// The children are stored together in chunk data memory.
    ///
    /// This may also be 0 for no children.
    /// If there are no children, this data is exactly or approximately that of all children.
    ///
    /// # Safety
    ///
    /// This index must be correct.
    children: u32,
}

impl ChunkNode {
    #[inline]
    fn has_children(self) -> bool {
        self.children > 0
    }

    #[inline]
    fn is_approximate(self) -> bool {
        VoxelData::is_reserved_bit_on(self.voxel_data)
    }

    #[inline]
    fn voxel(self) -> VoxelData {
        VoxelData::force_from_bits(self.voxel_data)
    }

    #[inline]
    fn set_voxel_data(&mut self, data: VoxelData, is_approximate: bool) {
        let bits = data.to_bits();
        if is_approximate {
            self.voxel_data = VoxelData::with_reserved_bit_on(bits);
        } else {
            self.voxel_data = bits;
        }
    }
}

struct Chunk {
    ldb_loc: IVec3,
    /// # Safety
    ///
    /// This must always have the root node at index 0.
    data: Vec<ChunkNode>,
}

impl Chunk {
    const CHUNK_SIZE: u32 = 1 << Self::CHUNK_NODE_DEPTH;
    const CHUNK_NODE_DEPTH: u32 = 8;
}
