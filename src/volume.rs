//! Contains the logic for specifying bulk [`VoxelData`].

use bevy_math::{IVec3, UVec3};

use crate::voxel::VoxelData;

/// To save memory we store [`VoxelData`] in a sparse tree per chunk.
/// This is a node in that tree.
#[derive(PartialEq, Eq, Clone, Copy)]
struct ChunkNode {
    voxel: VoxelData,
    /// Stores the index to the ldb child of this node (1 of 8).
    /// The children are stored together in chunk data memory.
    ///
    /// This may also be a high value for no children.
    ///
    /// # Safety
    ///
    /// This index must be correct.
    children: u32,
}

impl ChunkNode {
    const NO_CHILDREN_EXACT: u32 = u32::MAX;
    const NO_CHILDREN_APPROX: u32 = u32::MAX - 1;

    #[inline]
    fn is_approx(self) -> bool {
        self.children == Self::NO_CHILDREN_APPROX
    }

    #[inline]
    fn is_exact(self) -> bool {
        self.children == Self::NO_CHILDREN_EXACT
    }

    #[inline]
    fn has_children(self) -> bool {
        self.children < Self::NO_CHILDREN_APPROX
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

    /// Sets the `voxel` data at this position `relative_to_ldb` at this `lod`.
    /// The lod must be less than or equal to [`Chunk::CHUNK_NODE_DEPTH`] of course.
    fn set_voxel_within(&mut self, relative_to_ldb: UVec3, voxel: VoxelData, lod: u32) {
        let mut inverse_depth = Self::CHUNK_NODE_DEPTH;
        // SAFETY: This must always be valid.
        let mut node_index = 0;
        while inverse_depth > lod {
            // SAFETY: node_index is always valid.
            let node = unsafe { self.data.get_unchecked_mut(node_index) };

            let children_index = if !node.has_children() {
                let child_data = *node;
                let children_index = self.data.len();
                for _ in 0..8 {
                    self.data.push(child_data);
                }

                // SAFETY: node_index is always valid.
                let node = unsafe { self.data.get_unchecked_mut(node_index) };
                node.children = children_index as u32;

                children_index
            } else {
                node.children as usize
            };

            let px = relative_to_ldb.x & inverse_depth >> (inverse_depth - 1);
            let py = relative_to_ldb.y & inverse_depth >> (inverse_depth - 1);
            let pz = relative_to_ldb.z & inverse_depth >> (inverse_depth - 1);
            let child_offset = px | (py << 1) | (pz << 2);
            inverse_depth >>= 1;

            // SAFETY: These indices are correct
            node_index = children_index + child_offset as usize;
        }

        // SAFETY: node_index is always valid.
        let node = unsafe { self.data.get_unchecked_mut(node_index) };
        node.voxel = voxel;
    }
}
