//! Contains the logic for specifying bulk [`VoxelData`].

use core::{num::NonZero, u32};

use bevy_math::IVec3;
use bevy_platform::prelude::*;

use crate::storage::voxel::VoxelData;

/// Represents a level of detail or layer of octree.
/// This value must be in range 0..=[`Lod::MAX`], 0 meaning an exact voxel, max meaning an entire chunk.
/// A [`Lod`] that is greater than another represents a bigger area and will be less detailed/precise.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Lod(u8);

impl Lod {
    /// There are 9 levels of detail; this is the max.
    pub const MAX: Self = Self(8);
    /// There are 9 levels of detail; this is the max.
    pub const MIN: Self = Self(0);

    /// The length of the cube that this lod would cover.
    pub const fn length(self) -> u32 {
        1u32 << self.0 as u32
    }
}

/// To save memory we store [`VoxelData`] in a sparse voxel octree per chunk.
/// This is a node in that tree.
///
/// This needs to store a few things.
/// If it is a leaf node:
///
/// - The [`VoxelData`] that best represents its cube,
/// - To what [`Lod`] is that [`VoxelData`] exact (when should we ask for a more specific one).
///
/// If it is a parent node:
///
/// - The [`VoxelData`] that best represents the children via [`VoxelData::approximate`],
/// - Whether or not that [`VoxelData`] approximation is up to date (It's a useful cache, but we only update it as needed),
/// - The index of the first of the 8 children, which will be stored next to each other.
///
/// This can also represent a "free" node, which is void and pending reuse.
#[derive(PartialEq, Eq, Clone, Copy)]
struct ChunkNode {
    /// The voxel data for this cube.
    voxel_data: VoxelData,
    /// This contains most other metadata about the node.
    /// The least significant 25 bits store the index to the child nodes.
    /// Since the root node is always index 0, if these 25 bits are 0, there are no children.
    /// The other 7 bits in order of most to least significant are as follows:
    ///
    /// If it is a leaf node:
    ///
    /// - 4 bits for the [`Lod`] to which this [`VoxelData`] is precise to (If this lod or higher is requested, we can use this data exactly),
    /// - 3 bits currently unused,
    ///
    /// If it is a parent node:
    ///
    /// - 1 bit for if the [`VoxelData`] has been dirtied or not (If it has, we will need to redo [`VoxelData::approximate`]),
    /// - 6 bits currently unused,
    ///
    /// If this is nod a node and is "free", this just stores the index to the next free node, or 0 if there is no free node.
    ///
    /// # Safety
    ///
    /// To speed up indexing, child and free indices must be correct.
    meta: u32,
}

impl ChunkNode {
    const CHILDREN_INDEX_BITS: u32 = (1u32 << 25) - 1;
    const DIRTY_BIT: u32 = 1 << 31;
    const EXACTNESS_SHIFT: u32 = 28;
    const EXACTNESS_BITS: u32 = 0b1111 << Self::EXACTNESS_SHIFT;

    /// Since this is a tree, we never iterate the chunk's list, but there are sometimes gaps.
    /// This is a dummy value that can be used as a placeholder.
    const PLACEHOLDER: Self = Self {
        voxel_data: VoxelData::PLACEHOLDER,
        meta: u32::MAX,
    };

    /// Gets the index of the first of 8 children (stored densely) if it is a parent node.
    #[inline]
    fn children(self) -> Option<NonZero<u32>> {
        NonZero::new(self.meta & Self::CHILDREN_INDEX_BITS)
    }

    /// This sets [`Self::children`].
    /// If `children_index` is 0, this makes this a leaf node, and [`Self::set_exact_to`] should be called immediately.
    /// If `children_index` is not 0, this makes this a parent node, and [`Self::is_dirty`] should be set immediately.
    #[inline]
    fn set_children(&mut self, children_index: u32) {
        self.meta = children_index;
    }

    /// If this is a leaf node, returns the [`Lod`] to which this [`ChunkNode::voxel_data`] is exact.
    /// Higher (less detailed/bigger) lods can use this value exactly.
    /// Lower (more detailed/smaller) lods should prefer to fetch a new sample.
    ///
    /// If this is not a leaf node, this result is meaningless.
    /// It is still safe, but using it as correct is a logic error.
    #[inline]
    fn exact_to(self) -> Lod {
        Lod((self.meta >> Self::EXACTNESS_SHIFT) as u8)
    }

    /// Sets [`Self::exact_to`].
    /// This should only be done if this is a leaf node.
    #[inline]
    fn set_exact_to(&mut self, value: Lod) {
        self.meta =
            (self.meta & !Self::EXACTNESS_BITS) | ((value.0 as u32) << Self::EXACTNESS_SHIFT);
    }

    /// If this is a parent node, returns if the current voxel data still correctly approximates its children.
    /// If not, this should be recalculated with [`VoxelData::approximate`].
    ///
    /// If this is not a parent node, this result is meaningless.
    /// It is still safe, but using it as correct is a logic error.
    #[inline]
    fn is_dirty(self) -> bool {
        self.meta & Self::DIRTY_BIT > 0
    }
    /// Sets [`Self::is_dirty`] to true.
    /// This should only be done if this is a parent node.
    #[inline]
    fn set_dirty_true(&mut self) {
        self.meta |= Self::DIRTY_BIT;
    }

    /// Sets [`Self::is_dirty`] to false.
    /// This should only be done if this is a parent node.
    #[inline]
    fn set_dirty_false(&mut self) {
        self.meta &= !Self::DIRTY_BIT;
    }
}

/// Represents a octree of [`ChunkNode`]s.
///
/// # Safety
///
/// This must always have the root node at index 0.
struct ChunkTree(Vec<ChunkNode>);

impl ChunkTree {
    const CHUNK_NODE_DEPTH: u32 = Lod::MAX.0 as u32;

    /// Compresses this chunk data into a very dense form.
    /// This is mainly used for serialization, but is generally useful for compressing data as needed.
    pub fn compress<O>(&self) -> Vec<u8> {
        todo!()
    }

    /// Expands chunk data from a very dense form.
    /// This is mainly used for deserialization, but is generally useful for restoring [`Self::compress`]ed data as needed.
    pub fn expand(compressed: &[u8]) -> Self {
        todo!()
    }
}

/// Represents a cubic block of [`VoxelData`].
pub struct Chunk {
    ldb_loc: IVec3,
    tree: ChunkTree,
}

impl Chunk {
    /// This is the length of the cube that the [`Chunk`] represents.
    pub const CHUNK_SIZE: u32 = Lod::MAX.length();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_children_bits() {
        let total_possible_nodes: u32 = (0..=Lod::MAX.0).map(|l| 8u32.pow(l as u32)).sum();
        // The child index must only ever take 25 bits.
        assert!(total_possible_nodes < 2u32.pow(25));
    }
}
