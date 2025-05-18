//! Contains the logic for specifying bulk [`VoxelData`].

use core::num::NonZero;

use arrayvec::ArrayVec;
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
struct ChunkTree {
    tree: Vec<ChunkNode>,
    index_to_free: Option<NonZero<u32>>,
}

impl ChunkTree {
    const CHUNK_NODE_DEPTH: u32 = Lod::MAX.0 as u32;
    const ENCODER_VERSION: u32 = 1;

    /// Compresses this chunk data into a very dense form.
    /// This is mainly used for serialization, but is generally useful for compressing data as needed.
    pub fn compress(&self) -> Vec<u8> {
        let mut data = Vec::new();
        // Parent nodes only need 4 bytes.
        // Leaf nodes take 5 bytes (4 for voxel data, 1 for exactness).
        // Parent nodes take 4 bytes (for child index).
        // Most nodes will probably be leaves.
        // Plus four for version.
        // Plus four for length.
        // This will over allocate, but better than allocating twice.
        data.reserve_exact(self.tree.len() * 5 + 8);
        data.extend_from_slice(&Self::ENCODER_VERSION.to_be_bytes());
        data.extend_from_slice(&(self.tree.len() as u32).to_be_bytes());

        fn compress_node(node: ChunkNode, data: &mut Vec<u8>) {
            match node.children() {
                Some(children) => {
                    // Turn on reserved bit to mark it as a parent.
                    let bits = children.get() | VoxelData::RESERVED_BIT;
                    data.extend_from_slice(&bits.to_be_bytes());
                }
                None => {
                    data.extend_from_slice(&node.voxel_data.to_bits().to_be_bytes());
                    data.push(node.exact_to().0);
                }
            }
        }

        // SAFETY: 0 is always valid
        let root = unsafe { *self.tree.get_unchecked(0) };
        compress_node(root, &mut data);

        // SAFETY: 0 is always valid, and this will only be used on self.
        let mut iter = unsafe { ChunkNodeIterator::start_at(0) };
        while let Some(index) = iter.progress_next_node_depth_first(self) {
            // SAFETY: `ChunkNodeIterator` indices are always valid.
            let node = unsafe { *self.tree.get_unchecked(index.get() as usize) };
            compress_node(node, &mut data);
        }

        data
    }

    /// Expands chunk data from a very dense form.
    /// This is mainly used for deserialization, but is generally useful for restoring [`Self::compress`]ed data as needed.
    pub fn expand(compressed: &[u8]) -> Self {
        todo!()
    }
}

/// Stores a path down the tree and a way to traverse it.
///
/// Each item in the list is a `u32` that represents the index of its node and the spatial index (0..8) of its child, which is or will be in the following list item.
/// The node's index in the list is in the least significant 25 bits, and the spatial index of the child following it (0..8) is in the most significant 3 bits.
///
/// # Safety
///
/// This array must never be empty.
/// This array may only be interacted with through the type's methods.
#[derive(PartialEq, Eq, Clone)]
struct ChunkNodeIterator(ArrayVec<u32, { ChunkTree::CHUNK_NODE_DEPTH as usize + 1 }>);

impl ChunkNodeIterator {
    const NODE_INDEX_BITS: u32 = ChunkNode::CHILDREN_INDEX_BITS;
    const SPATIAL_INDEX_SHIFT: u32 = 29;

    /// Creates a [`ChunkNodeIterator`] starting at this node `index`.
    /// Note that the `index` is never returned in iteration.
    ///
    /// The index must be less than the max; no high bits may be on.
    /// Doing otherwise is not a safety issues, but would be a nasty logic error.
    ///
    /// # Safety
    ///
    /// The `index` must be a valid index in the [`ChunkTree`] it will be used in.
    /// This must only ever be used for that same tree.
    unsafe fn start_at(index: u32) -> Self {
        let mut vec = ArrayVec::new_const();
        // SAFETY: CAP > 1
        unsafe {
            vec.push_unchecked(index);
        }
        Self(vec)
    }

    /// Finds the next node in a depth first search.
    /// This is guaranteed to return a valid index in the tree.
    fn progress_next_node_depth_first(&mut self, tree: &ChunkTree) -> Option<NonZero<u32>> {
        let c = *self.0.last()?;
        // SAFETY: These indices are valid.
        let node = unsafe {
            *tree
                .tree
                .get_unchecked((c & Self::NODE_INDEX_BITS) as usize)
        };

        match node.children() {
            // If we can go down, do.
            Some(children) => {
                // SAFETY: The capacity is the maximum depth of the node.
                unsafe {
                    self.0.push_unchecked(children.get());
                }
                Some(children)
            }
            // If we can't go down, we need to go up, until we can go down again.
            None => loop {
                // pop this fully explored node
                // SAFETY: We just got a last item `c`.
                _ = unsafe { self.0.pop().unwrap_unchecked() };

                let c = self.0.last_mut()?;
                if let Some(next) = c.checked_add(1 << Self::SPATIAL_INDEX_SHIFT) {
                    // We have another child to explore
                    *c = next;
                    let node_index = next & Self::NODE_INDEX_BITS;
                    // SAFETY: These indices are valid.
                    let node = unsafe { *tree.tree.get_unchecked(node_index as usize) };

                    // SAFETY: We just popped a child of `node`, so `node` has children.
                    let index = unsafe { node.children().unwrap_unchecked().get() }
                        + (next >> Self::SPATIAL_INDEX_SHIFT);
                    // Push on the new node to explore its children.
                    self.0.push(index);
                    // SAFETY: The only time we add an element to the array is either when it is a child (which is non-zero) or in index 0 from `start_at(0)`.
                    // And we never return the first item; when all has been explored, we pop the only element in the list, and then return upon a `None` last.
                    // If this `if let Some` branch is taken, clearly index is not 0 since it came from a nonzero.
                    break Some(unsafe { NonZero::new_unchecked(index) });
                }
                // Else: We have finished exploring the node and its children, loop and pop the explored node.
            },
        }
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
