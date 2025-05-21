//! Contains the logic for specifying bulk [`VoxelData`].

use core::num::NonZero;

use arrayvec::ArrayVec;
use bevy_math::{U8Vec3, UVec3};
use bevy_platform::{prelude::*, sync::Arc};
use fixedbitset::FixedBitSet;
use thiserror::Error;

use crate::storage::voxel::VoxelData;

use super::voxel::VoxelLocation;

/// Represents a level of detail or layer of octree.
/// This value must be in range 0..=[`Lod::MAX`], 0 meaning an exact voxel, max meaning an entire chunk.
/// A [`Lod`] that is greater than another represents a bigger area and will be less detailed/precise.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
#[repr(transparent)]
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

    const fn try_from_index(index: u8) -> Option<Self> {
        if index <= Self::MAX.0 {
            Some(Self(index))
        } else {
            None
        }
    }

    /// The index of the lod; its inner representation.
    pub const fn index(self) -> u8 {
        self.0
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
/// If it is free (available for reuse):
///
/// - The index (or 0) of another 8 free nodes.
///
/// This can also represent a "free" node, which is void and pending reuse.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct ChunkNode {
    /// The voxel data for this cube. (Unless it is free)
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
    /// - 7 bits currently unused,
    ///
    /// If this is not a node and is "free", this just stores the index to the next free node, or 0 if there is no free node.
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

    /// Sets [`Self::exact_to`], making this a leaf node.
    #[inline]
    fn set_exact_to(&mut self, value: Lod) {
        self.meta = (value.0 as u32) << Self::EXACTNESS_SHIFT;
    }
}

/// Represents a octree of [`ChunkNode`]s.
///
/// # Safety
///
/// This must always have the root node at index 0.
#[derive(Debug, PartialEq, Eq, Clone)]
struct ChunkTree {
    tree: Vec<ChunkNode>,
    /// Points to a region of 8 free nodes, or `None` if none are free.
    index_to_free: Option<NonZero<u32>>,
}

/// An error that can occur when expanding a chunk tree.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ChunkExpansionError {
    /// The version number does not correspond to any version ever.
    #[error("The version is not recognized.")]
    InvalidVersion(u32),
    /// Th stamped length was impossible (0 or really big)
    #[error("The tree's length stamp is impossible: {0}.")]
    ImpossibleTreeLen(u32),
    /// The compressed buffer was smaller than expected.
    #[error("The compressed buffer ended before a valid tree structure could be found.")]
    UnexpectedBufferEnd,
    /// The buffer did not end when the tree was finished being expanded.
    #[error("data store disconnected")]
    UnexpectedBufferExcess,
    /// A child index of the tree was invalid.
    #[error("A parent's node's child index is wrong:")]
    InvalidIndex(u32),
    /// The [`Lod`] for exactness was invalid.
    #[error("A leaf node's exactness was invalid.")]
    InvalidExactness(u8),
    /// The data spelled out by the tree was deeper than is allowed.
    #[error("The tree seems to be deeper than a chunk can be.")]
    TreeDepthExceeded,
}

impl ChunkTree {
    const CHUNK_NODE_DEPTH: u32 = Lod::MAX.0 as u32;
    const ENCODER_VERSION: u32 = 1;
    const MAX_NODES: u32 = 19173961;

    /// Compresses this chunk data into a very dense form.
    /// This is mainly used for serialization, but is generally useful for compressing data as needed.
    fn compress(&self) -> Vec<u8> {
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
                    data.push(node.exact_to().index());
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
    fn expand(compressed: &[u8]) -> Result<Self, ChunkExpansionError> {
        fn next_u32(buff: &mut impl Iterator<Item = u8>) -> Result<u32, ChunkExpansionError> {
            Ok(u32::from_be_bytes([
                buff.next()
                    .ok_or(ChunkExpansionError::UnexpectedBufferEnd)?,
                buff.next()
                    .ok_or(ChunkExpansionError::UnexpectedBufferEnd)?,
                buff.next()
                    .ok_or(ChunkExpansionError::UnexpectedBufferEnd)?,
                buff.next()
                    .ok_or(ChunkExpansionError::UnexpectedBufferEnd)?,
            ]))
        }

        let mut buff = compressed.iter().copied();
        let version = next_u32(&mut buff)?;
        if version != Self::ENCODER_VERSION {
            return Err(ChunkExpansionError::InvalidVersion(version));
        }
        let len = next_u32(&mut buff)?;
        if len == 0 || len > Self::MAX_NODES || len & 0b111 != 1 {
            return Err(ChunkExpansionError::ImpossibleTreeLen(len));
        }

        let first = next_u32(&mut buff)?;
        // It's data
        if first & VoxelData::RESERVED_BIT == 0 {
            let mut tree = Vec::new();
            let exactness = buff
                .next()
                .ok_or(ChunkExpansionError::UnexpectedBufferEnd)?;
            let mut root = ChunkNode::PLACEHOLDER;
            root.set_exact_to(
                Lod::try_from_index(exactness)
                    .ok_or(ChunkExpansionError::InvalidExactness(exactness))?,
            );
            root.voxel_data = VoxelData::from_bits_unchecked(first); // Reserved bit was off.
            tree.push(root);

            if buff.next().is_some() {
                return Err(ChunkExpansionError::UnexpectedBufferExcess);
            }

            Ok(Self {
                tree,
                index_to_free: None,
            })
        }
        // It's a parent
        else {
            let mut tree = Vec::with_capacity(len as usize);
            tree.resize(len as usize, ChunkNode::PLACEHOLDER);

            let first_child_idx = first & !VoxelData::RESERVED_BIT;
            let last_child_idx = first_child_idx.saturating_add(7);
            if last_child_idx >= len || first_child_idx == 0 || first_child_idx & 0b111 != 1 {
                return Err(ChunkExpansionError::InvalidIndex(first_child_idx));
            }

            // SAFETY: We checked for len == 0 earlier.
            let root = unsafe { tree.get_unchecked_mut(0) };
            root.meta = first_child_idx;

            // This follows the same format as `ChunkNodeIterator`, but we're building the tree as we iterate it.
            // SAFETY: We must only ever push valid indexes here. (We checked earlier that the len > 0.)
            let mut path =
                ArrayVec::<u32, { ChunkTree::CHUNK_NODE_DEPTH as usize + 1 }>::new_const();
            // SAFETY: Cap > 2.
            unsafe {
                // Each item points to the next.
                // At any given point, the last item is the next to pull from the buffer and the one before it should point to it.
                path.push_unchecked(first_child_idx);
                path.push_unchecked(first_child_idx);
            }

            'decode: loop {
                // SAFETY: We start the path with 1 element. There is only one place we pop,
                // and it is immediately followed by checking if it's now empty.
                let path_item = unsafe { path.last_mut().unwrap_unchecked() };
                let node_index = *path_item & ChunkNodeIterator::NODE_INDEX_BITS;
                let bulk = next_u32(&mut buff)?;

                // It's data
                if bulk & VoxelData::RESERVED_BIT == 0 {
                    // SAFETY: path indices are correct.
                    let node = unsafe { tree.get_unchecked_mut(node_index as usize) };
                    node.meta = 0;
                    let exactness = buff
                        .next()
                        .ok_or(ChunkExpansionError::UnexpectedBufferEnd)?;
                    node.set_exact_to(
                        Lod::try_from_index(exactness)
                            .ok_or(ChunkExpansionError::InvalidExactness(exactness))?,
                    );
                    node.voxel_data = VoxelData::from_bits_unchecked(bulk); // Reserved bit was off.

                    // Now we need to find the next branch to explore
                    'walk_up: loop {
                        // pop this fully explored node
                        // SAFETY: We just got this last node.
                        _ = unsafe { path.pop().unwrap_unchecked() };

                        let Some(c) = path.last_mut() else {
                            // We have gone through the whole tree.
                            break 'decode;
                        };
                        if let Some(next) =
                            c.checked_add(1 << ChunkNodeIterator::SPATIAL_INDEX_SHIFT)
                        {
                            // We have another child to explore
                            *c = next;
                            // SAFETY: We just popped, so capacity is ok.
                            unsafe {
                                path.push_unchecked(
                                    (next & ChunkNodeIterator::NODE_INDEX_BITS)
                                        + (next >> ChunkNodeIterator::SPATIAL_INDEX_SHIFT),
                                );
                            }

                            break 'walk_up;
                        }
                        // Else: We have finished exploring the node and its children, loop and pop the explored node.
                    }
                }
                // It's a parent
                else {
                    let first_child_idx = bulk & !VoxelData::RESERVED_BIT;
                    let last_child_idx = first_child_idx.saturating_add(7);
                    if last_child_idx >= len || first_child_idx == 0 || first_child_idx & 0b111 != 1
                    {
                        return Err(ChunkExpansionError::InvalidIndex(first_child_idx));
                    }

                    // SAFETY: path indices are correct.
                    let node = unsafe { tree.get_unchecked_mut(node_index as usize) };
                    node.meta = first_child_idx;

                    // SAFETY: We just verified it is in bounds.
                    path.try_push(first_child_idx)
                        .ok()
                        .ok_or(ChunkExpansionError::TreeDepthExceeded)?;
                }
            }

            if buff.next().is_some() {
                return Err(ChunkExpansionError::UnexpectedBufferExcess);
            }

            let mut free_idx = 0u32;
            for iter_idx in 0..(len >> 3) {
                let idx = (iter_idx << 3) + 1;
                // SAFETY: `len` is the length of the `tree`, it ends in 0b001, and the shifts cancel each other out.
                let node = unsafe { tree.get_unchecked_mut(idx as usize) };
                if *node == ChunkNode::PLACEHOLDER {
                    node.meta = free_idx;
                    free_idx = idx;
                }
            }

            Ok(Self {
                tree,
                index_to_free: NonZero::new(free_idx),
            })
        }
    }

    /// Applies all changes in these changes, creating new nodes as needed.
    fn apply_changes(&mut self, changes: &mut ChunkChanges) {
        for (loc, data) in changes.0.drain(..) {
            let node_index = self.find_or_make_node_index(loc);
            // SAFETY: This index is correct.
            let node = unsafe { self.tree.get_unchecked_mut(node_index as usize) };
            node.voxel_data = data;
            // TODO: What if this node is a parent? Silently ignore, Delete children, Make approximation wrong (current), Report error?
        }
    }

    /// Finds the index of the node that represents or approximates the `find` region.
    /// If the exact region is found, `Ok` is returned. If it is an approximation, returns `Err`.
    #[inline]
    fn find_node_index(&self, find: ChunkRegion) -> Result<u32, u32> {
        // SAFETY: This must always be valid. (And 0 always is).
        let mut idx = 0;
        let mut lod_layer = Lod::MAX.0;
        while lod_layer > find.lod.0 {
            lod_layer -= 1;
            let layer_split = (find.loc.x >> lod_layer)
                + ((find.loc.y >> lod_layer) << 1)
                + ((find.loc.z >> lod_layer) << 2);
            // SAFETY: Index is always valid.
            let node = unsafe { self.tree.get_unchecked(idx as usize) };
            match node.children() {
                Some(children) => {
                    idx = children.get() + layer_split as u32;
                }
                None => return Err(idx),
            }
        }
        Ok(idx)
    }

    /// Finds the index of the node that represents or approximates the `find` region.
    /// If the exact region is found, `Ok` is returned. If it is an approximation, returns `Err`.
    #[inline]
    fn find_or_make_node_index(&mut self, find: ChunkRegion) -> u32 {
        // SAFETY: This must always be valid. (And 0 always is).
        let mut idx = 0;
        let mut lod_layer = Lod::MAX.0;
        while lod_layer > find.lod.0 {
            lod_layer -= 1;
            let layer_split = (find.loc.x >> lod_layer)
                + ((find.loc.y >> lod_layer) << 1)
                + ((find.loc.z >> lod_layer) << 2);
            // SAFETY: Index is always valid.
            let node = unsafe { *self.tree.get_unchecked(idx as usize) };
            let children = match node.children() {
                Some(children) => children.get(),
                None => {
                    // We need to create a new block of 8 nodes.
                    // We use the approximation of the parent node for their data.
                    match self.index_to_free {
                        Some(new_children_index) => {
                            // SAFETY: This index is valid
                            let child = unsafe {
                                self.tree
                                    .get_unchecked_mut(new_children_index.get() as usize)
                            };
                            self.index_to_free = NonZero::new(child.meta);
                            *child = node;
                            for child_offset in 1..8 {
                                // SAFETY: This index is valid
                                unsafe {
                                    *self.tree.get_unchecked_mut(
                                        (new_children_index.get() + child_offset) as usize,
                                    ) = node;
                                };
                            }
                            new_children_index.get()
                        }
                        None => {
                            let new_children_index = self.tree.len() as u32;
                            for _ in 0..8 {
                                self.tree.push(node);
                            }
                            new_children_index
                        }
                    }
                }
            };
            idx = children + layer_split as u32;
        }
        idx
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

/// Represents the location of a [`Chunk`].
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct ChunkLocation(UVec3);

impl ChunkLocation {
    /// Gets the [`ChunkLocation`] of the chunk that contains `voxel`.
    pub fn of_location(voxel: VoxelLocation) -> Self {
        Self((voxel.location_mapped() >> Lod::MAX.0) << Lod::MAX.0)
    }

    #[inline]
    fn least_corner_location(self) -> UVec3 {
        self.0
    }
}

/// Represents a particular region within a [`Chunk`].
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[repr(align(4))] // To treat as a u32.
pub struct ChunkRegion {
    lod: Lod,
    // Each chunk is 256 voxels wide, so this is exactly the right space.
    loc: U8Vec3,
}

struct ChunkChanges(Vec<(ChunkRegion, VoxelData)>);

/// Manages a cubic block of [`VoxelData`].
pub struct ChunkManager {
    /// This is the prime reference, the source of truth.
    chunk: ChunkReference,
    // TODO: In the future, maybe also support arrays of data for a region.
    changes: ChunkChanges,
}

impl ChunkManager {
    /// This is the length of the cube that the [`Chunk`] represents.
    pub const CHUNK_SIZE: u32 = Lod::MAX.length();

    /// Creates a [`Chunk`] at a particular `location`, based on an `approximation` of the whole chunk.
    pub fn new(location: ChunkLocation, approximation: VoxelData) -> Self {
        Self {
            chunk: ChunkReference {
                location,
                tree: Arc::new(ChunkTree {
                    tree: vec![ChunkNode {
                        voxel_data: approximation,
                        meta: 0,
                    }],
                    index_to_free: None,
                }),
            },
            changes: ChunkChanges(Vec::new()),
        }
    }

    /// Attempts to apply any changes staged in the manager.
    /// Returns true if successful.
    pub fn try_apply_changes(&mut self) -> bool {
        match Arc::get_mut(&mut self.chunk.tree) {
            Some(chunk) => {
                chunk.apply_changes(&mut self.changes);
                true
            }
            None => false,
        }
    }

    /// Applies any changes to the chunk immediately.
    /// This may clone the chunk's data and will not affect any current references to the chunk.
    /// Returns true if there are other references to this chunk.
    pub fn force_apply_changes(&mut self) {
        Arc::make_mut(&mut self.chunk.tree).apply_changes(&mut self.changes);
    }

    /// Creates a shared reference to this chunk.
    /// While this reference exists, mutating the chunk will be harder,
    /// so don't hold onto it for longer than needed.
    pub fn reference(&self) -> ChunkReference {
        self.chunk.clone()
    }
}

/// Represents a shared reference to a chunk.
/// This is a snapshot of chunk data.
/// Holding onto this will make it harder to change the chunk, so don't keep it around longer than you need to.
#[derive(Clone)]
pub struct ChunkReference {
    tree: Arc<ChunkTree>,
    location: ChunkLocation,
}

#[cfg(test)]
mod tests {
    use crate::storage::voxel::{VoxelFullness, VoxelLayer, VoxelMaterial};

    use super::*;

    #[test]
    fn chunk_children_bits() {
        let total_possible_nodes: u32 = (0..=Lod::MAX.0).map(|l| 8u32.pow(l as u32)).sum();
        // The child index must only ever take 25 bits.
        assert!(total_possible_nodes < 2u32.pow(25));
        assert_eq!(total_possible_nodes, ChunkTree::MAX_NODES);
    }

    #[test]
    fn tree_round_trip() {
        let tree = ChunkTree {
            tree: vec![ChunkNode {
                voxel_data: VoxelData::PLACEHOLDER,
                meta: 0,
            }],
            index_to_free: None,
        };
        let compressed = tree.compress();
        let expanded = ChunkTree::expand(&compressed).unwrap();
        assert_eq!(tree, expanded);

        let layer = VoxelLayer::from_u32(1).unwrap();
        let fullness = VoxelFullness::from_quantized(122);
        let tree = ChunkTree {
            tree: vec![
                ChunkNode {
                    voxel_data: VoxelData::PLACEHOLDER,
                    meta: 1,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(1).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(2).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(3).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(4).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(5).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(6).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(7).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
                ChunkNode {
                    voxel_data: VoxelData::new(
                        layer,
                        VoxelMaterial::from_u32(8).unwrap(),
                        fullness,
                    ),
                    meta: 0,
                },
            ],
            index_to_free: None,
        };
        let compressed = tree.compress();
        assert_eq!(compressed.len(), 52);
        let expanded = ChunkTree::expand(&compressed).unwrap();
        assert_eq!(tree.tree, expanded.tree);
        assert_eq!(tree, expanded);
    }
}
