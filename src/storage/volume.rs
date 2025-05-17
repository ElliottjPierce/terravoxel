//! Contains the logic for specifying collections of [`Chunk`]s

use bevy_math::IVec3;
use bevy_platform::collections::HashMap;

use crate::chunk::Chunk;

/// Represents a collection of [`Chunk`]s and how to mutate them.
/// All data is specified externally and arbitrarily.
/// This allows the volume to treat data sources from network, disk, procedural generators, and player interaction in the same way.
///
/// Data is never cloned.
/// The [`Chunk`]s are always readable, but may not always be exclusive;
/// Some chunks may be being poligonized or be shipped to some other async task.
/// When a mutation is made, if the chunk is available, it is made directly.
/// Otherwise, it is queued to be mutated when the chunk next becomes available.
///
/// Chunks can be inserted and removed freely.
/// Readonly access to a chunk may also be freely requested.
/// This allows saving and loading chunk data arbitrarily.
/// This can also, on paper, be used for physics collision detection with the terrain.
///
/// Finally, meshes can be generated for regions of the volume that are loaded, at varying degrees of detail, quality, etc.
/// This is a long process that requires holding onto relevant chunk information, which would cause mutations to be queued.
/// Allowing arbitrary meshes to be constructed like this empowers configuring caching, quality, and speed of mesh generation.
pub struct Volume {
    chunks: HashMap<IVec3, Chunk>,
    // changes: tbd
}
