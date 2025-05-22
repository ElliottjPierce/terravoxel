#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

pub mod storage;
