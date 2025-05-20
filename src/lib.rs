#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![cfg_attr(not(test), no_std)]
#![doc = include_str!("../README.md")]

pub mod storage;
