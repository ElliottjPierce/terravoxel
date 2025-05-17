#![allow(
    clippy::doc_markdown,
    reason = "These rules should not apply to the readme."
)]
#![no_std]
#![doc = include_str!("../README.md")]

pub mod storage;
