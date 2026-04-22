//! GPU/VRAM utilities for benchmarking.
//!
//! Re-exports VRAM querying from `tama_core` for use in benchmark handlers.

pub use tama_core::gpu::{query_vram, VramInfo};
