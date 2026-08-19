//! Headless Senmei service: transport-agnostic `core` + adapters.
//!
//! Decision (2026-08-19): `senmei-server` = thin `core` service (probe/models/
//! render/queue + license & confirm gates) with adapters. **MCP (stdio) first**;
//! REST/HTTP is an optional cargo feature, added only when a real consumer
//! exists (YAGNI). MCP is a transport, not the core — an HTTP API later must
//! not require a refactor.

pub mod core;
pub mod mcp;
