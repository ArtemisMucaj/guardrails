//! guardrail — a transparent proxy that repairs malformed tool calls from
//! local OpenAI-compatible servers before they reach the client.
//!
//! The macOS release binary is Developer ID-signed and notarized, so it runs
//! without a Gatekeeper exception.

pub mod admin;
pub mod application;
pub mod cli;
pub mod connector;
pub mod domain;

// Re-export the public API.
pub use admin::{build_admin_app, AdminInfo, AdminState};
pub use application::{build_app, AppState, Guardrails};
