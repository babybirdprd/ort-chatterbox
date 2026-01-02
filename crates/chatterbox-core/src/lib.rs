//! # Chatterbox Core
//!
//! High-performance TTS inference library using ONNX Runtime for Chatterbox Turbo.
//!
//! ## Features
//!
//! - **Zero-shot voice cloning** from 5+ second audio samples
//! - **Paralinguistic tags** like `[laugh]`, `[chuckle]`, `[sigh]`
//! - **GPU acceleration** via CUDA (feature-gated)
//! - **Multiple quantization levels** (FP32, FP16)
//! - **Streaming generation** with token or audio chunk callbacks
//! - **Voice caching** for repeated inference with same voice
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use chatterbox_core::{ChatterboxTTS, Config, Device, GenerateOptions};
//!
//! let config = Config::builder()
//!     .device(Device::Cuda(0))
//!     .build();
//!
//! let mut tts = ChatterboxTTS::new(config)?;
//! tts.add_voice("narrator", "narrator.wav")?;
//!
//! let audio = tts.generate("Hello world! [chuckle]", "narrator", GenerateOptions::default())?;
//! ```

pub mod audio;
pub mod config;
pub mod error;
pub mod inference;
pub mod models;
pub mod voices;

// Re-exports for convenience
pub use config::{Config, ConfigBuilder, Device, GenerateOptions, ModelDtype, StreamMode};
pub use error::{Error, Result};
pub use inference::ChatterboxTTS;
pub use voices::VoiceCache;
