//! Error types for Chatterbox.

use thiserror::Error;

/// Result type alias using Chatterbox Error.
pub type Result<T> = std::result::Result<T, Error>;

/// Chatterbox error types.
#[derive(Error, Debug)]
pub enum Error {
    /// Model file not found or failed to download
    #[error("Model error: {0}")]
    Model(String),

    /// Audio file error (read/write/format)
    #[error("Audio error: {0}")]
    Audio(String),

    /// Voice not found in cache
    #[error("Voice '{0}' not found. Add it first with add_voice()")]
    VoiceNotFound(String),

    /// Voice audio too short (needs 5+ seconds)
    #[error("Voice audio must be at least 5 seconds, got {0:.1}s")]
    VoiceAudioTooShort(f32),

    /// ONNX Runtime error
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Shape/dimension mismatch
    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    /// Generic error for other cases
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}
