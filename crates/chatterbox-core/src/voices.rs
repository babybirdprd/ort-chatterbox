//! Voice embedding cache for efficient repeated inference.

use crate::{Error, Result};
use ndarray::ArrayD;
use std::collections::HashMap;

/// Cached voice embeddings for TTS generation.
///
/// Pre-computing voice embeddings avoids redundant speech encoder runs
/// when generating multiple utterances with the same voice.
#[derive(Debug, Clone)]
pub struct VoiceEmbedding {
    /// Speaker embedding for conditioning the decoder
    pub speaker_embeddings: ArrayD<f32>,
    /// Speaker features for conditioning
    pub speaker_features: ArrayD<f32>,
    /// Prompt tokens from speech encoder
    pub prompt_token: ArrayD<i64>,
    /// Conditioning embedding for the language model
    pub cond_emb: ArrayD<f32>,
}

/// In-memory cache of voice embeddings.
#[derive(Debug, Default)]
pub struct VoiceCache {
    voices: HashMap<String, VoiceEmbedding>,
}

impl VoiceCache {
    /// Create a new empty voice cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a voice embedding to the cache.
    pub fn add(&mut self, id: impl Into<String>, embedding: VoiceEmbedding) {
        self.voices.insert(id.into(), embedding);
    }

    /// Get a voice embedding from the cache.
    pub fn get(&self, id: &str) -> Option<&VoiceEmbedding> {
        self.voices.get(id)
    }

    /// Remove a voice from the cache.
    pub fn remove(&mut self, id: &str) -> bool {
        self.voices.remove(id).is_some()
    }

    /// List all cached voice IDs.
    pub fn list(&self) -> Vec<&str> {
        self.voices.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a voice exists in the cache.
    pub fn contains(&self, id: &str) -> bool {
        self.voices.contains_key(id)
    }

    /// Clear all cached voices.
    pub fn clear(&mut self) {
        self.voices.clear();
    }

    /// Number of cached voices.
    pub fn len(&self) -> usize {
        self.voices.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }
}

/// Validate that audio is suitable for voice cloning.
///
/// Returns the duration in seconds or an error if too short.
pub fn validate_voice_audio(samples: &[f32], sample_rate: u32) -> Result<f32> {
    let duration = samples.len() as f32 / sample_rate as f32;
    if duration < 5.0 {
        return Err(Error::VoiceAudioTooShort(duration));
    }
    Ok(duration)
}
