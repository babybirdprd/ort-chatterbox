//! Configuration types for Chatterbox TTS.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Execution device for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// CPU-only inference (always available)
    Cpu,
    /// CUDA GPU inference (requires `cuda` feature)
    #[cfg(feature = "cuda")]
    Cuda(u32),
}

impl Default for Device {
    fn default() -> Self {
        #[cfg(feature = "cuda")]
        {
            Device::Cuda(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Device::Cpu
        }
    }
}

/// Model quantization/precision level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ModelDtype {
    /// Full 32-bit float precision
    Fp32,
    /// Half 16-bit float precision (recommended for GPU)
    #[default]
    Fp16,
    /// 8-bit integer quantization
    Q8,
    // Q4 and Q4F16 deferred for now - need dynamic dtype handling
}

impl ModelDtype {
    /// Returns the filename suffix for this dtype.
    pub fn suffix(&self) -> &'static str {
        match self {
            ModelDtype::Fp32 => "",
            ModelDtype::Fp16 => "_fp16",
            ModelDtype::Q8 => "_quantized",
        }
    }
}

/// Streaming mode for generation callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamMode {
    /// Emit callback for each token generated (before audio decode)
    Token,
    /// Accumulate tokens and emit audio chunks
    Audio {
        /// Number of tokens to accumulate before decoding to audio
        chunk_tokens: usize,
    },
}

impl Default for StreamMode {
    fn default() -> Self {
        StreamMode::Token
    }
}

/// Options for text-to-speech generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateOptions {
    /// Sampling temperature (higher = more random). Default: 0.8
    pub temperature: f32,
    /// Top-K sampling. Default: 1000
    pub top_k: usize,
    /// Top-P (nucleus) sampling. Default: 0.95
    pub top_p: f32,
    /// Repetition penalty. Default: 1.2
    pub repetition_penalty: f32,
    /// Maximum tokens to generate. Default: 1024
    pub max_tokens: usize,
    /// Streaming mode (only used with streaming methods)
    pub stream_mode: StreamMode,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 1000,
            top_p: 0.95,
            repetition_penalty: 1.2,
            max_tokens: 1024,
            stream_mode: StreamMode::default(),
        }
    }
}

/// Configuration for ChatterboxTTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Execution device
    pub device: Device,
    /// Model precision/quantization
    pub dtype: ModelDtype,
    /// Directory to cache downloaded models (None = HF cache default)
    pub cache_dir: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: Device::default(),
            dtype: ModelDtype::default(),
            cache_dir: None,
        }
    }
}

impl Config {
    /// Create a new config builder.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

/// Builder for Config.
#[derive(Debug, Clone, Default)]
pub struct ConfigBuilder {
    device: Option<Device>,
    dtype: Option<ModelDtype>,
    cache_dir: Option<PathBuf>,
}

impl ConfigBuilder {
    /// Set the execution device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the model dtype.
    pub fn dtype(mut self, dtype: ModelDtype) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Set the model cache directory.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Build the config.
    pub fn build(self) -> Config {
        Config {
            device: self.device.unwrap_or_default(),
            dtype: self.dtype.unwrap_or_default(),
            cache_dir: self.cache_dir,
        }
    }
}
