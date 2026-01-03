//! Model downloading and ONNX session management with lazy loading.

use crate::config::{Config, Device, ModelDtype};
use crate::{Error, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use ort::session::Session;
use std::path::PathBuf;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;

/// HuggingFace model repository ID.
pub const MODEL_REPO: &str = "ResembleAI/chatterbox-turbo-ONNX";

/// Paths to downloaded ONNX model files.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub speech_encoder: PathBuf,
    pub embed_tokens: PathBuf,
    pub language_model: PathBuf,
    pub conditional_decoder: PathBuf,
    pub tokenizer: PathBuf,
}

/// Download models from HuggingFace Hub.
///
/// Models are cached locally after first download.
pub fn download_models(dtype: ModelDtype) -> Result<ModelPaths> {
    let api = Api::new().map_err(|e| Error::Model(format!("Failed to create HF API: {}", e)))?;
    let repo = api.repo(Repo::new(MODEL_REPO.to_string(), RepoType::Model));

    let suffix = dtype.suffix();

    let get_model = |name: &str| -> Result<PathBuf> {
        let filename = format!("{}{}.onnx", name, suffix);
        println!("Downloading {}...", filename);

        let model_path = repo
            .get(&format!("onnx/{}", filename))
            .map_err(|e| Error::Model(format!("Failed to download {}: {}", filename, e)))?;

        // Also download the weights data file if it exists
        let data_filename = format!("onnx/{}_data", filename);
        let _ = repo.get(&data_filename); // Ignore error - not all models have separate data files

        Ok(model_path)
    };

    let tokenizer = repo
        .get("tokenizer.json")
        .map_err(|e| Error::Model(format!("Failed to download tokenizer: {}", e)))?;

    Ok(ModelPaths {
        speech_encoder: get_model("speech_encoder")?,
        embed_tokens: get_model("embed_tokens")?,
        language_model: get_model("language_model")?,
        conditional_decoder: get_model("conditional_decoder")?,
        tokenizer,
    })
}

/// ONNX sessions for all model components with lazy loading support.
///
/// Sessions are loaded on-demand to reduce initial memory footprint.
/// Use the accessor methods (`speech_encoder()`, etc.) to get or load sessions.
pub struct ModelSessions {
    paths: ModelPaths,
    config: Config,
    ort_initialized: bool,
    // Lazy-loaded sessions
    speech_encoder: Option<Session>,
    embed_tokens: Option<Session>,
    language_model: Option<Session>,
    conditional_decoder: Option<Session>,
}

impl ModelSessions {
    /// Create a new ModelSessions container without loading any sessions.
    pub fn new(paths: ModelPaths, config: Config) -> Self {
        Self {
            paths,
            config,
            ort_initialized: false,
            speech_encoder: None,
            embed_tokens: None,
            language_model: None,
            conditional_decoder: None,
        }
    }

    /// Ensure ORT is initialized.
    fn ensure_ort_init(&mut self) -> Result<()> {
        if !self.ort_initialized {
            ort::init().with_name("chatterbox").commit()?;
            self.ort_initialized = true;
        }
        Ok(())
    }

    /// Build a session for the given path with current config.
    fn build_session(&self, path: &PathBuf) -> Result<Session> {
        let session_builder = Session::builder()?;

        match self.config.device {
            Device::Cpu => Ok(session_builder.commit_from_file(path)?),
            #[cfg(feature = "cuda")]
            Device::Cuda(_device_id) => {
                let builder = session_builder
                    .with_execution_providers([CUDAExecutionProvider::default().build()])?;
                Ok(builder.commit_from_file(path)?)
            }
            #[cfg(feature = "directml")]
            Device::DirectML(_device_id) => {
                let builder = session_builder
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])?;
                Ok(builder.commit_from_file(path)?)
            }
            Device::Auto => {
                #[cfg(feature = "cuda")]
                {
                    let builder = session_builder
                        .with_execution_providers([CUDAExecutionProvider::default().build()])?;
                    return Ok(builder.commit_from_file(path)?);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(session_builder.commit_from_file(path)?)
                }
            }
        }
    }

    /// Get the speech encoder session, loading it if necessary.
    pub fn speech_encoder(&mut self) -> Result<&mut Session> {
        self.ensure_ort_init()?;
        if self.speech_encoder.is_none() {
            println!("[Lazy] Loading speech_encoder...");
            self.speech_encoder = Some(self.build_session(&self.paths.speech_encoder.clone())?);
        }
        Ok(self.speech_encoder.as_mut().unwrap())
    }

    /// Get the embed_tokens session, loading it if necessary.
    pub fn embed_tokens(&mut self) -> Result<&mut Session> {
        self.ensure_ort_init()?;
        if self.embed_tokens.is_none() {
            println!("[Lazy] Loading embed_tokens...");
            self.embed_tokens = Some(self.build_session(&self.paths.embed_tokens.clone())?);
        }
        Ok(self.embed_tokens.as_mut().unwrap())
    }

    /// Get the language model session, loading it if necessary.
    pub fn language_model(&mut self) -> Result<&mut Session> {
        self.ensure_ort_init()?;
        if self.language_model.is_none() {
            println!("[Lazy] Loading language_model...");
            self.language_model = Some(self.build_session(&self.paths.language_model.clone())?);
        }
        Ok(self.language_model.as_mut().unwrap())
    }

    /// Get the conditional decoder session, loading it if necessary.
    pub fn conditional_decoder(&mut self) -> Result<&mut Session> {
        self.ensure_ort_init()?;
        if self.conditional_decoder.is_none() {
            println!("[Lazy] Loading conditional_decoder...");
            self.conditional_decoder =
                Some(self.build_session(&self.paths.conditional_decoder.clone())?);
        }
        Ok(self.conditional_decoder.as_mut().unwrap())
    }

    /// Unload all sessions to free memory.
    ///
    /// Sessions will be reloaded on next access.
    pub fn unload_all(&mut self) {
        println!("[Lazy] Unloading all sessions to free memory");
        self.speech_encoder = None;
        self.embed_tokens = None;
        self.language_model = None;
        self.conditional_decoder = None;
    }

    /// Preload all sessions (for backwards compatibility or eager loading).
    pub fn preload_all(&mut self) -> Result<()> {
        self.speech_encoder()?;
        self.embed_tokens()?;
        self.language_model()?;
        self.conditional_decoder()?;
        Ok(())
    }
}

/// Create a new ModelSessions container (lazy loading - sessions not loaded yet).
pub fn create_sessions(paths: &ModelPaths, config: &Config) -> Result<ModelSessions> {
    Ok(ModelSessions::new(paths.clone(), config.clone()))
}
