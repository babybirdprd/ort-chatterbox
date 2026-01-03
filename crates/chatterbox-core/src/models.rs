//! Model downloading and ONNX session management.

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

/// ONNX sessions for all model components.
pub struct ModelSessions {
    pub speech_encoder: Session,
    pub embed_tokens: Session,
    pub language_model: Session,
    pub conditional_decoder: Session,
}

/// Create ONNX Runtime sessions for all models.
pub fn create_sessions(paths: &ModelPaths, config: &Config) -> Result<ModelSessions> {
    // Initialize ORT
    ort::init().with_name("chatterbox").commit()?;

    let build_session = |path: &PathBuf| -> Result<Session> {
        let session_builder = Session::builder()?;

        // Build execution providers based on device config
        match config.device {
            Device::Cpu => {
                // CPU only - no additional providers
                Ok(session_builder.commit_from_file(path)?)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_device_id) => {
                // CUDA explicitly requested
                let builder = session_builder
                    .with_execution_providers([CUDAExecutionProvider::default().build()])?;
                Ok(builder.commit_from_file(path)?)
            }
            #[cfg(feature = "directml")]
            Device::DirectML(_device_id) => {
                // DirectML explicitly requested
                let builder = session_builder
                    .with_execution_providers([DirectMLExecutionProvider::default().build()])?;
                Ok(builder.commit_from_file(path)?)
            }
            Device::Auto => {
                // Try CUDA first (cuDNN auto-download should work now)
                // DirectML disabled in Auto due to LayerNormalization compatibility issues
                #[cfg(feature = "cuda")]
                {
                    let builder = session_builder
                        .with_execution_providers([CUDAExecutionProvider::default().build()])?;
                    return Ok(builder.commit_from_file(path)?);
                }

                // Fall back to CPU if no CUDA
                #[cfg(not(feature = "cuda"))]
                {
                    Ok(session_builder.commit_from_file(path)?)
                }
            }
        }
    };

    Ok(ModelSessions {
        speech_encoder: build_session(&paths.speech_encoder)?,
        embed_tokens: build_session(&paths.embed_tokens)?,
        language_model: build_session(&paths.language_model)?,
        conditional_decoder: build_session(&paths.conditional_decoder)?,
    })
}
