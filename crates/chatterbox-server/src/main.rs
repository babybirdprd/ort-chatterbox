//! Chatterbox Server - HTTP API for TTS generation.

use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use chatterbox_core::{
    audio::SAMPLE_RATE, ChatterboxTTS, Config, Device, GenerateOptions, ModelDtype,
};

type AppState = Arc<RwLock<ChatterboxTTS>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 Chatterbox Server starting...");

    // Check for API key
    let api_key = std::env::var("CHATTERBOX_API_KEY").ok();
    if api_key.is_some() {
        println!("🔐 API key authentication enabled");
    }

    // Initialize TTS
    let dtype = std::env::var("CHATTERBOX_DTYPE").unwrap_or_else(|_| "fp16".to_string());
    let use_cpu = std::env::var("CHATTERBOX_CPU").is_ok();

    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::default()
    };

    let dtype = match dtype.as_str() {
        "fp32" => ModelDtype::Fp32,
        "q8" => ModelDtype::Q8,
        _ => ModelDtype::Fp16,
    };

    let config = Config::builder().device(device).dtype(dtype).build();
    let tts = ChatterboxTTS::new(config)?;
    let state: AppState = Arc::new(RwLock::new(tts));

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .route("/voices", get(list_voices))
        .route("/voices", post(add_voice))
        .route("/voices/:id", delete(remove_voice))
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{}", port);
    println!("🎤 Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

#[derive(Deserialize)]
struct GenerateRequest {
    text: String,
    voice_id: String,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct GenerateResponse {
    sample_rate: u32,
    samples: Vec<f32>,
    duration_secs: f32,
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, AppError> {
    let opts = GenerateOptions {
        temperature: req.temperature.unwrap_or(0.8),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.2),
        max_tokens: req.max_tokens.unwrap_or(1024),
        ..Default::default()
    };

    let mut tts = state.write().await;
    let samples = tts.generate(&req.text, &req.voice_id, opts)?;
    let duration_secs = samples.len() as f32 / SAMPLE_RATE as f32;

    Ok(Json(GenerateResponse {
        sample_rate: SAMPLE_RATE,
        samples,
        duration_secs,
    }))
}

async fn list_voices(State(state): State<AppState>) -> impl IntoResponse {
    let tts = state.read().await;
    let voices: Vec<String> = tts
        .list_voices()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    Json(serde_json::json!({ "voices": voices }))
}

async fn add_voice(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, AppError> {
    let mut voice_id: Option<String> = None;
    let mut audio_data: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "id" => {
                voice_id = Some(field.text().await?);
            }
            "audio" => {
                audio_data = Some(field.bytes().await?.to_vec());
            }
            _ => {}
        }
    }

    let id = voice_id.ok_or_else(|| AppError::bad_request("Missing 'id' field".into()))?;
    let data = audio_data.ok_or_else(|| AppError::bad_request("Missing 'audio' field".into()))?;

    // Write to temp file
    let temp_path = std::env::temp_dir().join(format!("chatterbox_voice_{}.wav", id));
    std::fs::write(&temp_path, &data)?;

    let mut tts = state.write().await;
    tts.add_voice(&id, &temp_path)?;

    // Cleanup temp file
    let _ = std::fs::remove_file(&temp_path);

    Ok(Json(serde_json::json!({ "id": id, "status": "added" })))
}

async fn remove_voice(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    let mut tts = state.write().await;
    let removed = tts.remove_voice(&id);
    Json(serde_json::json!({ "id": id, "removed": removed }))
}

// Error handling
struct AppError(anyhow::Error);

impl AppError {
    fn bad_request(msg: String) -> Self {
        AppError(anyhow::anyhow!(msg))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(err: E) -> Self {
        AppError(err.into())
    }
}
