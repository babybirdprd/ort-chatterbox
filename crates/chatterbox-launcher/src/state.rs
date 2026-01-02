//! Application state management.

use std::path::PathBuf;
use std::sync::mpsc;

use crate::worker::WorkerCommand;

/// A saved voice in the library.
#[derive(Debug, Clone, PartialEq)]
pub struct Voice {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
}

/// Generation status.
#[derive(Debug, Clone, Default)]
pub enum GenerationStatus {
    #[default]
    Idle,
    Loading,
    Generating {
        tokens_generated: usize,
        max_tokens: usize,
    },
    Complete,
    Error(String),
}

/// Main application state.
#[derive(Default)]
pub struct AppState {
    // Voices
    pub voices: Vec<Voice>,
    pub selected_voice: Option<String>,

    // Text input
    pub text: String,

    // Settings
    pub temperature: f32,
    pub max_tokens: usize,

    // Generation
    pub generation_status: GenerationStatus,
    pub output_audio: Option<Vec<f32>>,
    pub is_playing: bool,

    // Worker communication (std channels for blocking TTS)
    pub worker_tx: Option<mpsc::Sender<WorkerCommand>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            temperature: 0.8,
            max_tokens: 1024,
            ..Default::default()
        }
    }
}
