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

/// A generated audio item in history.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedAudio {
    pub id: String,
    pub text: String,
    pub voice_name: String,
    pub samples: Vec<f32>,
    pub duration_secs: f32,
    pub file_path: Option<PathBuf>,
    pub created_at: String,
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
    /// Chunked generation mode (sentence-by-sentence)
    GeneratingChunked {
        current_chunk: usize,
        total_chunks: usize,
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
    /// Cursor position in text for tag insertion (None = end of text)
    pub text_cursor_position: Option<usize>,

    // Settings
    pub temperature: f32,
    pub max_tokens: usize,

    // Generation
    pub generation_status: GenerationStatus,
    pub output_audio: Option<Vec<f32>>,
    pub is_playing: bool,
    pub playback_progress: f32, // 0.0 to 1.0

    // Generation history
    pub history: Vec<GeneratedAudio>,
    pub selected_history_item: Option<String>,
    /// ID of the history item currently playing (for inline play buttons)
    pub currently_playing_id: Option<String>,

    // Output directory for auto-save
    pub output_dir: Option<PathBuf>,

    // Worker communication (std channels for blocking TTS)
    pub worker_tx: Option<mpsc::Sender<WorkerCommand>>,
}

impl AppState {
    pub fn new() -> Self {
        // Default output directory
        let output_dir = dirs::document_dir()
            .or_else(dirs::home_dir)
            .map(|p| p.join("Chatterbox"));

        Self {
            temperature: 0.8,
            max_tokens: 1024,
            output_dir,
            ..Default::default()
        }
    }

    /// Get the selected voice name for display
    pub fn selected_voice_name(&self) -> Option<String> {
        self.selected_voice.as_ref().and_then(|id| {
            self.voices
                .iter()
                .find(|v| &v.id == id)
                .map(|v| v.name.clone())
        })
    }
}
