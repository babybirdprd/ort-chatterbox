//! Background TTS worker for non-blocking generation.
//!
//! Uses std::sync::mpsc for communication to avoid async runtime conflicts.

use dioxus::prelude::*;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::thread;

use chatterbox_core::{ChatterboxTTS, Config, Device, GenerateOptions, StreamEvent};

use crate::state::{AppState, GenerationStatus};

/// Commands sent to the TTS worker.
#[derive(Debug)]
pub enum WorkerCommand {
    /// Add a voice from a file
    AddVoice {
        id: String,
        name: String,
        path: PathBuf,
    },
    /// Remove a voice
    RemoveVoice { id: String },
    /// Generate speech
    Generate {
        text: String,
        voice_id: String,
        temperature: f32,
        max_tokens: usize,
    },
}

/// Events from the TTS worker.
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    /// Engine is ready
    Ready,
    /// Voice added successfully
    VoiceAdded { id: String },
    /// Voice removed
    VoiceRemoved { id: String },
    /// A token was generated
    TokenGenerated { count: usize },
    /// Generation complete
    GenerationComplete { audio: Vec<f32> },
    /// An error occurred
    Error { message: String },
    /// Status update
    Status { message: String },
}

/// Spawn the TTS worker in a background thread.
/// Returns std channels for communication (no async).
pub fn spawn_worker() -> anyhow::Result<(
    std_mpsc::Sender<WorkerCommand>,
    std_mpsc::Receiver<WorkerEvent>,
)> {
    let (cmd_tx, cmd_rx) = std_mpsc::channel::<WorkerCommand>();
    let (event_tx, event_rx) = std_mpsc::channel::<WorkerEvent>();

    // Spawn blocking thread for TTS - no async runtime
    thread::spawn(move || {
        let _ = event_tx.send(WorkerEvent::Status {
            message: "Initializing TTS engine...".into(),
        });

        // Initialize TTS
        let config = Config::builder().device(Device::default()).build();

        let mut tts = match ChatterboxTTS::new(config) {
            Ok(tts) => {
                let _ = event_tx.send(WorkerEvent::Ready);
                tts
            }
            Err(e) => {
                let _ = event_tx.send(WorkerEvent::Error {
                    message: format!("Failed to initialize: {}", e),
                });
                return;
            }
        };

        // Process commands (blocking receive)
        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                WorkerCommand::AddVoice { id, name: _, path } => {
                    println!("[Worker] Adding voice: {} from {:?}", id, path);
                    match tts.add_voice(&id, &path) {
                        Ok(_) => {
                            println!("[Worker] Voice added successfully: {}", id);
                            let _ = event_tx.send(WorkerEvent::VoiceAdded { id });
                        }
                        Err(e) => {
                            println!("[Worker] Failed to add voice: {}", e);
                            let _ = event_tx.send(WorkerEvent::Error {
                                message: format!("Failed to add voice: {}", e),
                            });
                        }
                    }
                }

                WorkerCommand::RemoveVoice { id } => {
                    println!("[Worker] Removing voice: {}", id);
                    tts.remove_voice(&id);
                    let _ = event_tx.send(WorkerEvent::VoiceRemoved { id });
                }

                WorkerCommand::Generate {
                    text,
                    voice_id,
                    temperature,
                    max_tokens,
                } => {
                    println!("[Worker] Starting generation for voice: {}", voice_id);
                    println!("[Worker] Text: {}", text);

                    let opts = GenerateOptions {
                        temperature,
                        max_tokens,
                        ..Default::default()
                    };

                    let event_tx_clone = event_tx.clone();
                    let mut token_count = 0;

                    match tts.generate_streaming(&text, &voice_id, opts, |event| {
                        if let StreamEvent::Token(_) = event {
                            token_count += 1;
                            // Send progress every 10 tokens
                            if token_count % 10 == 0 {
                                println!("[Worker] Generated {} tokens", token_count);
                                let _ = event_tx_clone
                                    .send(WorkerEvent::TokenGenerated { count: token_count });
                            }
                        }
                    }) {
                        Ok(audio) => {
                            println!(
                                "[Worker] Generation complete! {} samples ({:.1}s)",
                                audio.len(),
                                audio.len() as f32 / 24000.0
                            );
                            let _ = event_tx.send(WorkerEvent::GenerationComplete { audio });
                        }
                        Err(e) => {
                            println!("[Worker] Generation error: {}", e);
                            let _ = event_tx.send(WorkerEvent::Error {
                                message: format!("Generation failed: {}", e),
                            });
                        }
                    }
                }
            }
        }
    });

    Ok((cmd_tx, event_rx))
}

/// Handle worker events by updating app state.
pub fn handle_event(state: &mut Signal<AppState>, event: WorkerEvent) {
    match event {
        WorkerEvent::Ready => {
            state.write().generation_status = GenerationStatus::Idle;
        }
        WorkerEvent::VoiceAdded { id: _ } => {
            state.write().generation_status = GenerationStatus::Idle;
        }
        WorkerEvent::VoiceRemoved { id } => {
            state.write().voices.retain(|v| v.id != id);
            if state.read().selected_voice.as_ref() == Some(&id) {
                state.write().selected_voice = None;
            }
        }
        WorkerEvent::TokenGenerated { count } => {
            let max_tokens = state.read().max_tokens;
            state.write().generation_status = GenerationStatus::Generating {
                tokens_generated: count,
                max_tokens,
            };
        }
        WorkerEvent::GenerationComplete { audio } => {
            // Get info for history entry
            let text = state.read().text.clone();
            let voice_name = state
                .read()
                .selected_voice_name()
                .unwrap_or_else(|| "Unknown".to_string());
            let duration_secs = audio.len() as f32 / 24000.0;
            let output_dir = state.read().output_dir.clone();

            // Auto-save to disk
            let file_path = crate::components::auto_save_audio(&audio, &text, &output_dir);

            // Create history entry
            let history_item = crate::state::GeneratedAudio {
                id: format!("gen_{}", chrono::Utc::now().timestamp_millis()),
                text: text.clone(),
                voice_name,
                samples: audio.clone(),
                duration_secs,
                file_path,
                created_at: chrono::Local::now().format("%H:%M:%S").to_string(),
            };

            // Update state
            state.write().history.insert(0, history_item); // Newest first
            state.write().output_audio = Some(audio);
            state.write().generation_status = GenerationStatus::Complete;
        }
        WorkerEvent::Error { message } => {
            state.write().generation_status = GenerationStatus::Error(message);
        }
        WorkerEvent::Status { message: _ } => {
            state.write().generation_status = GenerationStatus::Loading;
        }
    }
}
