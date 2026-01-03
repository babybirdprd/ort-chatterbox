//! Generation history component with CRUD operations.

use crate::state::{AppState, GeneratedAudio};
use dioxus::prelude::*;
use std::path::PathBuf;

#[component]
pub fn HistoryPanel(state: Signal<AppState>) -> Element {
    let history = state.read().history.clone();
    let selected = state.read().selected_history_item.clone();
    let currently_playing = state.read().currently_playing_id.clone();

    rsx! {
        div { class: "glass-card p-4",
            // Header
            div { class: "flex items-center justify-between mb-3",
                div { class: "flex items-center gap-2",
                    span { class: "text-accent-400", "📂" }
                    span { class: "text-sm font-medium text-gray-300", "Generation History" }
                }
                span { class: "text-xs text-gray-500", "{history.len()} items" }
            }

            if history.is_empty() {
                div { class: "text-center py-6 text-gray-500",
                    p { class: "text-2xl mb-2", "🎵" }
                    p { class: "text-sm", "No generations yet" }
                    p { class: "text-xs", "Generated audio will appear here" }
                }
            } else {
                // History list
                div { class: "space-y-1 max-h-48 overflow-y-auto",
                    for item in history.iter() {
                        HistoryItem {
                            item: item.clone(),
                            is_selected: selected.as_ref() == Some(&item.id),
                            is_playing: currently_playing.as_ref() == Some(&item.id),
                            on_select: move |id: String| {
                                // Load this item into the player
                                let item = state.read().history.iter()
                                    .find(|h| h.id == id)
                                    .cloned();
                                if let Some(item) = item {
                                    state.write().output_audio = Some(item.samples.clone());
                                    state.write().selected_history_item = Some(id);
                                }
                            },
                            on_delete: move |id: String| {
                                // Remove from history
                                state.write().history.retain(|h| h.id != id);
                                if state.read().selected_history_item.as_ref() == Some(&id) {
                                    state.write().selected_history_item = None;
                                }
                                // Stop playback if this item was playing
                                if state.read().currently_playing_id.as_ref() == Some(&id) {
                                    state.write().currently_playing_id = None;
                                }
                            },
                            on_play: move |(id, samples): (String, Vec<f32>)| {
                                let is_currently_playing = state.read().currently_playing_id.as_ref() == Some(&id);
                                
                                if is_currently_playing {
                                    // Stop playback
                                    state.write().currently_playing_id = None;
                                } else {
                                    // Start playback
                                    state.write().currently_playing_id = Some(id.clone());
                                    
                                    spawn(async move {
                                        play_audio_samples(&samples).await;
                                        state.write().currently_playing_id = None;
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

async fn play_audio_samples(samples: &[f32]) {
    use rodio::{OutputStream, Source};
    use std::time::Duration;

    if let Ok((_stream, handle)) = OutputStream::try_default() {
        let source = rodio::buffer::SamplesBuffer::new(1, 24000, samples.to_vec());
        if handle.play_raw(source.convert_samples()).is_ok() {
            let duration = samples.len() as f32 / 24000.0;
            tokio::time::sleep(Duration::from_secs_f32(duration)).await;
        }
    }
}

#[component]
fn HistoryItem(
    item: GeneratedAudio,
    is_selected: bool,
    is_playing: bool,
    on_select: EventHandler<String>,
    on_delete: EventHandler<String>,
    on_play: EventHandler<(String, Vec<f32>)>,
) -> Element {
    let id = item.id.clone();
    let id_for_delete = item.id.clone();
    let id_for_play = item.id.clone();
    let samples_for_play = item.samples.clone();
    let truncated_text = if item.text.len() > 30 {
        format!("{}...", &item.text[..30])
    } else {
        item.text.clone()
    };

    rsx! {
        div {
            class: if is_selected {
                "flex items-center gap-3 p-2 rounded-lg bg-accent-500/20 border-l-2 border-accent-500 cursor-pointer group"
            } else {
                "flex items-center gap-3 p-2 rounded-lg hover:bg-white/5 cursor-pointer group"
            },
            onclick: move |_| on_select.call(id.clone()),

            // Play button (replaces static icon)
            button {
                class: "w-8 h-8 rounded-lg bg-accent-500/20 flex items-center justify-center text-accent-400 text-sm hover:bg-accent-500/40 transition-colors",
                onclick: move |e| {
                    e.stop_propagation();
                    on_play.call((id_for_play.clone(), samples_for_play.clone()));
                },
                if is_playing { "⏸" } else { "▶" }
            }

            // Info
            div { class: "flex-1 min-w-0",
                p { class: "text-sm text-white truncate", "{truncated_text}" }
                div { class: "flex gap-2 text-xs text-gray-500",
                    span { "{item.duration_secs:.1}s" }
                    span { "•" }
                    span { "{item.voice_name}" }
                }
            }

            // Delete button
            button {
                class: "opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-red-500/20 text-gray-500 hover:text-red-400 transition-all",
                onclick: move |e| {
                    e.stop_propagation();
                    on_delete.call(id_for_delete.clone());
                },
                "×"
            }
        }
    }
}

/// Save audio to file and return the path
pub fn auto_save_audio(
    samples: &[f32],
    text: &str,
    output_dir: &Option<PathBuf>,
) -> Option<PathBuf> {
    let dir = output_dir.as_ref()?;

    // Create directory if it doesn't exist
    if !dir.exists() {
        if std::fs::create_dir_all(dir).is_err() {
            return None;
        }
    }

    // Generate filename from timestamp and text snippet
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let text_snippet: String = text
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .take(20)
        .collect::<String>()
        .replace(' ', "_");

    let filename = format!("{}_{}.wav", timestamp, text_snippet);
    let path = dir.join(&filename);

    // Save the audio
    if save_wav(&path, samples).is_ok() {
        Some(path)
    } else {
        None
    }
}

fn save_wav(path: &std::path::Path, samples: &[f32]) -> anyhow::Result<()> {
    use hound::{WavSpec, WavWriter};

    let spec = WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s)?;
    }
    writer.finalize()?;

    Ok(())
}
