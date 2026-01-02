//! Sidebar component with voice library.

use crate::state::{AppState, Voice};
use crate::worker::WorkerCommand;
use dioxus::prelude::*;

#[component]
pub fn Sidebar(state: Signal<AppState>) -> Element {
    let voices = state.read().voices.clone();
    let selected = state.read().selected_voice.clone();

    rsx! {
        div { class: "w-72 glass-card m-4 p-4 flex flex-col gap-4",
            // Header
            div { class: "flex items-center gap-3 mb-2",
                div { class: "w-10 h-10 rounded-xl bg-gradient-to-br from-accent-500 to-cyan-500 flex items-center justify-center",
                    span { class: "text-xl", "🎤" }
                }
                div {
                    h1 { class: "text-lg font-bold text-white", "Chatterbox" }
                    p { class: "text-xs text-gray-400", "Voice Library" }
                }
            }

            // Divider
            div { class: "h-px bg-white/10" }

            // Voice list
            div { class: "flex-1 overflow-y-auto space-y-1",
                if voices.is_empty() {
                    div { class: "text-center py-8 text-gray-500",
                        p { class: "text-3xl mb-2", "🎙️" }
                        p { class: "text-sm", "No voices yet" }
                        p { class: "text-xs", "Add a voice to get started" }
                    }
                } else {
                    for voice in voices.iter() {
                        VoiceCard {
                            voice: voice.clone(),
                            is_selected: selected.as_ref() == Some(&voice.id),
                            on_select: move |id: String| {
                                state.write().selected_voice = Some(id);
                            },
                            on_delete: move |id: String| {
                                if let Some(tx) = &state.read().worker_tx {
                                    let _ = tx.send(WorkerCommand::RemoveVoice { id });
                                }
                            }
                        }
                    }
                }
            }

            // Add voice button
            AddVoiceButton { state }
        }
    }
}

#[component]
fn VoiceCard(
    voice: Voice,
    is_selected: bool,
    on_select: EventHandler<String>,
    on_delete: EventHandler<String>,
) -> Element {
    let id = voice.id.clone();
    let id_for_delete = voice.id.clone();

    rsx! {
        div {
            class: if is_selected { "voice-card active" } else { "voice-card" },
            onclick: move |_| on_select.call(id.clone()),

            // Avatar
            div { class: "w-9 h-9 rounded-lg bg-accent-500/20 flex items-center justify-center text-accent-400",
                "🎙️"
            }

            // Name
            div { class: "flex-1 min-w-0",
                p { class: "text-sm font-medium text-white truncate", "{voice.name}" }
                p { class: "text-xs text-gray-500 truncate", "{voice.id}" }
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

#[component]
fn AddVoiceButton(state: Signal<AppState>) -> Element {
    rsx! {
        button {
            class: "w-full btn-ghost flex items-center justify-center gap-2 border border-dashed border-white/20 rounded-xl py-3",
            onclick: move |_| {
                spawn(async move {
                    // Open file picker
                    if let Some(path) = rfd::AsyncFileDialog::new()
                        .add_filter("Audio", &["wav"])
                        .pick_file()
                        .await
                    {
                        let path = path.path().to_path_buf();
                        let name = path.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Voice")
                            .to_string();
                        let id = format!("voice_{}", chrono::Utc::now().timestamp());

                        // Add to local state
                        state.write().voices.push(crate::state::Voice {
                            id: id.clone(),
                            name: name.clone(),
                            path: path.clone(),
                        });

                        // Send to worker
                        if let Some(tx) = &state.read().worker_tx {
                            let _ = tx.send(WorkerCommand::AddVoice {
                                id,
                                name,
                                path,
                            });
                        }
                    }
                });
            },
            span { class: "text-lg", "+" }
            span { "Add Voice" }
        }
    }
}
