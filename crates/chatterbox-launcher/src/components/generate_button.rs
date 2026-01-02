//! Generate button component.

use crate::state::{AppState, GenerationStatus};
use crate::worker::WorkerCommand;
use dioxus::prelude::*;

#[component]
pub fn GenerateButton(state: Signal<AppState>) -> Element {
    let is_generating = matches!(
        state.read().generation_status,
        GenerationStatus::Generating { .. } | GenerationStatus::Loading
    );
    let has_voice = state.read().selected_voice.is_some();
    let has_text = !state.read().text.trim().is_empty();
    let can_generate = has_voice && has_text && !is_generating;

    rsx! {
        div { class: "flex gap-4",
            // Generate button
            button {
                class: if can_generate { "btn-accent flex-1 flex items-center justify-center gap-2" } else { "btn-accent flex-1 flex items-center justify-center gap-2 opacity-50 cursor-not-allowed" },
                disabled: !can_generate,
                onclick: move |_| {
                    if can_generate {
                        let text = state.read().text.clone();
                        let voice_id = state.read().selected_voice.clone().unwrap();
                        let temperature = state.read().temperature;
                        let max_tokens = state.read().max_tokens;

                        state.write().generation_status = GenerationStatus::Loading;
                        state.write().output_audio = None;

                        if let Some(tx) = &state.read().worker_tx {
                            let _ = tx.send(WorkerCommand::Generate {
                                text,
                                voice_id,
                                temperature,
                                max_tokens,
                            });
                        }
                    }
                },

                if is_generating {
                    span { class: "animate-spin", "⏳" }
                    span { "Generating..." }
                } else {
                    span { "▶" }
                    span { "Generate" }
                }
            }

            // Batch button
            button {
                class: "btn-ghost border border-white/10",
                onclick: move |_| {
                    spawn(async move {
                        if let Some(file) = rfd::AsyncFileDialog::new()
                            .add_filter("JSON", &["json"])
                            .add_filter("Text", &["txt"])
                            .pick_file()
                            .await
                        {
                            // TODO: Handle batch file
                            let _path = file.path();
                        }
                    });
                },
                span { "📁" }
                span { "Batch" }
            }
        }

        // Helper text
        if !has_voice {
            p { class: "text-sm text-yellow-400/80 mt-2",
                "⚠️ Select a voice from the sidebar to generate speech"
            }
        }
    }
}
