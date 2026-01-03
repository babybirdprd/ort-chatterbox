//! Generation progress component.

use crate::state::{AppState, GenerationStatus};
use dioxus::prelude::*;

#[component]
pub fn GenerationProgress(state: Signal<AppState>) -> Element {
    let status = state.read().generation_status.clone();

    // Determine progress display based on mode
    let (progress_text, progress_pct, subtitle) = match &status {
        GenerationStatus::Generating {
            tokens_generated,
            max_tokens,
        } => {
            let pct = (*tokens_generated as f32 / *max_tokens as f32 * 100.0).min(100.0);
            (
                format!("{} / {} tokens", tokens_generated, max_tokens),
                pct,
                if *tokens_generated > 0 {
                    Some(format!(
                        "~{}s remaining",
                        (max_tokens - tokens_generated) / 50
                    ))
                } else {
                    None
                },
            )
        }
        GenerationStatus::GeneratingChunked {
            current_chunk,
            total_chunks,
        } => {
            let pct = (*current_chunk as f32 / *total_chunks as f32 * 100.0).min(100.0);
            (
                format!("Sentence {} / {}", current_chunk, total_chunks),
                pct,
                Some("Processing in chunks to save memory".to_string()),
            )
        }
        _ => ("Initializing...".to_string(), 0.0, None),
    };

    rsx! {
        div { class: "glass-card p-4",
            div { class: "flex items-center justify-between mb-2",
                div { class: "flex items-center gap-2",
                    span { class: "generating text-accent-400", "🔊" }
                    span { class: "text-sm font-medium text-gray-300", "Generating speech..." }
                }
                span { class: "text-sm font-mono text-gray-400",
                    "{progress_text}"
                }
            }

            // Progress bar
            div { class: "h-2 bg-white/10 rounded-full overflow-hidden",
                div {
                    class: "h-full bg-gradient-to-r from-accent-500 to-cyan-500 transition-all duration-300",
                    style: "width: {progress_pct}%"
                }
            }

            // Subtitle
            if let Some(sub) = subtitle {
                p { class: "text-xs text-gray-500 mt-2 text-right",
                    "{sub}"
                }
            }
        }
    }
}
