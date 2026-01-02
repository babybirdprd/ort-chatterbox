//! Generation progress component.

use crate::state::{AppState, GenerationStatus};
use dioxus::prelude::*;

#[component]
pub fn GenerationProgress(state: Signal<AppState>) -> Element {
    let status = state.read().generation_status.clone();

    let (tokens, max_tokens) = match &status {
        GenerationStatus::Generating {
            tokens_generated,
            max_tokens,
        } => (*tokens_generated, *max_tokens),
        _ => (0, 1024),
    };

    let progress = (tokens as f32 / max_tokens as f32 * 100.0).min(100.0);

    rsx! {
        div { class: "glass-card p-4",
            div { class: "flex items-center justify-between mb-2",
                div { class: "flex items-center gap-2",
                    span { class: "generating text-accent-400", "🔊" }
                    span { class: "text-sm font-medium text-gray-300", "Generating speech..." }
                }
                span { class: "text-sm font-mono text-gray-400",
                    "{tokens} / {max_tokens} tokens"
                }
            }

            // Progress bar
            div { class: "h-2 bg-white/10 rounded-full overflow-hidden",
                div {
                    class: "h-full bg-gradient-to-r from-accent-500 to-cyan-500 transition-all duration-300",
                    style: "width: {progress}%"
                }
            }

            // Estimated time
            if tokens > 0 {
                p { class: "text-xs text-gray-500 mt-2 text-right",
                    "Estimated: ~{(max_tokens - tokens) / 50}s remaining"
                }
            }
        }
    }
}
