//! Text input component.

use crate::state::AppState;
use dioxus::prelude::*;

#[component]
pub fn TextInput(state: Signal<AppState>) -> Element {
    rsx! {
        div { class: "glass-card p-4",
            label { class: "block text-sm font-medium text-gray-300 mb-2", "Enter text to synthesize" }
            textarea {
                id: "text-input-area",
                class: "input-field min-h-[120px] resize-none",
                placeholder: "Type something... Try using tags like [laugh] or [chuckle]!",
                value: "{state.read().text}",
                oninput: move |e| {
                    state.write().text = e.value();
                    // Reset cursor position to end on input
                    state.write().text_cursor_position = Some(e.value().len());
                },
                // Track cursor position on click (user positions cursor)
                onclick: move |_| {
                    // Use JS to get the actual cursor position
                    spawn(async move {
                        if let Ok(result) = dioxus::document::eval(
                            r#"document.getElementById('text-input-area')?.selectionStart || 0"#
                        ).await {
                            if let Some(pos) = result.as_i64() {
                                state.write().text_cursor_position = Some(pos as usize);
                            }
                        }
                    });
                },
                // Track cursor position after keyboard navigation
                onkeyup: move |_| {
                    spawn(async move {
                        if let Ok(result) = dioxus::document::eval(
                            r#"document.getElementById('text-input-area')?.selectionStart || 0"#
                        ).await {
                            if let Some(pos) = result.as_i64() {
                                state.write().text_cursor_position = Some(pos as usize);
                            }
                        }
                    });
                }
            }

            // Character count
            div { class: "flex justify-end mt-2",
                span { class: "text-xs text-gray-500",
                    "{state.read().text.len()} characters"
                }
            }
        }
    }
}

