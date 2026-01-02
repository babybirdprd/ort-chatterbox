//! Text input component.

use crate::state::AppState;
use dioxus::prelude::*;

#[component]
pub fn TextInput(state: Signal<AppState>) -> Element {
    rsx! {
        div { class: "glass-card p-4",
            label { class: "block text-sm font-medium text-gray-300 mb-2", "Enter text to synthesize" }
            textarea {
                class: "input-field min-h-[120px] resize-none",
                placeholder: "Type something... Try using tags like [laugh] or [chuckle]!",
                value: "{state.read().text}",
                oninput: move |e| {
                    state.write().text = e.value();
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
