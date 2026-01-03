//! Paralinguistic tag buttons.

use crate::state::AppState;
use dioxus::prelude::*;

const TAGS: &[(&str, &str)] = &[
    ("😄", "[laugh]"),
    ("😊", "[chuckle]"),
    ("😮‍💨", "[sigh]"),
    ("🤧", "[cough]"),
    ("😲", "[gasp]"),
    ("😩", "[groan]"),
    ("😤", "[sniff]"),
    ("🤫", "[shush]"),
    ("🗣️", "[clear throat]"),
];

#[component]
pub fn TagButtons(state: Signal<AppState>) -> Element {
    rsx! {
        div { class: "flex flex-wrap gap-2",
            span { class: "text-sm text-gray-400 mr-2", "Tags:" }
            for (emoji, tag) in TAGS.iter() {
                button {
                    class: "tag-btn flex items-center gap-1.5",
                    onclick: move |_| {
                        let text = state.read().text.clone();
                        let cursor_pos = state.read().text_cursor_position.unwrap_or(text.len());
                        // Clamp cursor position to valid range
                        let pos = cursor_pos.min(text.len());
                        // Insert tag at cursor position with space
                        let new_text = format!("{}{} {}", &text[..pos], tag, &text[pos..]);
                        let new_cursor = pos + tag.len() + 1; // Move cursor after inserted tag + space
                        state.write().text = new_text;
                        state.write().text_cursor_position = Some(new_cursor);
                    },
                    span { "{emoji}" }
                    span { "{tag}" }
                }
            }
        }
    }
}
