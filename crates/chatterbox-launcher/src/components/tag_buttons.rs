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
                        let current = state.read().text.clone();
                        state.write().text = format!("{} {}", current, tag);
                    },
                    span { "{emoji}" }
                    span { "{tag}" }
                }
            }
        }
    }
}
