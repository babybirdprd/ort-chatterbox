//! Header component.

use dioxus::prelude::*;

#[component]
pub fn Header() -> Element {
    rsx! {
        div { class: "flex items-center justify-between",
            div {
                h2 { class: "text-2xl font-bold text-white", "Text to Speech" }
                p { class: "text-sm text-gray-400", "Generate natural speech from text" }
            }

            // Status indicator
            div { class: "flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/20 text-green-400 text-sm",
                div { class: "w-2 h-2 rounded-full bg-green-400" }
                "Ready"
            }
        }
    }
}
