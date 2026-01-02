//! Settings panel component.

use crate::state::AppState;
use dioxus::prelude::*;

#[component]
pub fn Settings(state: Signal<AppState>) -> Element {
    rsx! {
        div { class: "glass-card p-4",
            div { class: "flex items-center gap-2 mb-4",
                span { class: "text-gray-400", "⚙️" }
                span { class: "text-sm font-medium text-gray-300", "Generation Settings" }
            }

            div { class: "grid grid-cols-2 gap-6",
                // Temperature setting
                div {
                    div { class: "flex items-center justify-between mb-2",
                        label { class: "text-sm text-gray-400", "Temperature" }
                        span { class: "text-sm font-mono text-accent-400",
                            "{state.read().temperature:.1}"
                        }
                    }
                    input {
                        r#type: "range",
                        class: "slider",
                        min: "0.1",
                        max: "1.5",
                        step: "0.1",
                        value: "{state.read().temperature}",
                        oninput: move |e| {
                            if let Ok(v) = e.value().parse::<f32>() {
                                state.write().temperature = v;
                            }
                        }
                    }
                    p { class: "text-xs text-gray-500 mt-1", "Higher = more varied speech" }
                }

                // Max tokens setting
                div {
                    div { class: "flex items-center justify-between mb-2",
                        label { class: "text-sm text-gray-400", "Max Tokens" }
                        span { class: "text-sm font-mono text-accent-400",
                            "{state.read().max_tokens}"
                        }
                    }
                    input {
                        r#type: "range",
                        class: "slider",
                        min: "256",
                        max: "2048",
                        step: "128",
                        value: "{state.read().max_tokens}",
                        oninput: move |e| {
                            if let Ok(v) = e.value().parse::<usize>() {
                                state.write().max_tokens = v;
                            }
                        }
                    }
                    p { class: "text-xs text-gray-500 mt-1", "Maximum speech length" }
                }
            }
        }
    }
}
