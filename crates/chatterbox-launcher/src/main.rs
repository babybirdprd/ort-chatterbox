//! Chatterbox Launcher - Modern desktop TTS application
//!
//! A beautiful desktop app for voice cloning and text-to-speech generation.

mod components;
mod state;
mod worker;

use dioxus::prelude::*;
use state::{AppState, GenerationStatus};
use std::sync::{Arc, Mutex};

fn main() {
    dioxus_logger::init(dioxus_logger::tracing::Level::INFO).expect("logger failed");
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    // Initialize app state with proper defaults
    let mut state = use_signal(AppState::new);

    // Spawn worker once and wrap receiver in Arc<Mutex> for use_hook compatibility
    let worker = use_hook(|| match worker::spawn_worker() {
        Ok((tx, rx)) => Some((tx, Arc::new(Mutex::new(rx)))),
        Err(e) => {
            eprintln!("Failed to spawn worker: {}", e);
            None
        }
    });

    // Clone for closures before moving
    let worker_for_effect = worker.clone();
    let worker_for_future = worker.clone();

    // Set worker_tx into state on first render
    use_effect(move || {
        if state.read().worker_tx.is_none() {
            if let Some((ref tx, _)) = worker_for_effect {
                state.write().worker_tx = Some(tx.clone());
            }
        }
    });

    // Poll for worker events
    let _event_poller = use_future(move || {
        let worker = worker_for_future.clone();
        async move {
            loop {
                if let Some((_, ref rx_mutex)) = worker {
                    if let Ok(rx) = rx_mutex.try_lock() {
                        // Drain all available events
                        loop {
                            match rx.try_recv() {
                                Ok(event) => {
                                    worker::handle_event(&mut state, event);
                                }
                                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
                            }
                        }
                    }
                }
                // Poll every 50ms
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
    });

    rsx! {
        document::Stylesheet { href: asset!("/input.css") }

        div { class: "min-h-screen flex",
            // Sidebar
            components::Sidebar { state }

            // Main content
            div { class: "flex-1 flex flex-col p-6 gap-6",
                // Header
                components::Header {}

                // Text input area
                components::TextInput { state }

                // Tag buttons
                components::TagButtons { state }

                // Settings
                components::Settings { state }

                // Generate button
                components::GenerateButton { state }

                // Audio player (when audio is available)
                if state.read().output_audio.is_some() {
                    components::AudioPlayer { state }
                }

                // Generation progress
                if matches!(state.read().generation_status, GenerationStatus::Generating { .. }) {
                    components::GenerationProgress { state }
                }

                // Generation history
                if !state.read().history.is_empty() {
                    components::HistoryPanel { state }
                }
            }
        }
    }
}
