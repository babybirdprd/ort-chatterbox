//! UI Components for Chatterbox Launcher.

mod audio_player;
mod generate_button;
mod generation_progress;
mod header;
mod history;
mod settings;
mod sidebar;
mod tag_buttons;
mod text_input;

pub use audio_player::AudioPlayer;
pub use generate_button::GenerateButton;
pub use generation_progress::GenerationProgress;
pub use header::Header;
pub use history::{auto_save_audio, HistoryPanel};
pub use settings::Settings;
pub use sidebar::Sidebar;
pub use tag_buttons::TagButtons;
pub use text_input::TextInput;
