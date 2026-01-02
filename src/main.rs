mod chatterbox;

use anyhow::Result;

fn main() -> Result<()> {
    chatterbox::run_inference()
}
