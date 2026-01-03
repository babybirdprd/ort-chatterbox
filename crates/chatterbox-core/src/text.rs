//! Text processing utilities for TTS.

/// Split text into sentences for chunked generation.
///
/// Splits on sentence-ending punctuation (`.`, `!`, `?`) while preserving
/// the punctuation with each sentence. Handles paralinguistic tags gracefully.
///
/// # Examples
/// ```
/// use chatterbox_core::text::split_by_sentence;
///
/// let sentences = split_by_sentence("Hello world! How are you? I'm fine.");
/// assert_eq!(sentences, vec!["Hello world!", "How are you?", "I'm fine."]);
/// ```
pub fn split_by_sentence(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![];
    }

    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        current.push(c);

        // Check for sentence-ending punctuation
        if c == '.' || c == '!' || c == '?' {
            // Look ahead - skip if followed by more punctuation or closing bracket
            let next = chars.peek();
            let is_end = match next {
                None => true,
                Some(&nc) => {
                    nc.is_whitespace() || nc == '"' || nc == '\'' || nc == ')' || nc == ']'
                }
            };

            if is_end {
                // Consume any trailing quotes/brackets that belong to this sentence
                while let Some(&nc) = chars.peek() {
                    if nc == '"' || nc == '\'' || nc == ')' || nc == ']' {
                        current.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }

                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();

                // Skip leading whitespace for next sentence
                while let Some(&nc) = chars.peek() {
                    if nc.is_whitespace() {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Add any remaining text as final sentence
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sentences() {
        let result = split_by_sentence("Hello world! How are you? I'm fine.");
        assert_eq!(result, vec!["Hello world!", "How are you?", "I'm fine."]);
    }

    #[test]
    fn test_single_sentence() {
        let result = split_by_sentence("Hello world!");
        assert_eq!(result, vec!["Hello world!"]);
    }

    #[test]
    fn test_no_punctuation() {
        let result = split_by_sentence("Hello world");
        assert_eq!(result, vec!["Hello world"]);
    }

    #[test]
    fn test_empty() {
        let result = split_by_sentence("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let result = split_by_sentence("   ");
        assert!(result.is_empty());
    }

    #[test]
    fn test_with_tags() {
        let result = split_by_sentence("Oh wow! [chuckle] That's great. Really cool!");
        assert_eq!(
            result,
            vec!["Oh wow!", "[chuckle] That's great.", "Really cool!"]
        );
    }

    #[test]
    fn test_quoted_sentence() {
        let result = split_by_sentence("He said \"Hello!\" Then left.");
        assert_eq!(result, vec!["He said \"Hello!\"", "Then left."]);
    }

    #[test]
    fn test_multiple_punctuation() {
        let result = split_by_sentence("What?! Really?!");
        assert_eq!(result, vec!["What?!", "Really?!"]);
    }
}
