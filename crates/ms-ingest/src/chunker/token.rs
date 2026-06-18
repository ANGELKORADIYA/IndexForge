use tiktoken_rs::cl100k_base;

pub struct TokenChunker {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl TokenChunker {
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let bpe = cl100k_base().expect("failed to initialize cl100k_base tokenizer");
        let tokens = bpe.encode_ordinary(text);
        if tokens.is_empty() {
            return Vec::new();
        }
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < tokens.len() {
            let end = (start + self.max_tokens).min(tokens.len());
            let window = &tokens[start..end];
            if let Ok(decoded) = bpe.decode(window.to_vec()) {
                chunks.push(decoded);
            }
            let advance = self.max_tokens.saturating_sub(self.overlap_tokens).max(1);
            start += advance;
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_chunker_basic() {
        let chunker = TokenChunker {
            max_tokens: 5,
            overlap_tokens: 2,
        };
        let text = "The quick brown fox jumps over the lazy dog and more words here";
        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_token_chunker_empty() {
        let chunker = TokenChunker {
            max_tokens: 512,
            overlap_tokens: 50,
        };
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_token_chunker_small_input() {
        let chunker = TokenChunker {
            max_tokens: 512,
            overlap_tokens: 50,
        };
        let chunks = chunker.chunk("Hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }
}
