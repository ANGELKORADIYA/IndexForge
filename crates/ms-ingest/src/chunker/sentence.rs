use unicode_segmentation::UnicodeSegmentation;

pub struct SentenceChunker {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl SentenceChunker {
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let sentences: Vec<&str> = text.unicode_sentences().collect();
        let mut chunks = Vec::new();
        let mut current = String::new();

        for sentence in &sentences {
            if current.len() + sentence.len() > self.chunk_size && !current.is_empty() {
                chunks.push(current.clone());
                // keep overlap: take last N chars
                let overlap_start = current.len().saturating_sub(self.overlap);
                current = current[overlap_start..].to_string();
            }
            current.push_str(sentence);
        }
        if !current.is_empty() { chunks.push(current); }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_chunker() {
        let chunker = SentenceChunker {
            chunk_size: 20,
            overlap: 5,
        };
        let text = "This is sentence one. This is sentence two. And three.";
        let chunks = chunker.chunk(text);
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "This is sentence one. ");
        assert_eq!(chunks[1], "one. This is sentence two. ");
        assert_eq!(chunks[2], "two. And three.");
    }
}