use tree_sitter::Parser;

pub struct CodeChunker {
    pub max_chunk_size: usize,
}

const RUST_TOP_LEVEL: &[&str] = &[
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "mod_item",
];

const PYTHON_TOP_LEVEL: &[&str] = &["function_definition", "class_definition"];

const JS_TOP_LEVEL: &[&str] = &[
    "function_declaration",
    "class_declaration",
    "arrow_function",
    "method_definition",
];

impl CodeChunker {
    pub fn chunk(&self, source: &str, language: &str) -> Vec<String> {
        let lang = match language {
            "rust" => Some(tree_sitter_rust::language()),
            "python" => Some(tree_sitter_python::language()),
            "javascript" => Some(tree_sitter_javascript::language()),
            _ => None,
        };

        let lang = match lang {
            Some(l) => l,
            None => return self.fallback_chunk(source),
        };

        let top_level_kinds: &[&str] = match language {
            "rust" => RUST_TOP_LEVEL,
            "python" => PYTHON_TOP_LEVEL,
            "javascript" => JS_TOP_LEVEL,
            _ => return self.fallback_chunk(source),
        };

        let mut parser = Parser::new();
        parser
            .set_language(lang)
            .expect("failed to set tree-sitter language");

        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return self.fallback_chunk(source),
        };

        let root = tree.root_node();
        let mut chunks = Vec::new();
        let mut interstitial = String::new();
        let mut last_end = 0;

        for i in 0..root.child_count() {
            let node = root.child(i).unwrap();
            let kind = node.kind();
            let node_start = node.start_byte();
            let node_end = node.end_byte();

            if top_level_kinds.contains(&kind) {
                let gap = &source[last_end..node_start];
                if !gap.trim().is_empty() {
                    interstitial.push_str(gap);
                }

                if !interstitial.trim().is_empty() {
                    self.push_chunks(&mut chunks, interstitial.trim().to_string());
                    interstitial.clear();
                }

                let node_text = &source[node_start..node_end];
                self.push_chunks(&mut chunks, node_text.to_string());
                last_end = node_end;
            } else {
                let gap = &source[last_end..node_end];
                interstitial.push_str(gap);
                last_end = node_end;
            }
        }

        if last_end < source.len() {
            let remaining = &source[last_end..];
            if !remaining.trim().is_empty() {
                interstitial.push_str(remaining);
            }
        }

        if !interstitial.trim().is_empty() {
            self.push_chunks(&mut chunks, interstitial.trim().to_string());
        }

        chunks
    }

    fn push_chunks(&self, chunks: &mut Vec<String>, text: String) {
        if text.len() <= self.max_chunk_size {
            chunks.push(text);
        } else {
            let chars: Vec<char> = text.chars().collect();
            let mut start = 0;
            while start < chars.len() {
                let end = (start + self.max_chunk_size).min(chars.len());
                let chunk: String = chars[start..end].iter().collect();
                chunks.push(chunk);
                start = end;
            }
        }
    }

    fn fallback_chunk(&self, source: &str) -> Vec<String> {
        if source.is_empty() {
            return Vec::new();
        }
        let chars: Vec<char> = source.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let end = (start + self.max_chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);
            start = end;
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_chunker_rust() {
        let chunker = CodeChunker {
            max_chunk_size: 5000,
        };
        let source = r#"
use std::io;

fn hello() {
    println!("hello");
}

struct Foo {
    x: i32,
}

fn world() {
    println!("world");
}
"#;
        let chunks = chunker.chunk(source, "rust");
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_code_chunker_python() {
        let chunker = CodeChunker {
            max_chunk_size: 5000,
        };
        let source = r#"
import os

def hello():
    print("hello")

class Foo:
    def bar(self):
        pass
"#;
        let chunks = chunker.chunk(source, "python");
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_code_chunker_unknown_language() {
        let chunker = CodeChunker {
            max_chunk_size: 20,
        };
        let source = "some random text that is longer than the max chunk size limit";
        let chunks = chunker.chunk(source, "cobol");
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_code_chunker_large_function_fallback() {
        let chunker = CodeChunker {
            max_chunk_size: 30,
        };
        let source = r#"fn very_long_function_name_here() {
    let x = 1;
    let y = 2;
    let z = 3;
}"#;
        let chunks = chunker.chunk(source, "rust");
        assert!(chunks.len() > 1);
    }
}
