use std::collections::{HashMap, HashSet};

/// A dictionary entry with a term and its frequency count.
#[derive(Debug, Clone)]
pub struct DictEntry {
    pub term: String,
    pub count: u64,
}

/// A suggestion returned by SymSpell lookup.
#[derive(Debug, Clone)]
pub struct SuggestItem {
    pub term: String,
    pub distance: usize,
    pub count: u64,
}

/// SymSpell-style spelling corrector.
///
/// Works by pre-computing "delete edits" of dictionary words up to a
/// configurable maximum edit distance. At query time it generates delete
/// edits of the input word and performs O(1) HashMap lookups to find
/// candidate corrections, which are then validated with
/// Damerau-Levenshtein distance.
pub struct SymSpell {
    max_edit_distance: usize,
    dictionary: HashMap<String, DictEntry>,
    deletes: HashMap<String, Vec<String>>,
}

impl SymSpell {
    /// Creates a new, empty `SymSpell` corrector.
    ///
    /// `max_edit_distance` is the maximum number of edits considered when
    /// building the delete index (typically 2).
    pub fn new(max_edit_distance: usize) -> Self {
        Self {
            max_edit_distance,
            dictionary: HashMap::new(),
            deletes: HashMap::new(),
        }
    }

    /// Loads a dictionary of (word, frequency) pairs into the corrector.
    ///
    /// For each word the method generates all delete variants up to
    /// `max_edit_distance` and indexes them for fast lookup.
    pub fn load_dictionary(&mut self, entries: &[(String, u64)]) {
        for (word, count) in entries {
            let lower = word.to_lowercase();
            self.dictionary.insert(
                lower.clone(),
                DictEntry {
                    term: lower.clone(),
                    count: *count,
                },
            );

            let delete_variants = generate_deletes(&lower, self.max_edit_distance);
            for variant in delete_variants {
                self.deletes
                    .entry(variant)
                    .or_default()
                    .push(lower.clone());
            }
        }
    }

    /// Looks up spelling suggestions for `input`.
    ///
    /// Returns candidates within `max_distance` edits, sorted by
    /// (distance ascending, frequency descending).
    pub fn lookup(&self, input: &str, max_distance: usize) -> Vec<SuggestItem> {
        let input_lower = input.to_lowercase();
        let effective_max = max_distance.min(self.max_edit_distance);

        let mut seen: HashSet<String> = HashSet::new();
        let mut suggestions: Vec<SuggestItem> = Vec::new();

        // 1. Check for an exact dictionary match.
        if let Some(entry) = self.dictionary.get(&input_lower) {
            seen.insert(entry.term.clone());
            suggestions.push(SuggestItem {
                term: entry.term.clone(),
                distance: 0,
                count: entry.count,
            });
        }

        // 2. Gather candidates from the delete index.
        let mut candidate_keys: HashSet<String> = HashSet::new();

        // The input itself might be a delete variant of a dictionary word.
        candidate_keys.insert(input_lower.clone());

        // Generate delete variants of the input.
        let input_deletes = generate_deletes(&input_lower, effective_max);
        for d in &input_deletes {
            candidate_keys.insert(d.clone());
        }

        for key in &candidate_keys {
            if let Some(originals) = self.deletes.get(key) {
                for original in originals {
                    if seen.contains(original) {
                        continue;
                    }
                    let dist = damerau_levenshtein(&input_lower, original);
                    if dist <= effective_max {
                        seen.insert(original.clone());
                        if let Some(entry) = self.dictionary.get(original) {
                            suggestions.push(SuggestItem {
                                term: entry.term.clone(),
                                distance: dist,
                                count: entry.count,
                            });
                        }
                    }
                }
            }
        }

        // 3. Sort: distance ascending, then count descending.
        suggestions.sort_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then_with(|| b.count.cmp(&a.count))
        });

        suggestions
    }

    /// Corrects every word in `query` independently and reassembles the
    /// result.
    ///
    /// Words that have no suggestions within the max edit distance are
    /// kept unchanged.
    pub fn correct_query(&self, query: &str) -> String {
        query
            .split_whitespace()
            .map(|word| {
                let suggestions = self.lookup(word, self.max_edit_distance);
                if let Some(best) = suggestions.first() {
                    best.term.clone()
                } else {
                    word.to_lowercase()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Computes the Damerau-Levenshtein distance between two strings.
///
/// This distance counts the minimum number of single-character insertions,
/// deletions, substitutions, and transpositions of two adjacent characters
/// needed to transform `a` into `b`.
fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    // dp[i][j] = distance between a[..i] and b[..j]
    let mut dp = vec![vec![0usize; b_len + 1]; a_len + 1];

    for i in 0..=a_len {
        dp[i][0] = i;
    }
    for j in 0..=b_len {
        dp[0][j] = j;
    }

    for i in 1..=a_len {
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1)     // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution

            // Transposition of two adjacent characters.
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + cost);
            }
        }
    }

    dp[a_len][b_len]
}

/// Recursively generates all strings obtainable by deleting up to
/// `max_distance` characters from `word`.
fn generate_deletes(word: &str, max_distance: usize) -> HashSet<String> {
    let mut results = HashSet::new();
    if max_distance == 0 || word.is_empty() {
        return results;
    }

    let chars: Vec<char> = word.chars().collect();
    for i in 0..chars.len() {
        // Build a string with the i-th character removed.
        let mut deleted = String::with_capacity(word.len() - 1);
        for (j, &ch) in chars.iter().enumerate() {
            if j != i {
                deleted.push(ch);
            }
        }

        results.insert(deleted.clone());

        // Recurse for deeper deletions.
        if max_distance > 1 {
            let deeper = generate_deletes(&deleted, max_distance - 1);
            results.extend(deeper);
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_symspell() -> SymSpell {
        let mut ss = SymSpell::new(2);
        ss.load_dictionary(&[
            ("research".to_string(), 100),
            ("search".to_string(), 200),
            ("memory".to_string(), 150),
            ("index".to_string(), 300),
            ("query".to_string(), 250),
            ("result".to_string(), 180),
            ("algorithm".to_string(), 90),
            ("database".to_string(), 120),
            ("vector".to_string(), 110),
            ("document".to_string(), 160),
        ]);
        ss
    }

    #[test]
    fn test_basic_correction() {
        let ss = build_test_symspell();
        let suggestions = ss.lookup("resarch", 2);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].term, "research");
    }

    #[test]
    fn test_exact_match() {
        let ss = build_test_symspell();
        let suggestions = ss.lookup("research", 2);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].term, "research");
        assert_eq!(suggestions[0].distance, 0);
    }

    #[test]
    fn test_multi_word_correction() {
        let ss = build_test_symspell();
        let corrected = ss.correct_query("resarch qurey");
        assert_eq!(corrected, "research query");
    }

    #[test]
    fn test_correct_words_unchanged() {
        let ss = build_test_symspell();
        let corrected = ss.correct_query("research query");
        assert_eq!(corrected, "research query");
    }

    #[test]
    fn test_damerau_levenshtein_distance() {
        assert_eq!(damerau_levenshtein("cat", "cat"), 0);
        assert_eq!(damerau_levenshtein("cat", "car"), 1);
        assert_eq!(damerau_levenshtein("cat", "ca"), 1);
        assert_eq!(damerau_levenshtein("cat", "cats"), 1);
        assert_eq!(damerau_levenshtein("cat", "act"), 1); // transposition
        assert_eq!(damerau_levenshtein("", "abc"), 3);
        assert_eq!(damerau_levenshtein("abc", ""), 3);
    }

    #[test]
    fn test_generate_deletes() {
        let deletes = generate_deletes("abc", 1);
        assert!(deletes.contains("ab"));
        assert!(deletes.contains("ac"));
        assert!(deletes.contains("bc"));
        assert_eq!(deletes.len(), 3);
    }

    #[test]
    fn test_generate_deletes_depth_two() {
        let deletes = generate_deletes("abc", 2);
        // Single deletions: ab, ac, bc
        assert!(deletes.contains("ab"));
        assert!(deletes.contains("ac"));
        assert!(deletes.contains("bc"));
        // Double deletions: a, b, c
        assert!(deletes.contains("a"));
        assert!(deletes.contains("b"));
        assert!(deletes.contains("c"));
        assert_eq!(deletes.len(), 6);
    }

    #[test]
    fn test_empty_input() {
        let ss = build_test_symspell();
        let suggestions = ss.lookup("", 2);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_no_match() {
        let ss = build_test_symspell();
        let suggestions = ss.lookup("xylophone", 2);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggestions_sorted_by_distance_then_count() {
        let mut ss = SymSpell::new(2);
        ss.load_dictionary(&[
            ("test".to_string(), 100),
            ("tast".to_string(), 50),
            ("text".to_string(), 200),
        ]);
        let suggestions = ss.lookup("tost", 2);
        // All should be distance 1 from "tost"
        // "text" has highest count among distance-1 matches
        if suggestions.len() >= 2 {
            assert!(suggestions[0].distance <= suggestions[1].distance);
            if suggestions[0].distance == suggestions[1].distance {
                assert!(suggestions[0].count >= suggestions[1].count);
            }
        }
    }
}
