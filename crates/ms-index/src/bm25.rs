use tantivy::query::{BooleanQuery, QueryParser, TermQuery, Occur};
use tantivy::collector::TopDocs;
use tantivy::schema::*;
use tantivy::{Index, IndexReader};

pub struct BM25Result {
    pub id: String,
    pub text: String,
    pub score: f32,
}

pub fn build_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("id",   STRING | STORED);   // UUID
    builder.add_text_field("text", TEXT | STORED);     // chunk text (BM25 indexed)
    builder.add_text_field("mode", STRING | STORED);   // filter by mode
    builder.add_text_field("source", STRING | STORED); // source file path
    builder.add_u64_field("chunk_index", STORED);
    builder.build()
}

pub struct BM25Index {
    pub index: Index,
    pub schema: Schema,
    pub reader: IndexReader,
}

impl BM25Index {
    pub fn search(&self, query: &str, mode: &str, top_k: usize) -> tantivy::Result<Vec<BM25Result>> {
        let searcher = self.reader.searcher();
        let text_field = self.schema.get_field("text").unwrap();
        let mode_field = self.schema.get_field("mode").unwrap();

        let query_parser = QueryParser::for_index(&self.index, vec![text_field]);
        let text_query = query_parser.parse_query(query)?;
        
        let mode_query = TermQuery::new(
            Term::from_field_text(mode_field, mode),
            IndexRecordOption::Basic
        );
        
        let combined = BooleanQuery::new(vec![
            (Occur::Must, text_query),
            (Occur::Must, Box::new(mode_query)),
        ]);

        let top_docs = searcher.search(&combined, &TopDocs::with_limit(top_k))?;
        
        let mut results = Vec::new();
        let id_field = self.schema.get_field("id").unwrap();
        let text_field = self.schema.get_field("text").unwrap();
        
        for (score, doc_address) in top_docs {
            let doc = searcher.doc(doc_address)?;
            let id = doc.get_first(id_field)
                .and_then(|v| v.as_text())
                .unwrap_or_default()
                .to_string();
            let text = doc.get_first(text_field)
                .and_then(|v| v.as_text())
                .unwrap_or_default()
                .to_string();
            
            results.push(BM25Result {
                id,
                text,
                score,
            });
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::doc;

    #[test]
    fn test_bm25_search() -> tantivy::Result<()> {
        let schema = build_schema();
        let index = Index::create_in_ram(schema.clone());
        let mut index_writer = index.writer(15_000_000)?;

        let id_field = schema.get_field("id").unwrap();
        let text_field = schema.get_field("text").unwrap();
        let mode_field = schema.get_field("mode").unwrap();
        let source_field = schema.get_field("source").unwrap();
        let chunk_index_field = schema.get_field("chunk_index").unwrap();

        index_writer.add_document(doc!(
            id_field => "doc1",
            text_field => "The quick brown fox jumps over the lazy dog",
            mode_field => "default",
            source_field => "test.txt",
            chunk_index_field => 0u64
        ))?;
        
        index_writer.add_document(doc!(
            id_field => "doc2",
            text_field => "A fast brown fox",
            mode_field => "default",
            source_field => "test2.txt",
            chunk_index_field => 0u64
        ))?;
        
        index_writer.add_document(doc!(
            id_field => "doc3",
            text_field => "The quick brown fox jumps over the lazy dog",
            mode_field => "other",
            source_field => "test3.txt",
            chunk_index_field => 0u64
        ))?;

        index_writer.commit()?;

        let reader = index.reader()?;
        let bm25_index = BM25Index {
            index,
            schema,
            reader,
        };

        let results = bm25_index.search("fox", "default", 10)?;
        assert_eq!(results.len(), 2);
        
        // Let's verify we don't get the one with mode "other"
        let ids: Vec<String> = results.into_iter().map(|r| r.id).collect();
        assert!(ids.contains(&"doc1".to_string()));
        assert!(ids.contains(&"doc2".to_string()));
        
        let results_other = bm25_index.search("fox", "other", 10)?;
        assert_eq!(results_other.len(), 1);
        assert_eq!(results_other[0].id, "doc3");

        Ok(())
    }
}