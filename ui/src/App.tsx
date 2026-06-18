import { useState } from 'react';
import './App.css';

type SearchArm = 'BM25' | 'Fuzzy' | 'Semantic';

interface SearchResult {
  chunk_id: string;
  text: string;
  score: number;
  arm_scores: Record<SearchArm, number>;
}

interface RagAnswer {
  answer: string;
  sources: SearchResult[];
  model: string;
}

interface SearchResponse {
  results: SearchResult[];
  rag_answer?: RagAnswer;
}

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('default');
  const [topK, setTopK] = useState(10);
  const [rerank, setRerank] = useState(false);
  const [rag, setRag] = useState(false);
  
  const [results, setResults] = useState<SearchResult[]>([]);
  const [ragAnswer, setRagAnswer] = useState<RagAnswer | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResults([]);
    setRagAnswer(undefined);

    try {
      const params = new URLSearchParams({
        q: query,
        mode,
        top_k: topK.toString(),
        rerank: rerank.toString(),
        rag: rag.toString(),
      });

      const response = await fetch(`http://localhost:8080/api/search?${params}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
      setRagAnswer(data.rag_answer);
    } catch (err: any) {
      setError(err.message || 'Failed to search');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>MemorySearch 🔍</h1>
        <p>A fast, local-first search engine with 3-arm retrieval and RAG capabilities.</p>
      </header>

      <main className="main-content">
        <form className="search-form" onSubmit={handleSearch}>
          <div className="search-bar-wrapper">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search your knowledge base..."
              className="search-input"
              autoFocus
            />
            <button type="submit" className="search-button" disabled={isLoading}>
              {isLoading ? <div className="spinner"></div> : 'Search'}
            </button>
          </div>

          <div className="filters-container">
            <div className="filter-group">
              <label>Mode:</label>
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="default">Default</option>
                <option value="notes">Notes</option>
                <option value="codebase">Codebase</option>
                <option value="wikipedia">Wikipedia</option>
              </select>
            </div>

            <div className="filter-group">
              <label>Top K:</label>
              <input
                type="number"
                min="1"
                max="50"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
              />
            </div>

            <label className="toggle-label">
              <input
                type="checkbox"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
              />
              Enable Cross-Encoder Re-ranker
            </label>

            <label className="toggle-label">
              <input
                type="checkbox"
                checked={rag}
                onChange={(e) => setRag(e.target.checked)}
              />
              Enable RAG (LLM Generation)
            </label>
          </div>
        </form>

        {error && <div className="error-message">{error}</div>}

        {ragAnswer && (
          <div className="rag-answer-card fade-in">
            <div className="rag-header">
              <h2>🤖 RAG Answer</h2>
              <span className="rag-model-badge">{ragAnswer.model}</span>
            </div>
            <p className="rag-text">{ragAnswer.answer}</p>
          </div>
        )}

        <div className="results-container">
          {results.map((result, idx) => (
            <div key={result.chunk_id || idx} className="result-card fade-in" style={{ animationDelay: `${idx * 0.05}s` }}>
              <div className="result-header">
                <span className="result-rank">#{idx + 1}</span>
                <span className="result-score">Score: {result.score.toFixed(4)}</span>
              </div>
              
              <div className="result-arms">
                {Object.entries(result.arm_scores).map(([arm, score]) => (
                  <span key={arm} className={`arm-badge arm-${arm.toLowerCase()}`}>
                    {arm}: {(score as number).toFixed(3)}
                  </span>
                ))}
              </div>

              <div className="result-text">
                {result.text}
              </div>
            </div>
          ))}

          {!isLoading && !error && query && results.length === 0 && (
            <div className="no-results">No results found for "{query}".</div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
