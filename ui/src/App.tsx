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

interface PerArmResults {
  bm25: SearchResult[];
  fuzzy: SearchResult[];
  semantic: SearchResult[];
}

interface SearchResponse {
  results: SearchResult[];
  rag_answer?: RagAnswer;
  arm_results?: PerArmResults;
}

interface Stats {
  total_documents: number;
  total_chunks: number;
  modes: Record<string, number>;
  arms: Record<string, string>;
}

function App() {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('default');
  const [topK, setTopK] = useState(10);
  const [rerank, setRerank] = useState(false);
  const [rag, setRag] = useState(false);
  const [arms, setArms] = useState(false);
  const [activeTab, setActiveTab] = useState<'merged' | 'bm25' | 'fuzzy' | 'semantic'>('merged');
  
  const [results, setResults] = useState<SearchResult[]>([]);
  const [ragAnswer, setRagAnswer] = useState<RagAnswer | undefined>(undefined);
  const [armResults, setArmResults] = useState<PerArmResults | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [stats, setStats] = useState<Stats | null>(null);
  const [showStats, setShowStats] = useState(false);
  
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMode, setUploadMode] = useState('default');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');

  const fetchStats = async () => {
    try {
      const res = await fetch('http://localhost:8080/api/stats');
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error('Failed to fetch stats');
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResults([]);
    setRagAnswer(undefined);
    setArmResults(undefined);

    try {
      const params = new URLSearchParams({
        q: query,
        mode,
        top_k: topK.toString(),
        rerank: rerank.toString(),
        rag: rag.toString(),
        arms: arms.toString(),
      });

      const response = await fetch(`http://localhost:8080/api/search?${params}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
      setRagAnswer(data.rag_answer);
      setArmResults(data.arm_results);
      setActiveTab('merged');
    } catch (err: any) {
      setError(err.message || 'Failed to search');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) return;

    setIsUploading(true);
    setUploadMsg('');
    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('mode', uploadMode);

    try {
      const res = await fetch('http://localhost:8080/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setUploadMsg(data.message || 'Upload success');
      fetchStats();
    } catch (err: any) {
      setUploadMsg('Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>MemorySearch 🔍</h1>
        <p>A fast, local-first search engine with 3-arm retrieval and RAG capabilities.</p>
        <div className="header-actions">
          <button onClick={() => { setShowStats(!showStats); if (!showStats) fetchStats(); }}>
            📊 {showStats ? 'Hide Stats' : 'View DB Stats & Upload'}
          </button>
        </div>
      </header>

      <main className="main-content">
        {showStats && (
          <div className="stats-panel fade-in">
            <div className="stats-grid">
              <div className="stat-card">
                <h3>Global Index</h3>
                {stats ? (
                  <>
                    <p><strong>Documents:</strong> {stats.total_documents}</p>
                    <p><strong>Total Chunks:</strong> {stats.total_chunks}</p>
                  </>
                ) : <p>Loading...</p>}
              </div>
              <div className="stat-card">
                <h3>Indexed Modes</h3>
                {stats && Object.entries(stats.modes).map(([m, c]) => (
                  <p key={m}><strong>{m}:</strong> {c} chunks</p>
                ))}
              </div>
              <div className="stat-card" style={{ gridColumn: 'span 2' }}>
                <h3>Search Arms Health</h3>
                {stats && Object.entries(stats.arms).map(([arm, info]) => (
                  <p key={arm}><strong>{arm.toUpperCase()}:</strong> {info}</p>
                ))}
              </div>
            </div>

            <form className="upload-form" onSubmit={handleUpload}>
              <h3>Upload & Index New File</h3>
              <div className="upload-row">
                <input type="file" onChange={(e) => setUploadFile(e.target.files?.[0] || null)} />
                <select value={uploadMode} onChange={(e) => setUploadMode(e.target.value)}>
                  <option value="default">Default</option>
                  <option value="notes">Notes</option>
                  <option value="codebase">Codebase</option>
                  <option value="wikipedia">Wikipedia</option>
                </select>
                <button type="submit" disabled={isUploading || !uploadFile}>
                  {isUploading ? 'Uploading...' : 'Upload File'}
                </button>
              </div>
              {uploadMsg && <p className="upload-msg">{uploadMsg}</p>}
            </form>
          </div>
        )}

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

            <label className="toggle-label">
              <input
                type="checkbox"
                checked={arms}
                onChange={(e) => setArms(e.target.checked)}
              />
              Show Individual Arm Results
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

        {armResults && (
          <div style={{ display: 'flex', gap: '10px', margin: '20px 0', borderBottom: '2px solid rgba(255,255,255,0.1)', paddingBottom: '10px', flexWrap: 'wrap' }}>
            <button
              onClick={() => setActiveTab('merged')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'merged' ? '#6366f1' : '#1e293b', color: '#fff', cursor: 'pointer', fontWeight: 600 }}
            >
              🏆 {rerank ? 'Unique Re-ranked Results' : 'Merged Results'} ({results.length})
            </button>
            <button
              onClick={() => setActiveTab('bm25')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'bm25' ? '#3b82f6' : '#1e293b', color: '#fff', cursor: 'pointer', fontWeight: 600 }}
            >
              📝 BM25 Top ({armResults.bm25.length})
            </button>
            <button
              onClick={() => setActiveTab('fuzzy')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'fuzzy' ? '#f59e0b' : '#1e293b', color: '#fff', cursor: 'pointer', fontWeight: 600 }}
            >
              🔤 Fuzzy Top ({armResults.fuzzy.length})
            </button>
            <button
              onClick={() => setActiveTab('semantic')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'semantic' ? '#10b981' : '#1e293b', color: '#fff', cursor: 'pointer', fontWeight: 600 }}
            >
              🧠 Semantic Top ({armResults.semantic.length})
            </button>
          </div>
        )}

        <div className="results-container">
          {(() => {
            const displayedResults = activeTab === 'merged' ? results :
                                     activeTab === 'bm25' && armResults ? armResults.bm25 :
                                     activeTab === 'fuzzy' && armResults ? armResults.fuzzy :
                                     activeTab === 'semantic' && armResults ? armResults.semantic : results;
            return (
              <>
                {displayedResults.map((result, idx) => (
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

                {!isLoading && !error && query && displayedResults.length === 0 && (
                  <div className="no-results">No results found for "{query}".</div>
                )}
              </>
            );
          })()}
        </div>
      </main>
    </div>
  );
}

export default App;
