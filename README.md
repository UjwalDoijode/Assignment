# Multi-Agent Deep Research System

A corpus-agnostic, multi-agent deep research system that answers complex natural language questions over private knowledge bases using LangGraph orchestration.

## Features

- **Corpus-Agnostic Design**: Auto-discovers and indexes any PDFs and SQLite databases
- **Multi-Agent Architecture**: Orchestrated fan-out to specialized agents (RAG, SQL, Web)
- **Agentic RAG**: Query formulation, sufficiency checking, and iterative refinement
- **Hybrid Retrieval**: BM25 + dense embeddings with Reciprocal Rank Fusion
- **Centroid-Based Routing**: Auto-routes queries to relevant collections using embedding similarity
- **Verification**: Second-pass fact-checking of generated research briefs
- **Full Observability**: LangSmith cloud tracing + Rich console output + JSON trace files
- **Performance Modes**: Ultra-Fast (~5s), Balanced (~8-12s), Full Quality (~15-20s)

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `TAVILY_API_KEY`: Your Tavily API key (for web search)

### 3. Prepare Your Corpus

Drop your documents into the corpus folders:

```
corpus/
├── pdfs/               ← Drop your PDF files here
└── databases/          ← Drop your SQLite .db or .sqlite files here
```

### 4. Index Your Corpus

```bash
python ingest.py
```

This will:
- Parse all PDFs (text + tables)
- Embed and index to ChromaDB + BM25
- Introspect all SQLite databases
- Generate collection metadata using LLM
- Write `registry.json`

### 5. Ask Questions

```bash
python main.py "What's the difference in battery life between the M-100 and M-300?"
```

The system will:
- Decompose your question into sub-questions
- Route to appropriate agents (RAG, SQL, Web)
- Retrieve evidence with citations
- Synthesize a research brief
- Verify claims against sources
- Print results to console
- Save JSON trace to `traces/`

## API Server

Start the API server:

```bash
pip install fastapi "uvicorn[standard]"
python server.py
```

The server will start on `http://localhost:8000`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/research` | Submit a research question |
| GET | `/health` | Health check + registry status |
| GET | `/collections` | List indexed document collections |

### Example

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the battery life of the M-300?", "mode": "balanced"}'
```

**Performance Modes:**
- `ultra_fast`: Direct retrieval, no verification (~5-7s)
- `balanced`: Agentic retrieval, no verification (~8-12s)
- `full_quality`: Full agentic loop + verification (~15-20s)

**Interactive API Docs:**  
Visit `http://localhost:8000/docs` for Swagger UI with live endpoint testing.

## Architecture

```
┌──────────────┐
│ Orchestrator │  ← Decomposes question into sub-questions
└──────┬───────┘
       │
       ├─────────┬──────────┬─────────┐
       ▼         ▼          ▼         ▼
  ┌────────┐ ┌─────────┐ ┌──────┐ ┌──────────┐
  │  RAG   │ │   SQL   │ │ Web  │ │ Compute  │
  │ Agent  │ │  Agent  │ │Agent │ │   Tool   │
  └────┬───┘ └────┬────┘ └───┬──┘ └──────────┘
       │          │           │
       └──────────┴───────────┘
                  │
            ┌─────▼──────┐
            │ Synthesis  │  ← Compose + verify brief
            └────────────┘
```

### RAG Agent Flow

```
Formulate Query → Retrieve (BM25 + Dense → RRF) → Check Sufficiency
       ▲                                                    │
       │                                                    │
       └────────────────── Retry (max 3) ─────────────────┘
                           if insufficient
```

## Example Questions

```bash
# Product comparisons
python main.py "Compare battery life and payload capacity of M-100 vs M-300"

# Financial queries
python main.py "What was Q4 FY2025 revenue and how does it compare to Q3?"

# Multi-source queries
python main.py "Which model has the best price-to-performance ratio based on sales data?"
```

## Project Structure

```
multi-agent-research/
├── corpus/
│   ├── pdfs/               ← Your PDF documents
│   └── databases/          ← Your SQLite databases
├── src/
│   ├── agents/             ← Agent implementations
│   │   ├── orchestrator.py
│   │   ├── rag_agent.py
│   │   ├── sql_agent.py
│   │   ├── web_agent.py
│   │   └── synthesis.py
│   ├── ingestion/          ← PDF parsing and indexing
│   │   ├── pdf_parser.py
│   │   └── indexer.py
│   ├── retrieval/          ← Hybrid search
│   │   └── hybrid.py
│   ├── tools/              ← LangChain tools
│   │   ├── kb_search.py
│   │   ├── sql_query.py
│   │   ├── compute.py
│   │   └── web_search.py
│   ├── config.py           ← Settings management
│   ├── models.py           ← Pydantic state models
│   ├── registry.py         ← Collection registry
│   ├── tracer.py           ← Observability
│   └── graph.py            ← LangGraph orchestration
├── ingest.py               ← Ingestion CLI
├── main.py                 ← Research CLI
├── requirements.txt
└── .env.example
```

## How It Works

### 1. Ingestion (`ingest.py`)

- Scans `corpus/pdfs/` and `corpus/databases/`
- For each PDF:
  - Extracts text + tables (tables → markdown)
  - Chunks text (400 words, 60-word overlap)
  - Embeds with sentence-transformers
  - Upserts to ChromaDB collection
  - Builds BM25 index
  - Computes centroid embedding
  - Generates description + keywords via LLM
- For each SQLite DB:
  - Introspects schema via `PRAGMA table_info`
  - Samples 3 rows per table
- Writes `registry.json`

### 2. Research (`main.py`)

1. **Orchestrator**: Decomposes question → sub-questions with intent routing
2. **Agent Fan-Out**: Routes sub-questions to appropriate agents:
   - `kb_lookup` → RAG agent
   - `sql_query` → SQL agent
   - `web_search` → Web agent
   - `compute` → Compute tool
3. **RAG Agent** (agentic loop):
   - Formulates optimized retrieval query
   - Executes hybrid search (BM25 + dense → RRF)
   - Checks sufficiency
   - Retries up to 3 times if insufficient
4. **SQL Agent**:
   - Converts NL → SQL using schema from registry
   - Executes SELECT query
   - Returns formatted results
5. **Synthesis**:
   - Composes research brief with inline citations
   - Verifies claims against evidence
   - Returns final brief

### 3. Registry-Based Routing

- Each collection has a centroid embedding (mean of all chunk embeddings)
- Query routing uses cosine similarity: `query_embedding · centroid`
- Threshold: 0.25, Top-k: 2 collections
- No hardcoded keyword lists — fully dynamic

### 4. Hybrid Retrieval (RRF)

```python
# Reciprocal Rank Fusion
score(doc) = Σ (1 / (k + rank_in_list))
```

- BM25 for exact keyword matches
- Dense embeddings for semantic similarity
- RRF combines both rankings (k=60)
- Returns top 5 chunks with citations

## Verification

The synthesis agent performs two LLM passes:

1. **Generation**: Compose research brief from evidence
2. **Verification**: Check each claim against source text

Verification catches:
- Hallucinated claims
- Incorrect numbers
- Wrong source attributions

## Observability

### Console Output (Rich)

- Real-time agent progress
- Sub-question decomposition table
- Evidence retrieval counts
- Final brief with markdown rendering
- Verification status

### JSON Trace

All execution saved to `traces/trace_YYYYMMDD_HHMMSS.json`:

```json
{
  "question": "...",
  "start_time": "...",
  "events": [
    {"type": "orchestrator", "data": {...}},
    {"type": "rag_agent_start", "data": {...}},
    {"type": "rag_agent_result", "data": {...}},
    ...
  ],
  "duration_seconds": 12.34
}
```

## Configuration

All settings in `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `LLM_MODEL` | Model to use | `openai/gpt-3.5-turbo` |
| `TAVILY_API_KEY` | Tavily API key | Required for web search |
| `LANGSMITH_API_KEY` | LangSmith API key | Optional (for tracing) |
| `LANGCHAIN_PROJECT` | LangSmith project name | `valiance-research` |
| `MAX_RAG_ITERATIONS` | Max retrieval retries | `1` (optimized) |
| `DIRECT_RETRIEVAL` | Skip agentic query loop | `true` (optimized) |
| `SKIP_VERIFICATION` | Skip verification pass | `true` (optimized) |
| `EMBEDDING_MODEL` | Local embedding model | `all-MiniLM-L6-v2` |
| `CHROMA_PATH` | ChromaDB storage | `./chroma_db` |

## Extending the System

### Add a New Tool

1. Create `src/tools/your_tool.py`
2. Use `@tool` decorator
3. Import in agent that needs it

### Add a New Agent

1. Create `src/agents/your_agent.py`
2. Implement node function: `def your_agent_node(state: ResearchState) -> ResearchState`
3. Add node to graph in `src/graph.py`
4. Add routing logic

### Add a New Source Type

1. Update `SourceType` enum in `src/models.py`
2. Add ingestion logic in `ingest.py`
3. Add retrieval logic in appropriate agent
4. Update registry schema

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### 📖 [ARCHITECTURE.md](docs/ARCHITECTURE.md)
Complete system architecture with flowcharts:
- High-level overview and component diagrams
- **Detailed LangGraph execution flow with Mermaid diagrams**
- Node-by-node breakdown of all agents
- State management and routing logic
- Performance characteristics and bottlenecks
- Technology stack details

### 💻 [CODE_GUIDE.md](docs/CODE_GUIDE.md)
Implementation details for developers:
- Configuration system walkthrough
- Pydantic state models explained
- Registry and centroid routing implementation
- Document ingestion pipeline
- Hybrid retrieval (RRF) algorithm
- LangGraph definition and routing
- All agent implementations with code samples
- Extension points for customization

### ⚡ [PERFORMANCE.md](docs/PERFORMANCE.md)
Performance optimization guide:
- Quick wins to reduce latency
- Ultra-Fast, Balanced, and Full Quality modes
- Latency breakdown by component
- Configuration examples
- Testing your setup

### 📊 [LANGSMITH.md](docs/LANGSMITH.md)
LangSmith observability integration:
- Complete trace visibility
- Token usage tracking
- Prompt engineering insights
- Debugging failed queries
- Configuration and usage

## Performance Tuning

Your system is currently optimized for **speed** with these settings in `.env`:

```bash
DIRECT_RETRIEVAL=true        # Skip agentic query formulation (~5-8s faster)
SKIP_VERIFICATION=true       # Skip verification LLM call (~2-4s faster)
MAX_RAG_ITERATIONS=1         # No retrieval retries (~3-8s faster)
```

**Current Performance**: ~5-11 seconds per query  
**Previous Performance**: ~15-20 seconds (full quality mode)

### Performance Modes

| Mode | Latency | Configuration | Best For |
|------|---------|---------------|----------|
| **Ultra-Fast** ⚡ | ~5-7s | `DIRECT_RETRIEVAL=true`<br/>`SKIP_VERIFICATION=true`<br/>`MAX_RAG_ITERATIONS=1` | Demos, testing, interactive chat |
| **Balanced** ⚖️ | ~8-12s | `DIRECT_RETRIEVAL=false`<br/>`SKIP_VERIFICATION=true`<br/>`MAX_RAG_ITERATIONS=1` | Production (good accuracy + speed) |
| **Full Quality** 🎯 | ~15-20s | `DIRECT_RETRIEVAL=false`<br/>`SKIP_VERIFICATION=false`<br/>`MAX_RAG_ITERATIONS=3` | Critical research, max accuracy |

### Latency Breakdown (Ultra-Fast Mode)

```
Orchestrator:           2-3s   (question decomposition)
RAG Direct Retrieval:   1-2s   (skip query formulation loop)
Synthesis:              3-4s   (research brief generation)
Verification:           SKIP   (disabled)
─────────────────────────────
Total:                  ~5-11s
```

### Test Performance

```bash
# Run performance comparison test
python test_performance.py

# Or manually test with timing
python main.py "What models does Meridian offer?"
# Check "Duration: X.XXs" at the end
```

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for detailed optimization guide.

## Observability

### LangSmith Cloud Tracing

Your system is integrated with LangSmith for complete observability:

🔗 **Dashboard**: https://smith.langchain.com  
📊 **Project**: `valiance-research`

Every query execution creates a trace showing:
- Full LLM prompts and completions
- Token counts and costs
- Latency per LLM call
- Graph execution flow
- Error details

**To disable**: Set `LANGCHAIN_TRACING_V2=false` in `.env`

### Local Traces

All executions also saved to `traces/trace_*.json` with custom metadata.



### "Registry not found" error

Run `python ingest.py` first to index your corpus.

### No results from knowledge base

- Check that PDFs are in `corpus/pdfs/`
- Verify ingestion succeeded (check console output)
- Try a more specific query
- Check centroid routing threshold (default 0.25)

### SQL errors

- Verify database schema in `registry.json`
- Check that column names match your query
- Use `PRAGMA table_info(table_name)` to inspect schema

### Rate limits (OpenRouter)

The free tier has rate limits. Switch to a paid model or add delays between queries.

## Troubleshooting

### "Registry not found" error

Run `python ingest.py` first to index your corpus.

### No results from knowledge base

- Check that PDFs are in `corpus/pdfs/`
- Verify ingestion succeeded (check console output)
- Try a more specific query
- Check centroid routing threshold (default 0.25)

### SQL errors

- Verify database schema in `registry.json`
- Check that column names match your query
- Use `PRAGMA table_info(table_name)` to inspect schema

### Slow queries

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for optimization guide. Quick fix:
```bash
# Add to .env
DIRECT_RETRIEVAL=true
SKIP_VERIFICATION=true
MAX_RAG_ITERATIONS=1
```

### LangSmith traces not appearing

- Check `LANGSMITH_API_KEY` is set in `.env`
- Verify `LANGCHAIN_TRACING_V2=true`
- Check correct project name at https://smith.langchain.com

## Additional Resources

- 📖 **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system architecture with flowcharts
- 💻 **[CODE_GUIDE.md](docs/CODE_GUIDE.md)** - Implementation details for developers  
- ⚡ **[PERFORMANCE.md](docs/PERFORMANCE.md)** - Performance optimization guide
- 📊 **[LANGSMITH.md](docs/LANGSMITH.md)** - Observability and tracing setup
- 🎨 **[DESIGN.md](docs/DESIGN.md)** - Architecture decisions and tradeoffs

## License

MIT

## Citation

```bibtex
@software{multi_agent_research,
  title={Multi-Agent Deep Research System},
  author={Ujwal K Doijode},
  year={2026},
  url={https://github.com/UjwalDoijode/Assignment}
}
```
