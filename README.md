# Adaptive RAG (Agentic RAG) Application

An interactive, production-ready implementation of the **Adaptive RAG Architecture** (Agentic AI) featuring dynamic intent classification, query rewriting, local FAISS vector indexing, and automatic context relevance self-correction loops.

Designed with a premium glassmorphism dark-mode Streamlit dashboard that visualizes each step of the agentic pipeline in real time.

---

## 🚀 Key Features

- **Dynamic Query Routing**: Automatically classifies if a query is conversational (routes to Direct Response) or technical (routes to Retrieval).
- **Self-Correction & Relevance Loops**: If retrieved document chunks are irrelevant, the agent automatically rewrites the query, increases keyword width, and retries search (up to a configured max retry limit) before generating the final response.
- **Local FAISS Vector DB**: Uses a local HuggingFace embedding model (`all-MiniLM-L6-v2` via `sentence-transformers`) for lightning-fast embeddings with zero external API key requirements.
- **Multiple LLM Backends**:
  - **Local / Mock Mode**: Runs fully offline out of the box using smart heuristics (great for testing/verification with zero API keys).
  - **Gemini API**: Full support using the new `google-genai` SDK.
  - **OpenAI API**: Full support using the `openai` SDK.
- **Custom Knowledge Ingestion**: Easily upload PDFs or TXT documents via the sidebar to expand the vector store in real-time.
- **Interactive Visual Tracer**: A sleek vertical flowchart that highlights active steps in real-time, displays JSON trace logs, and shows confidence scores.

---

## 📂 Codebase Structure

- `app.py`: Streamlit frontend dashboard with custom CSS, visual status cards, and layout grid.
- `rag_agent.py`: Pipeline engine implementing the `AdaptiveRAG` workflow, intent router, rewriter, context evaluator, and LLM providers.
- `vector_store.py`: Local vector storage manager using `FAISS` and character chunking. Contains preloaded technical documents (SpaceX, AlphaFold, Quantum Computing, Agentic RAG).
- `test_rag.py`: Programmatic verification suite that validates pipeline routes and self-correction loops.
- `requirements.txt`: Project dependencies list.

---

## 🛠️ Installation & Setup

1. **Clone or Navigate to the Directory**:
   ```bash
   cd "c:\Users\Abhishek\OneDrive\Desktop\agentic RAG"
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.10+ installed. Install requirements using:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ How to Run

### 1. Launch the Web Dashboard (Streamlit)
To start the interactive UI:
```bash
streamlit run app.py
```
This will open the application in your browser (typically at `http://localhost:8501`).

*Note:* If you don't have API keys, keep the engine set to **Local / Mock Simulation** in the sidebar. You can instantly test queries like:
- `"What is SpaceX Starship?"` (Retrieval route)
- `"Hello, how are you today?"` (Direct route)
- `"Tell me about quantum computing cargo capacity"` (Triggers retrieval relevance retry loop)

### 2. Run Automated Verification Tests
To run the programmatic test cases verifying intent routing and retry loops:
```bash
python test_rag.py
```

---

## ⚙️ How it Works (Under the Hood)

The workflow strictly matches the **Adaptive RAG Architecture**:

1. **User Query**: Receives query in Streamlit input.
2. **Intent Classifier**: Inspects if the query needs external database facts.
3. **Route Decisions**:
   - **No**: Directly calls the LLM with user prompt.
   - **Yes**: Begins retrieval.
4. **Query Rewriter**: Rewrites query for optimal vector search.
5. **Retriever**: Queries the local FAISS index (built on Sentence-Transformers embeddings).
6. **Relevance Auditor**: Grades document relevance from `0.0` to `1.0`.
   - **Under Threshold (< 0.45)**: Loops back to **Query Rewriter** to widen and retry query keywords.
   - **Over Threshold (>= 0.45)**: Continues.
7. **LLM Generator**: Answers user prompt utilizing only relevant facts.
8. **Final Answer**: Displays response along with clickable sources.
