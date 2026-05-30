import os
import time
import json
import streamlit as st
from vector_store import LocalVectorStoreManager
from rag_agent import AdaptiveRAG

# Page Configuration
st.set_page_config(
    page_title="Adaptive RAG - Agentic AI Architecture",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Mode, Glassmorphism, Animations)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    /* Root Variables & Theme Override */
    :root {
        --bg-gradient: linear-gradient(135deg, #0f0c1b 0%, #15102a 50%, #05050f 100%);
        --card-bg: rgba(22, 19, 39, 0.45);
        --card-border: rgba(138, 43, 226, 0.15);
        --accent-glow: 0 8px 32px 0 rgba(138, 43, 226, 0.2);
        --text-primary: #f1f0f7;
        --text-muted: #9f9bbd;
        --color-success: #00ff88;
        --color-warning: #ffb700;
        --color-error: #ff4a5a;
        --color-running: #8a2be2;
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-primary);
    }
    
    /* Gradient Title */
    .hero-title {
        background: linear-gradient(90deg, #ff007f 0%, #8a2be2 50%, #00f0ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        text-shadow: 0 0 40px rgba(138, 43, 226, 0.3);
    }
    
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Custom Card Style */
    .premium-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: var(--accent-glow);
        transition: all 0.3s ease;
    }
    
    .premium-card:hover {
        border-color: rgba(138, 43, 226, 0.35);
        box-shadow: 0 12px 40px 0 rgba(138, 43, 226, 0.35);
        transform: translateY(-2px);
    }
    
    /* Pipeline Step Cards */
    .pipeline-step {
        padding: 14px 18px;
        margin-bottom: 12px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.4s ease;
    }
    
    .step-idle {
        background: rgba(255, 255, 255, 0.02);
        opacity: 0.4;
        border-left: 4px solid rgba(255, 255, 255, 0.1);
    }
    
    .step-running {
        background: rgba(138, 43, 226, 0.15);
        border-left: 4px solid var(--color-running);
        border-color: rgba(138, 43, 226, 0.4);
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.2);
        animation: pulseBorder 1.5s infinite alternate;
    }
    
    .step-completed {
        background: rgba(0, 255, 136, 0.06);
        border-left: 4px solid var(--color-success);
        border-color: rgba(0, 255, 136, 0.2);
    }
    
    .step-number {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 12px;
    }
    
    .step-name {
        font-weight: 600;
        flex-grow: 1;
        font-size: 0.95rem;
    }
    
    .step-status-pill {
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-idle { background: rgba(255,255,255,0.1); color: var(--text-muted); }
    .status-running { background: var(--color-running); color: white; }
    .status-completed { background: rgba(0, 255, 136, 0.2); color: var(--color-success); }
    
    /* Trace Details Box */
    .trace-details {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        color: #c5c2e0;
        white-space: pre-wrap;
    }
    
    /* final answer */
    .answer-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--color-success);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .answer-text {
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--text-primary);
    }
    
    /* Animations */
    @keyframes pulseBorder {
        0% { border-color: rgba(138, 43, 226, 0.3); }
        100% { border-color: rgba(138, 43, 226, 0.8); }
    }
</style>
""", unsafe_allow_html=True)

# Cache Vector Database to prevent reloading embedding model on every render
@st.cache_resource
def get_vector_store():
    # Model: all-MiniLM-L6-v2 runs fast locally (384-dim embeddings)
    manager = LocalVectorStoreManager()
    # Check if empty and initialize defaults
    manager.initialize_with_defaults()
    return manager

vector_store_manager = get_vector_store()

# Sidebar Setup
st.sidebar.markdown("<h2 style='font-family: Outfit; font-weight: 800; color: #8a2be2;'>⚙️ Configuration</h2>", unsafe_allow_html=True)

# 1. Select Model Provider
model_provider = st.sidebar.selectbox(
    "LLM Provider / Engine",
    ["Local / Mock Simulation", "Gemini API", "OpenAI API"],
    help="Select the backend LLM engine. Local Mode works out of the box without keys."
)

api_key = None
if model_provider == "Gemini API":
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key.")
    st.sidebar.caption("[Get a Gemini API key](https://aistudio.google.com/)")
elif model_provider == "OpenAI API":
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key.")
    st.sidebar.caption("[Get an OpenAI API key](https://platform.openai.com/api-keys)")

# Translate selection to engine modes
mode_map = {
    "Local / Mock Simulation": "mock",
    "Gemini API": "gemini",
    "OpenAI API": "openai"
}
selected_mode = mode_map[model_provider]

# RAG Params
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'/>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color: #9f9bbd;'>🛠️ RAG Hyperparameters</h4>", unsafe_allow_html=True)
similarity_threshold = st.sidebar.slider(
    "Relevance Threshold", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="Documents with a similarity score below this will trigger query rewriting and retrieval retries."
)
max_retries = st.sidebar.number_input("Max Retrieval Retries", min_value=1, max_value=5, value=2)

# Document Management in Sidebar
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'/>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='color: #9f9bbd;'>📂 Knowledge Base</h4>", unsafe_allow_html=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload Knowledge Files (.pdf, .txt)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    upload_clicked = st.sidebar.button("Ingest Uploaded Files", use_container_width=True)
    if upload_clicked:
        with st.sidebar.status("Processing and indexing documents...", expanded=True) as status:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            for f in uploaded_files:
                file_path = os.path.join(temp_dir, f.name)
                with open(file_path, "wb") as out_f:
                    out_f.write(f.getbuffer())
                
                status.write(f"Chunking and embedding `{f.name}`...")
                if f.name.endswith(".pdf"):
                    vector_store_manager.ingest_pdf(file_path)
                else:
                    with open(file_path, "r", encoding="utf-8") as txt_f:
                        text = txt_f.read()
                    vector_store_manager.ingest_text_content(text, f.name)
                
                # Cleanup temp file
                os.remove(file_path)
            
            # Remove temp dir
            try:
                os.rmdir(temp_dir)
            except:
                pass
                
            status.update(label="Ingestion complete! FAISS database updated.", state="complete")
            st.sidebar.success("Documents successfully added to Vector Store!")

# Display current documents count / sources
if vector_store_manager.vector_store:
    # Try to extract unique sources if available
    try:
        # FAISS indexes store document objects inside docstore
        docstore = vector_store_manager.vector_store.docstore
        sources = set(doc.metadata.get("source", "unknown") for doc in docstore._dict.values())
        st.sidebar.markdown(f"**Indexed Documents ({len(sources)}):**")
        for s in sorted(sources):
            st.sidebar.caption(f"📄 {s}")
    except:
        st.sidebar.caption("FAISS database contains active index.")

# Reset DB button
st.sidebar.markdown("<hr style='border-color: rgba(255,255,255,0.05);'/>", unsafe_allow_html=True)
if st.sidebar.button("Rebuild / Reset Database", use_container_width=True, help="Re-initializes the database back to default documents."):
    # Clear index directory
    import shutil
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    vector_store_manager.vector_store = None
    vector_store_manager.initialize_with_defaults()
    st.sidebar.success("Database reset to defaults!")
    st.rerun()

# Main Interface Layout
col_header, col_status = st.columns([3, 1])
with col_header:
    st.markdown("<h1 class='hero-title'>Adaptive RAG</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>Interactive Agentic RAG Architecture with Live Intent Routing & Retrieval Self-Correction</p>", unsafe_allow_html=True)
with col_status:
    # Status card
    provider_pill = f"<span style='background-color:#8a2be2; padding:6px 12px; border-radius:30px; font-weight:600; font-size:0.85rem;'>Engine: {model_provider}</span>"
    st.markdown(f"<div style='text-align: right; padding-top: 15px;'>{provider_pill}</div>", unsafe_allow_html=True)

# Query Input
query = st.text_input(
    "Ask a question to the Adaptive RAG Agent:", 
    placeholder="e.g. 'What is the cargo capacity of SpaceX Starship?' or 'Hello! How are you today?'",
    key="query_input"
)

# Helper to convert numpy numeric types to native Python types for JSON serialization
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    else:
        return obj

# Helper function to render a workflow step card
def render_step_card(step_num: int, step_name: str, current_step: int, completed_steps: list, data: dict = None, log_msg: str = ""):
    card_class = "step-idle"
    status_text = "Idle"
    status_class = "status-idle"
    
    if step_num == current_step:
        card_class = "step-running"
        status_text = "Running"
        status_class = "status-running"
    elif step_num in completed_steps:
        card_class = "step-completed"
        status_text = "Completed"
        status_class = "status-completed"
        
    st.markdown(f"""
    <div class="pipeline-step {card_class}">
        <div>
            <span class="step-number">{step_num:02d}</span>
            <span class="step-name">{step_name}</span>
        </div>
        <span class="step-status-pill {status_class}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if (step_num == current_step or step_num in completed_steps) and (data or log_msg):
        # Render details container
        details = ""
        if log_msg:
            details += f"Status: {log_msg}\n"
        if data:
            serializable_data = make_json_serializable(data)
            details += f"Data:\n{json.dumps(serializable_data, indent=2)}"
        st.markdown(f'<div class="trace-details">{details}</div>', unsafe_allow_html=True)

# Main Application logic
if query:
    # Validate API keys if selected
    if selected_mode in ["gemini", "openai"] and not api_key:
        st.error(f"Please provide a valid API Key in the sidebar to run using the {model_provider} engine.")
    else:
        # Create columns for Execution Flow (Right) and Final Result + Logs (Left)
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            st.markdown("<h3 style='margin-bottom:15px; color:#8a2be2;'>⚙️ Run Details & Output</h3>", unsafe_allow_html=True)
            output_container = st.empty()
            with output_container.container():
                st.info("Initializing Agentic RAG pipeline execution...")
        
        with col_right:
            st.markdown("<h3 style='margin-bottom:15px; color:#8a2be2;'>🧭 Interactive Flow Visualizer</h3>", unsafe_allow_html=True)
            flow_container = st.empty()
            
        # Initialize RAG agent
        rag = AdaptiveRAG(vector_store_manager, mode=selected_mode, api_key=api_key)
        rag.relevance_threshold = similarity_threshold
        rag.max_retries = max_retries
        
        # State tracking
        current_step = 1
        completed_steps = []
        step_logs = {}
        step_messages = {}
        final_answer_data = None
        need_retrieval_decision = None
        
        # Flow diagram mappings
        workflow_steps = [
            (1, "1. User Query Received"),
            (2, "2. Query Analyzer / Intent Classifier"),
            (3, "3. Need Retrieval? (Routing Decision)"),
            (4, "4. LLM Direct Response"),
            (5, "5. Query Rewriting / Planning"),
            (6, "6. Retriever (Query Vector DB)"),
            (8, "8. Retrieved Context Document Analysis"),
            (9, "9. Context Evaluation / Relevance Check"),
            (10, "10. LLM Generator"),
            (11, "11. Final Answer Formulation")
        ]
        
        # Helper function to refresh the visualizer column
        def refresh_visualizer():
            with flow_container.container():
                # Outer wrapper container
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                
                # Render steps based on path taken
                for step_idx, step_name in workflow_steps:
                    # Skip steps that are not in the path taken
                    if need_retrieval_decision is False and step_idx in [5, 6, 8, 9, 10]:
                        continue
                    if need_retrieval_decision is True and step_idx == 4:
                        continue
                        
                    render_step_card(
                        step_num=step_idx,
                        step_name=step_name,
                        current_step=current_step,
                        completed_steps=completed_steps,
                        data=step_logs.get(step_idx),
                        log_msg=step_messages.get(step_idx, "")
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        # Refresh initial flow
        refresh_visualizer()
        
        # Execute generator and consume steps
        try:
            for step_update in rag.run(query):
                step_idx = step_update["step"]
                name = step_update["name"]
                status = step_update["status"]
                msg = step_update["message"]
                data = step_update["data"]
                
                current_step = step_idx
                step_messages[step_idx] = msg
                
                if status == "completed":
                    if step_idx not in completed_steps:
                        completed_steps.append(step_idx)
                    step_logs[step_idx] = data
                    
                    # Capture RAG routing decision to update flowchart shape
                    if step_idx == 3:
                        need_retrieval_decision = data.get("need_retrieval")
                    
                    # Capture final answer
                    if step_idx == 11:
                        final_answer_data = data
                
                # Animate transition
                refresh_visualizer()
                time.sleep(0.5)
                
            # Process complete, update output panel
            with col_left:
                output_container.empty()
                with output_container.container():
                    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                    if final_answer_data:
                        st.markdown("""
                        <div class="answer-header">
                            <span>✅ Final Agentic RAG Answer</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<div class='answer-text'>{final_answer_data['answer']}</div>", unsafe_allow_html=True)
                        
                        # Display sources if any
                        sources = final_answer_data.get("sources", [])
                        if sources:
                            st.markdown("<br/><b>Sources Cited:</b>", unsafe_allow_html=True)
                            for s in sources:
                                st.markdown(f"🔹 `{s}`", unsafe_allow_html=True)
                    else:
                        st.error("No answer was returned by the pipeline.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display performance summary in left column
                    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                    st.markdown("<b>⚡ Performance Statistics:</b>", unsafe_allow_html=True)
                    st.markdown(f"- **Routing Route**: `{'Retrieval Path' if need_retrieval_decision else 'Direct LLM Path'}`")
                    if need_retrieval_decision:
                        # Find attempts from logs
                        attempts = sum(1 for k in step_logs.keys() if "Attempt" in str(workflow_steps)) # simplified
                        st.markdown(f"- **Relevance Threshold**: `{similarity_threshold}`")
                        # Show if retries were triggered
                        retry_nodes = [k for k, v in step_logs.items() if k == 9]
                        retry_count = len(retry_nodes)
                        st.markdown(f"- **Retrieval Attempts**: `{retry_count}`")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")
            import traceback
            st.code(traceback.format_exc())

# Information / Documentation Section at bottom
st.markdown("<br/><hr style='border-color: rgba(255,255,255,0.05);'/><br/>", unsafe_allow_html=True)
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    ### 🧠 About Adaptive RAG Architecture
    Standard RAG fetches documents and generates an answer in a single forward pass.
    **Adaptive RAG** makes the process dynamic and agentic by:
    1. **Analyzing the Query**: Routing conversational vs factual queries.
    2. **Query Rewriting**: Transforming user queries into optimized search keywords.
    3. **Evaluation Loop**: Automatically checking context relevance and triggering a retry loop with revised queries if the initial results are insufficient.
    """)

with col_info2:
    st.markdown("""
    ### 📂 Build custom context
    You can query the agent out-of-the-box on the following preloaded topics:
    * **SpaceX Starship** (specs, thrust, payload)
    * **DeepMind AlphaFold 3** (protein & molecular structures)
    * **Quantum Computing** (superposition, decoherence, error correction)
    * **Agentic RAG Flow** (architecture design explanations)
    
    *Or, upload your own `.txt` or `.pdf` files in the sidebar!*
    """)
