import os
import glob
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader

# Custom LangChain-compatible wrapper for SentenceTransformers to ensure local, zero-API embedding generation
class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # This will download the model locally on first run (or load from cache if already downloaded)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()

class LocalVectorStoreManager:
    def __init__(self, index_dir: str = "faiss_index", model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.embeddings = LocalSentenceTransformerEmbeddings(model_name)
        self.vector_store = None
        
        # Load or initialize vector store
        self.load_index()

    def load_index(self):
        """Loads the FAISS index if it exists, otherwise leaves it as None."""
        if os.path.exists(self.index_dir) and os.path.isdir(self.index_dir):
            try:
                # FAISS load_local requires allow_dangerous_deserialization since FAISS pickles details
                self.vector_store = FAISS.load_local(
                    self.index_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from {self.index_dir}")
            except Exception as e:
                print(f"Error loading index: {e}. Reinitializing.")
                self.vector_store = None
        else:
            print("No existing FAISS index found. Ready for document ingestion.")

    def save_index(self):
        """Saves the FAISS index to the local index directory."""
        if self.vector_store:
            os.makedirs(self.index_dir, exist_ok=True)
            self.vector_store.save_local(self.index_dir)
            print(f"FAISS index saved to {self.index_dir}")

    def add_documents(self, documents: List[Document]):
        """Adds a list of LangChain documents to the vector store and saves it."""
        if not documents:
            return
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
        self.save_index()

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Performs similarity search and returns a list of (Document, score) tuples.
        
        Note: FAISS score is L2 distance, so LOWER score means HIGHER similarity.
        We will normalize the score to a similarity confidence between 0 and 1.
        """
        if self.vector_store is None:
            return []
        
        # similarity_search_with_score returns L2 distance
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        processed_results = []
        for doc, l2_dist in results:
            # L2 distance is typically 0 to 2 for normalized embeddings.
            # Convert to a relevance score where 1.0 is exact match, and 0.0 is different.
            similarity = float(max(0.0, min(1.0, 1.0 - (float(l2_dist) / 2.0))))
            processed_results.append((doc, similarity))
            
        return processed_results

    def split_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Simple, robust character-based splitter with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks

    def ingest_text_content(self, text: str, source_name: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Splits raw text content and adds chunks to the vector database."""
        chunks = self.split_text(text, chunk_size, chunk_overlap)
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"source": source_name, "chunk_id": i}
            )
            documents.append(doc)
        self.add_documents(documents)
        print(f"Ingested {len(documents)} chunks from {source_name}")

    def ingest_pdf(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Extracts text from a PDF file, splits, and ingests it."""
        try:
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            source_name = os.path.basename(file_path)
            self.ingest_text_content(full_text, source_name, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")

    def ingest_directory(self, dir_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Scans a directory for pdf, txt, and md files and indexes them."""
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            return

        # Scan for PDFs
        for file_path in glob.glob(os.path.join(dir_path, "*.pdf")):
            self.ingest_pdf(file_path, chunk_size, chunk_overlap)

        # Scan for TXT and MD files
        for ext in ("*.txt", "*.md"):
            for file_path in glob.glob(os.path.join(dir_path, ext)):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    source_name = os.path.basename(file_path)
                    self.ingest_text_content(text, source_name, chunk_size, chunk_overlap)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    def initialize_with_defaults(self):
        """Initializes the database with high-quality sample documents if it is empty."""
        need_rebuild = False
        if self.vector_store is not None:
            try:
                docstore = self.vector_store.docstore
                sources = set(doc.metadata.get("source", "unknown") for doc in docstore._dict.values())
                if "Block_Cipher_Principles.txt" not in sources:
                    need_rebuild = True
            except Exception:
                need_rebuild = True
        else:
            need_rebuild = True

        if not need_rebuild:
            return
            
        print("Initializing database with default sample documents...")
        self.vector_store = None
        
        samples = {
            "SpaceX_Starship.txt": (
                "SpaceX Starship is a fully reusable, two-stage super heavy-lift launch vehicle designed "
                "by SpaceX. It consists of the Super Heavy booster stage and the Starship second stage spacecraft. "
                "The system is designed to carry both crew and cargo to Earth orbit, the Moon, Mars, and beyond. "
                "Starship is powered by Raptor engines, which burn liquid methane (CH4) and liquid oxygen (LOX). "
                "The Starship spacecraft has a payload capacity of over 100 to 150 metric tons to Low Earth Orbit (LEO) "
                "in a fully reusable configuration. The booster, Super Heavy, has 33 Raptor engines, producing "
                "more than 17 million pounds of thrust, which is double the power of the Saturn V rocket. "
                "SpaceX aims to use Starship for its Starlink satellite deployments, the NASA Artemis moon landing program, "
                "and eventually colonizing Mars by establishing a self-sustaining city."
            ),
            "DeepMind_AlphaFold.txt": (
                "AlphaFold is an artificial intelligence program developed by Google DeepMind which performs "
                "predictions of protein structure. Proteins are essential molecular machines that run all living organisms. "
                "The 3D shape of a protein determines its function, and predicting this shape from its amino acid sequence "
                "is known as protein folding, a 50-year-old challenge in biology. "
                "In 2024, Google DeepMind and Isomorphic Labs announced AlphaFold 3, a revolutionary model that predicts "
                "the structure and interactions of proteins, DNA, RNA, chemical compounds (ligands), and chemical modifications. "
                "AlphaFold 3 employs a diffusion model architecture, similar to those used in AI image generators, to assemble "
                "the molecules. This helps scientists understand biological pathways, target disease mechanisms, and speed up "
                "drug discovery, saving years of experimental laboratory work."
            ),
            "Quantum_Computing_Intro.txt": (
                "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve "
                "problems too complex for classical computers. It utilizes quantum bits, or qubits, as the basic unit of information. "
                "Unlike classical bits which can only be 0 or 1, qubits can exist in a state of superposition, representing "
                "both 0 and 1 simultaneously. "
                "Another core quantum concept is entanglement, which links qubits such that the state of one instantly influences "
                "the state of another, regardless of distance. This allows quantum computers to process massive combinations "
                "of data in parallel. Key challenges in building quantum computers include quantum decoherence, where external noise "
                "causes qubits to lose their quantum states, and high error rates. Researchers use techniques like quantum error correction "
                "to stabilize the qubits, which often must be kept at temperatures close to absolute zero (millikelvin range) "
                "using dilution refrigerators."
            ),
            "Block_Cipher_Principles.txt": (
                "A block cipher is a symmetric cryptographic algorithm that encrypts a fixed-size block of plaintext "
                "into a ciphertext block of the same size using a shared secret key. The standard block size "
                "is typically 64 or 128 bits. The fundamental principles of modern block cipher design, "
                "first outlined by Claude Shannon, are diffusion and confusion. "
                "Confusion hides the relationship between the ciphertext and the symmetric key, usually achieved via "
                "substitution boxes (S-boxes) which perform non-linear mappings. "
                "Diffusion spreads the influence of single plaintext bits over many ciphertext bits to obscure statistical "
                "patterns, commonly achieved through permutations and linear transformations (P-boxes). "
                "Popular block cipher designs include the Feistel Network (used in DES and Blowfish) and the "
                "Substitution-Permutation Network (SPN, utilized in AES)."
            ),
            "Agentic_RAG_Flows.txt": (
                "Agentic Retrieval-Augmented Generation (Agentic RAG) is an advanced RAG pattern where LLM agents control the "
                "retrieval and generation process. Standard RAG takes a query, fetches documents, and generates an answer in a "
                "single linear step. Adaptive and Agentic RAG architectures improve on this by introducing decision-making loops. "
                "First, a Query Analyzer decides if a query needs external knowledge retrieval. If not, the LLM answers directly. "
                "If retrieval is needed, a Query Rewriter refines the user's prompt to optimize it for vector search. "
                "After the Retriever fetches documents, a Context Evaluator checks if the retrieved information is relevant "
                "and sufficient. If it is not, the agent refines the query and retries retrieval (looping back to the retriever). "
                "Once sufficient context is gathered, the LLM Generator generates the final response. This loop guarantees higher "
                "relevance and reduces hallucinations."
            )
        }

        documents = []
        for filename, text in samples.items():
            chunks = self.split_text(text, chunk_size=300, chunk_overlap=30)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": filename, "chunk_id": i}
                )
                documents.append(doc)
        
        self.add_documents(documents)
        print("Default database initialized successfully.")
