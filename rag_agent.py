import json
import re
import time
from typing import Dict, Any, List, Generator, Tuple
from google import genai
from openai import OpenAI
from langchain_core.documents import Document
from vector_store import LocalVectorStoreManager

class LLMClient:
    """Helper client to handle Gemini, OpenAI, or Mock LLM generations."""
    def __init__(self, mode: str = "mock", api_key: str = None):
        self.mode = mode.lower()
        self.api_key = api_key
        self.gemini_client = None
        self.openai_client = None
        
        if self.mode == "gemini" and api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        elif self.mode == "openai" and api_key:
            self.openai_client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        if self.mode == "gemini" and self.gemini_client:
            try:
                # Use standard gemini-2.5-flash for general generation
                contents = prompt
                if system_instruction:
                    contents = f"{system_instruction}\n\nUser request:\n{prompt}"
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                )
                return response.text
            except Exception as e:
                # Catch quota errors, invalid keys, etc. and fallback gracefully to mock
                print(f"[API Warning] Gemini API Error: {str(e)}. Gracefully falling back to Local Simulation Mode.")
                return self._mock_generate(prompt, system_instruction)
                
        elif self.mode == "openai" and self.openai_client:
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                # Catch quota errors, invalid keys, etc. and fallback gracefully to mock
                print(f"[API Warning] OpenAI API Error: {str(e)}. Gracefully falling back to Local Simulation Mode.")
                return self._mock_generate(prompt, system_instruction)
                
        else:
            # Mock mode logic
            return self._mock_generate(prompt, system_instruction)

    def _mock_generate(self, prompt: str, system_instruction: str = None) -> str:
        """Heuristics-based local response generator for zero-API mock mode."""
        sys_inst = (system_instruction or "").lower()
        prompt_lower = prompt.lower()
        
        # 1. Intent Classification
        if "intent classifier" in sys_inst:
            # Extract user query from prompt (first line starting with 'Query:')
            match = re.search(r'Query:\s*(.*?)\n', prompt, re.IGNORECASE)
            user_query = match.group(1).lower() if match else prompt_lower
            
            # Simple keyword matching to decide routing
            needs_retrieval = any(w in user_query for w in [
                "starship", "spacex", "booster", "thrust", "payload", "capacity", "orbit", "raptor", "engine",
                "alphafold", "protein", "dna", "rna", "deepmind", "isomorphic",
                "quantum", "qubit", "superposition", "entanglement", "decoherence", "refrigerator",
                "agentic", "rag", "retrieval", "adaptive", "evaluator", "context",
                "cipher", "cryptography", "encryption", "decryption", "security", "block"
            ])
            explanation = (
                "Detected domain-specific keywords requiring technical document retrieval." 
                if needs_retrieval 
                else "Conversational or general query. Routed to direct LLM response."
            )
            return json.dumps({"need_retrieval": needs_retrieval, "explanation": explanation})
            
        # 2. Query Rewriter
        if "search query optimizer" in sys_inst:
            match = re.search(r'Query:\s*(.*?)\n', prompt, re.IGNORECASE)
            query = match.group(1) if match else prompt
            rewritten = query.lower()
            fillers = ["please", "tell me about", "what is", "do you know", "can you write", "search for", "find me", "information on"]
            for f in fillers:
                rewritten = rewritten.replace(f, "")
            rewritten = rewritten.strip()
            if not rewritten:
                rewritten = query
            return json.dumps({"rewritten_query": rewritten})
            
        # 3. Context Evaluator / Relevance Check
        if "relevance auditor" in sys_inst:
            is_relevant = True
            score = 0.8
            query_match = re.search(r'Query:\s*(.*?)\n', prompt, re.IGNORECASE)
            context_match = re.search(r'Context:\s*(.*)', prompt, re.DOTALL | re.IGNORECASE)
            
            if query_match and context_match:
                q = query_match.group(1).lower()
                c = context_match.group(1).lower()
                # Check keyword overlap (words > 3 chars)
                keywords = [w for w in re.split(r'\W+', q) if len(w) > 3]
                if keywords:
                    matches = sum(1 for w in keywords if w in c)
                    overlap_ratio = matches / len(keywords)
                    if overlap_ratio < 0.2:
                        is_relevant = False
                        score = overlap_ratio
                    else:
                        score = min(0.9, 0.4 + overlap_ratio * 0.5)
            
            return json.dumps({
                "relevant": is_relevant,
                "score": score,
                "explanation": f"Document keyword overlap ratio is {score:.2f}."
            })
            
        # 4. Generator (or general context-based answering)
        if "helpful qa system" in sys_inst or "context:" in prompt_lower:
            context_section = re.search(r'Context:\s*(.*?)\n\nQuery:', prompt, re.DOTALL | re.IGNORECASE)
            query_section = re.search(r'Query:\s*(.*)', prompt, re.IGNORECASE)
            
            context = context_section.group(1) if context_section else ""
            query = query_section.group(1) if query_section else prompt
            
            intro = f"[Simulated Response - Local Mode] Based on the retrieved context:\n\n"
            summary_sentences = []
            
            # Simple summarization by finding sentences with matching words
            query_words = [w.lower() for w in re.split(r'\W+', query) if len(w) > 3]
            sentences = re.split(r'\. |\n', context)
            for s in sentences:
                s = s.strip()
                if not s: continue
                if any(qw in s.lower() for qw in query_words):
                    if s not in summary_sentences:
                        summary_sentences.append(s)
            
            if not summary_sentences:
                summary_sentences = [s for s in sentences[:3] if s.strip()]
                
            body = ". ".join(summary_sentences) + "."
            footer = "\n\n(Note: This answer was synthesized by extracting key facts from the vector database.)"
            return intro + body + footer

        # Default response for direct answering
        return (
            f"[Direct Response - Local Mode] Hello! I classified this as a direct response query "
            f"because it does not require external documents. Query: '{prompt}'. How else can I assist you today?"
        )

class AdaptiveRAG:
    def __init__(self, vector_store: LocalVectorStoreManager, mode: str = "mock", api_key: str = None):
        self.vector_store = vector_store
        self.llm = LLMClient(mode, api_key)
        self.relevance_threshold = 0.45
        self.max_retries = 2

    def run(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """Runs the Adaptive RAG pipeline and yields step-by-step trace dictionary logs."""
        yield {
            "step": 1,
            "name": "User Query",
            "status": "completed",
            "message": "Received user query.",
            "data": {"query": query}
        }
        
        # --- Step 2 & 3: Query Analyzer / Intent Classifier ---
        yield {
            "step": 2,
            "name": "Query Analyzer / Intent Classifier",
            "status": "running",
            "message": "Analyzing query intent to decide if retrieval is required...",
            "data": {}
        }
        
        analysis_prompt = (
            f"Query: {query}\n\n"
            "Classify the query into a JSON object with keys:\n"
            '1. "need_retrieval": boolean (true if the query requires external facts or technical database documents, false otherwise)\n'
            '2. "explanation": string (brief reasoning for classification)\n\n'
            "Strictly return only valid JSON."
        )
        
        system_instruction = (
            "You are an Intent Classifier. Your job is to determine if a user query requires retrieving information "
            "from a technical knowledge base (e.g. specifics about SpaceX Starship, AlphaFold, deep learning, quantum bits). "
            "Return JSON matching the schema."
        )
        
        try:
            analysis_raw = self.llm.generate(analysis_prompt, system_instruction)
            # Parse JSON
            analysis = self._parse_json(analysis_raw)
            need_retrieval = analysis.get("need_retrieval", True)
            explanation = analysis.get("explanation", "Classification processed.")
        except Exception as e:
            # Fallback
            need_retrieval = True
            explanation = f"Error during intent classification ({e}). Defaulting to retrieval."

        yield {
            "step": 3,
            "name": "Need Retrieval Decision",
            "status": "completed",
            "message": f"Intent Classifier result: {'Retrieval Path' if need_retrieval else 'Direct LLM Path'}.",
            "data": {"need_retrieval": need_retrieval, "explanation": explanation}
        }
        
        # --- Route: Direct LLM Response ---
        if not need_retrieval:
            yield {
                "step": 4,
                "name": "LLM Direct Response",
                "status": "running",
                "message": "Generating direct response using LLM parametric knowledge...",
                "data": {}
            }
            
            direct_prompt = f"Please answer the user's conversational query: {query}"
            direct_response = self.llm.generate(direct_prompt)
            
            yield {
                "step": 4,
                "name": "LLM Direct Response",
                "status": "completed",
                "message": "Direct response generated.",
                "data": {"response": direct_response}
            }
            
            yield {
                "step": 11,
                "name": "Final Answer",
                "status": "completed",
                "message": "Pipeline finished.",
                "data": {"answer": direct_response}
            }
            return

        # --- Route: Retrieval Path ---
        current_query = query
        retries = 0
        all_retrievals = []
        
        while retries <= self.max_retries:
            # --- Step 5: Query Rewriting / Planning ---
            yield {
                "step": 5,
                "name": f"Query Rewriting / Planning (Attempt {retries + 1})",
                "status": "running",
                "message": f"Rewriting query to optimize vector search overlap...",
                "data": {"original_query": query, "attempt": retries + 1}
            }
            
            rewrite_prompt = (
                f"Query: {current_query}\n\n"
                "Rewrite this query into a concise search term focusing on nouns, synonyms, and facts. "
                "Output JSON: " + '{"rewritten_query": "concise search term"}'
            )
            
            rewrite_system = "You are a Search Query Optimizer. Clean up questions into search terms. Return JSON."
            
            try:
                rewrite_raw = self.llm.generate(rewrite_prompt, rewrite_system)
                rewrite_json = self._parse_json(rewrite_raw)
                search_query = rewrite_json.get("rewritten_query", current_query)
            except Exception as e:
                search_query = current_query
                
            yield {
                "step": 5,
                "name": f"Query Rewriting / Planning (Attempt {retries + 1})",
                "status": "completed",
                "message": f"Query optimized: '{search_query}'",
                "data": {"original_query": current_query, "rewritten_query": search_query}
            }
            
            # --- Step 6 & 7 & 8: Retriever & Vector DB ---
            yield {
                "step": 6,
                "name": f"Retriever (Attempt {retries + 1})",
                "status": "running",
                "message": f"Querying FAISS database with '{search_query}'...",
                "data": {}
            }
            
            retrieved_docs_with_scores = self.vector_store.similarity_search(search_query, k=3)
            
            # Formulate trace records
            doc_records = []
            for doc, score in retrieved_docs_with_scores:
                doc_records.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": float(score)
                })
                
            yield {
                "step": 8,
                "name": f"Retrieved Context (Attempt {retries + 1})",
                "status": "completed",
                "message": f"Retrieved {len(retrieved_docs_with_scores)} document chunks.",
                "data": {"documents": doc_records}
            }
            
            # --- Step 9: Context Evaluation / Relevance Check ---
            yield {
                "step": 9,
                "name": f"Context Evaluation (Attempt {retries + 1})",
                "status": "running",
                "message": "Evaluating relevance of retrieved context...",
                "data": {}
            }
            
            # Build evaluation prompt
            context_str = "\n---\n".join([f"Doc {i+1} (Source: {d['source']}): {d['content']}" for i, d in enumerate(doc_records)])
            
            eval_prompt = (
                f"Query: {search_query}\n\n"
                f"Context:\n{context_str}\n\n"
                "Evaluate if the context contains enough helpful facts to answer the query. "
                "Output JSON: " + '{"relevant": boolean, "score": float_between_0_and_1, "explanation": "brief reasoning"}'
            )
            
            eval_system = "You are a Retrieval Relevance Auditor. Grade context helpfulness. Return JSON."
            
            try:
                eval_raw = self.llm.generate(eval_prompt, eval_system)
                eval_json = self._parse_json(eval_raw)
                is_relevant = eval_json.get("relevant", True)
                eval_score = eval_json.get("score", 0.7)
                eval_explanation = eval_json.get("explanation", "Evaluation completed.")
            except Exception as e:
                is_relevant = True
                eval_score = 0.8
                eval_explanation = f"Error during evaluation: {e}. Defaulting to relevant."
            
            # Override is_relevant based on threshold if score is explicitly low
            if eval_score < self.relevance_threshold:
                is_relevant = False

            yield {
                "step": 9,
                "name": f"Context Evaluation (Attempt {retries + 1})",
                "status": "completed",
                "message": f"Evaluation Result: {'RELEVANT' if is_relevant else 'IRRELEVANT'} (Score: {eval_score:.2f})",
                "data": {
                    "relevant": is_relevant,
                    "score": float(eval_score),
                    "explanation": eval_explanation,
                    "retry_count": retries,
                    "max_retries": self.max_retries
                }
            }
            
            if is_relevant:
                # Found good context, proceed to generator!
                all_retrievals = doc_records
                break
            else:
                # Irrelevant context, loop back with retry
                retries += 1
                if retries <= self.max_retries:
                    # Modify current query for the next iteration to widen the search
                    current_query = f"{search_query} details information"
                    yield {
                        "step": 9,
                        "name": "Retry / Refine Retrieval Link",
                        "status": "completed",
                        "message": f"Triggering retry loop. Refining query to broaden results.",
                        "data": {"next_query": current_query}
                    }
                else:
                    # Out of retries, proceed with what we have
                    all_retrievals = doc_records
                    yield {
                        "step": 9,
                        "name": "Max Retries Reached",
                        "status": "completed",
                        "message": "Max retries reached. Proceeding to generator with best available context.",
                        "data": {}
                    }
                    break

        # --- Step 10: LLM Generator ---
        yield {
            "step": 10,
            "name": "LLM Generator",
            "status": "running",
            "message": "Synthesizing final answer from retrieved documents...",
            "data": {}
        }
        
        # Build generator prompt
        final_context_str = "\n\n".join([f"[Source: {d['source']}]\n{d['content']}" for d in all_retrievals])
        
        generator_prompt = (
            f"Context:\n{final_context_str}\n\n"
            f"Query: {query}\n\n"
            "Answer the query accurately based on the context provided. If the context does not contain the answer, "
            "state that clearly, but try your best to answer based on any relevant details. Keep it structured and professional."
        )
        
        generator_system = (
            "You are a helpful QA system. Answer the query using ONLY the provided context. "
            "Cite sources by mentioning their source filename (e.g. SpaceX_Starship.txt)."
        )
        
        try:
            final_answer = self.llm.generate(generator_prompt, generator_system)
        except Exception as e:
            final_answer = f"Error generating final response: {e}."
            
        yield {
            "step": 10,
            "name": "LLM Generator",
            "status": "completed",
            "message": "Final answer generated.",
            "data": {"answer": final_answer}
        }
        
        # --- Step 11: Final Answer ---
        yield {
            "step": 11,
            "name": "Final Answer",
            "status": "completed",
            "message": "Adaptive RAG pipeline completed successfully.",
            "data": {
                "answer": final_answer,
                "sources": list(set([d["source"] for d in all_retrievals]))
            }
        }

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Tries to find and parse JSON block in LLM response."""
        # Clean response
        clean_text = text.strip()
        # Find JSON boundaries
        if "```json" in clean_text:
            match = re.search(r'```json\s*(.*?)\s*```', clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1)
        elif "```" in clean_text:
            match = re.search(r'```\s*(.*?)\s*```', clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1)
                
        # Parse
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # Try regex to clean up trailing commas
            try:
                # Find the first { and last }
                start = clean_text.find('{')
                end = clean_text.rfind('}')
                if start != -1 and end != -1:
                    json_str = clean_text[start:end+1]
                    return json.loads(json_str)
            except:
                pass
            
            # Rule based fallback if parsing fails completely
            # Let's inspect the text for need_retrieval or rewritten_query keywords
            ret = {}
            if "need_retrieval" in clean_text.lower():
                ret["need_retrieval"] = "true" in clean_text.lower()
                ret["explanation"] = "Fallback parsed need_retrieval."
            if "rewritten_query" in clean_text.lower():
                # Extract value
                m = re.search(r'"rewritten_query"\s*:\s*"(.*?)"', clean_text)
                ret["rewritten_query"] = m.group(1) if m else "query"
            if "relevant" in clean_text.lower():
                ret["relevant"] = "true" in clean_text.lower()
                ret["score"] = 0.8 if ret["relevant"] else 0.2
                ret["explanation"] = "Fallback parsed relevance."
                
            return ret if ret else {"need_retrieval": True, "rewritten_query": text, "relevant": True, "score": 0.8}
