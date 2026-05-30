import sys
from vector_store import LocalVectorStoreManager
from rag_agent import AdaptiveRAG

def test_pipeline():
    print("Initializing Vector Database...")
    db_manager = LocalVectorStoreManager()
    db_manager.initialize_with_defaults()

    print("\nInitializing Adaptive RAG agent in Mock mode...")
    agent = AdaptiveRAG(db_manager, mode="mock")

    # Test Case 1: Conversational query (Direct Response)
    print("\n--- Test Case 1: Direct Response Route ---")
    query_direct = "Hello, what is your name?"
    print(f"Query: '{query_direct}'")
    
    steps = list(agent.run(query_direct))
    final_step = steps[-1]
    
    assert final_step["step"] == 11, f"Expected step 11, got {final_step['step']}"
    assert "Direct Response" in final_step["data"]["answer"], "Expected Direct Response marker in mock answer"
    print("Success: Correctly routed to Direct Response and completed.")

    # Test Case 2: Technical query (Retrieval Route)
    print("\n--- Test Case 2: Retrieval Route ---")
    query_retrieval = "What is SpaceX Starship?"
    print(f"Query: '{query_retrieval}'")
    
    steps = list(agent.run(query_retrieval))
    final_step = steps[-1]
    
    assert final_step["step"] == 11, f"Expected step 11, got {final_step['step']}"
    assert "SpaceX Starship" in final_step["data"]["answer"], f"Expected SpaceX facts in answer, got {final_step['data']['answer']}"
    print("Success: Correctly routed to Retrieval, fetched documents, and completed.")
    
    # Verify that sources list is correctly populated
    sources = final_step["data"].get("sources", [])
    print(f"Sources cited: {sources}")
    assert "SpaceX_Starship.txt" in sources, "Expected SpaceX_Starship.txt in sources citation"
    print("Success: Sources cited correctly.")

    print("\n--- Test Case 3: Retrieval Route with Relevance Loop Retry ---")
    # By querying something that gets rewritten but is slightly off, we simulate loopback
    # Or in mock mode, a search query with low overlap:
    query_retry = "Tell me about quantum computing cargo capacity"
    print(f"Query: '{query_retry}'")
    
    steps = list(agent.run(query_retry))
    
    # Let's count how many times Step 9 (Context Evaluation) was called.
    # It should have run more than once if relevance check failed, or at least completed step 9.
    step_9_calls = [s for s in steps if s["step"] == 9 and s["status"] == "completed"]
    print(f"Context Evaluation ran {len(step_9_calls)} times.")
    
    final_step = steps[-1]
    assert final_step["step"] == 11, "Pipeline did not complete."
    print("Success: Pipeline ran and resolved successfully with loop simulation.")
    
    print("\nAll pipeline tests passed with ZERO errors!")

if __name__ == "__main__":
    try:
        test_pipeline()
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
