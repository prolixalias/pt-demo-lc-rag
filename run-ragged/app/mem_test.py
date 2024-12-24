import asyncio
from datetime import datetime
from conversation_memory import ConversationMemory

async def test_conversation_memory():
    """Test basic conversation memory functionality"""
    # Initialize memory without persistence
    memory = ConversationMemory(max_turns=3)
    
    # Test 1: Basic addition and retrieval
    print("\nTest 1: Basic Addition and Retrieval")
    memory.add_interaction(
        query="What is the weather?",
        response="It's sunny today.",
        metadata={"confidence": 0.9}
    )
    
    history = memory.get_history()
    print(f"History after one interaction:\n{history}\n")
    
    # Test 2: Multiple turns and rolling window
    print("\nTest 2: Rolling Window")
    test_conversations = [
        ("How are you?", "I'm doing well!"),
        ("What time is it?", "It's noon."),
        ("What's the capital of France?", "Paris."),
        ("This should push out the first message", "Indeed it will.")
    ]
    
    for query, response in test_conversations:
        memory.add_interaction(query, response)
        
    print(f"Full history (should only show last 3):\n{memory.get_history()}\n")
    
    # Test 3: Different format outputs
    print("\nTest 3: Structured Output")
    structured = memory.get_history(format="structured", include_metadata=True)
    print(f"Structured format:\n{structured}\n")
    
    # Test 4: Clear functionality
    print("\nTest 4: Clear Functionality")
    memory.clear()
    print(f"History after clear:\n{memory.get_history()}\n")

if __name__ == "__main__":
    asyncio.run(test_conversation_memory())
