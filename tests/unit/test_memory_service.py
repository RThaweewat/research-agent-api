import pytest
from src.services.memory_service import MemoryService
from langchain.schema import HumanMessage, AIMessage

def test_add_and_get_messages(memory_service):
    """Test adding and retrieving messages"""
    thread_id = "test-thread"
    message1 = HumanMessage(content="Test question")
    message2 = AIMessage(content="Test answer")
    
    memory_service.add_message(thread_id, message1)
    memory_service.add_message(thread_id, message2)
    
    messages = memory_service.get_messages(thread_id)
    assert len(messages) == 2
    assert messages[0].content == "Test question"
    assert messages[1].content == "Test answer"

def test_get_last_k_messages(memory_service):
    """Test retrieving last K messages"""
    thread_id = "test-thread"
    for i in range(5):
        memory_service.add_message(
            thread_id,
            HumanMessage(content=f"Message {i}")
        )
    
    messages = memory_service.get_messages(thread_id, last_k=3)
    assert len(messages) == 3
    assert messages[-1].content == "Message 4"

def test_clear_thread(memory_service):
    """Test clearing a conversation thread"""
    thread_id = "test-thread"
    memory_service.add_message(
        thread_id,
        HumanMessage(content="Test")
    )
    
    memory_service.clear_thread(thread_id)
    assert not memory_service.thread_exists(thread_id) 