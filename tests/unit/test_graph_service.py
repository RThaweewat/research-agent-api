import pytest
from src.services.graph_service import GraphService
from langchain.schema import HumanMessage, AIMessage

def test_process_question_no_documents(graph_service):
    """Test processing question when no documents are loaded"""
    response = graph_service.process_question("What is AI?")
    assert "don't have any research papers loaded" in response.lower()

def test_process_question_with_history(graph_service, mock_documents):
    """Test processing question with conversation history"""
    # First rebuild the retrieval pipeline
    from src.services.retrieval_service import retrieval_pipeline
    retrieval_pipeline.rebuild(mock_documents)
    
    history = [
        HumanMessage(content="What is machine learning?"),
        AIMessage(content="Machine learning is a subset of AI.")
    ]
    response = graph_service._process_with_memory(
        "Can you elaborate on that?",
        history
    )
    assert response is not None
    assert isinstance(response, str)

@pytest.mark.parametrize("recursion_count,expected_contains", [
    (5, "steps"),
    (10, "steps"),
    (15, "steps"),
])
def test_format_error_response(graph_service, recursion_count, expected_contains):
    """Test error response formatting"""
    response = graph_service.format_error_response("Test error", recursion_count)
    assert expected_contains in response.lower()