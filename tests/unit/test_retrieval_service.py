import pytest
from src.services.retrieval_service import RetrievalPipeline

def test_pipeline_initialization(retrieval_pipeline):
    """Test retrieval pipeline initialization"""
    assert retrieval_pipeline.vectorstore is None
    assert retrieval_pipeline.documents == []

def test_pipeline_rebuild(retrieval_pipeline, mock_documents):
    """Test rebuilding pipeline with documents"""
    retrieval_pipeline.rebuild(mock_documents)
    assert retrieval_pipeline.has_documents()
    assert len(retrieval_pipeline.documents) == 2

def test_retrieve_with_no_documents(retrieval_pipeline):
    """Test retrieval with no documents loaded"""
    with pytest.raises(ValueError):
        retrieval_pipeline.retrieve("What is AI?")

def test_retrieve_with_documents(retrieval_pipeline, mock_documents):
    """Test document retrieval"""
    retrieval_pipeline.rebuild(mock_documents)
    results = retrieval_pipeline.retrieve("machine learning")
    assert len(results) > 0
    assert "machine learning" in results[0].lower()

@pytest.mark.parametrize("question,expected_count", [
    ("machine learning", 1),
    ("neural networks", 1),
    ("irrelevant query", 0),
])
def test_retrieve_different_queries(
    retrieval_pipeline,
    mock_documents,
    question,
    expected_count
):
    """Test retrieval with different queries"""
    retrieval_pipeline.rebuild(mock_documents)
    results = retrieval_pipeline.retrieve(question)
    assert len([r for r in results if question.lower() in r.lower()]) >= expected_count 