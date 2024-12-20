import os
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.services.graph_service import GraphService
from src.services.llm_service import LLMService
from src.services.memory_service import MemoryService
from src.services.retrieval_service import RetrievalPipeline
from langchain.schema import Document

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment"""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    sample_pdf_path = os.path.join(fixtures_dir, "sample.pdf")
    if not os.path.exists(sample_pdf_path):
        with open(sample_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF")

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def sample_pdf_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")

@pytest.fixture
def mock_documents():
    return [
        Document(
            page_content="This is a test document about machine learning.",
            metadata={"source": "test1.pdf"}
        ),
        Document(
            page_content="This document discusses neural networks in detail.",
            metadata={"source": "test2.pdf"}
        )
    ]

@pytest.fixture
def graph_service():
    return GraphService()

@pytest.fixture
def llm_service():
    return LLMService()

@pytest.fixture
def memory_service():
    return MemoryService()

@pytest.fixture
def retrieval_pipeline():
    pipeline = RetrievalPipeline()
    yield pipeline
    pipeline.reset()  # Cleanup after each test 