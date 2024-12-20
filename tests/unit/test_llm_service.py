import pytest
from src.services.llm_service import LLMService

def test_llm_initialization():
    """Test LLM service initialization"""
    service = LLMService()
    assert service.model == "gpt-4o-mini"
    assert service.temperature == 0

def test_llm_invoke_with_system_prompt(llm_service):
    """Test LLM invocation with system prompt"""
    response = llm_service.invoke("What is AI?")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.parametrize("config", [
    {"temperature": 0.7},
    {"max_tokens": 100},
    {"temperature": 0, "max_tokens": 50},
])
def test_llm_invoke_with_config(llm_service, config):
    """Test LLM invocation with different configs"""
    response = llm_service.invoke("Test prompt", config)
    assert isinstance(response, str) 