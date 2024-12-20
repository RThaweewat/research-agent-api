"""
This module provides methods for interacting with Large Language
Models (LLMs). It manages OpenAI and TogetherAI integrations with
fallback logic to guarantee responses using `gpt-4o-mini` if
`Llama-3.3-70B` models fail. It includes system prompts to define
LLM behavior and incorporates error handling and tracing using `langfuse`.
"""

from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain.schema import SystemMessage
from src.utils.config import OPENAI_API_KEY, TOGETHER_API_KEY
from src.services.tracing_service import tracing_service
from loguru import logger
from typing import Optional, Dict

class LLMService:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0):
        self.primary_model = model
        self.backup_model = "gpt-4o-mini"
        self.temperature = temperature
        self._primary_llm = None
        self._backup_llm = None
        self.system_prompt = """You are a Research Paper Agent, build for research paper chatbot.
Always maintain a professional, academic tone while being helpful and clear."""
        
        self.setup_llm()
    
    def setup_llm(self):
        """Initialize both LLMs with error handling"""
        try:
            # Setup primary LLM (Together AI)
            self._primary_llm = ChatTogether(
                model=self.primary_model,
                temperature=self.temperature,
                api_key=TOGETHER_API_KEY,
                max_retries=2,
                callbacks=[tracing_service.get_handler()]
            )
            logger.info(f"Initialized primary LLM with model: {self.primary_model}")
        except Exception as e:
            logger.error(f"Error initializing primary LLM: {e}")
            self._primary_llm = None

        try:
            # Setup backup LLM (OpenAI)
            self._backup_llm = ChatOpenAI(
                model=self.backup_model,
                temperature=self.temperature,
                api_key=OPENAI_API_KEY,
                callbacks=[tracing_service.get_handler()]
            )
            logger.info(f"Initialized backup LLM with model: {self.backup_model}")
        except Exception as e:
            logger.error(f"Error initializing backup LLM: {e}")
            self._backup_llm = None
    
    @property
    def llm(self):
        """Get best available LLM instance with fallback logic"""
        if not self._primary_llm and not self._backup_llm:
            self.setup_llm()
        
        # Try primary LLM first
        if self._primary_llm:
            try:
                # Quick test to check if primary LLM is responsive
                self._primary_llm.invoke([{"role": "user", "content": "test"}])
                return self._primary_llm
            except Exception as e:
                logger.warning(f"Primary LLM failed, falling back to backup: {e}")
        
        # Fallback to backup LLM
        if self._backup_llm:
            return self._backup_llm
        
        raise RuntimeError("No LLM service available")
    
    def invoke(self, prompt: str, config: Optional[Dict] = None) -> str:
        """Invoke LLM with error handling and fallback"""
        try:
            logger.debug(f"Invoking LLM with prompt: {prompt[:100]}...")

            messages = [
                SystemMessage(content=self.system_prompt),
                {"role": "user", "content": prompt}
            ]
            
            # Merge configs
            invoke_config = {}
            if config:
                invoke_config.update(config)
            
            # Get best available LLM
            current_llm = self.llm
            
            response = current_llm.invoke(
                messages,
                config=invoke_config
            )
            
            response_text = str(response.content) if hasattr(response, 'content') else str(response)
            logger.debug(f"LLM response: {response_text[:100]}...")
            
            # Log which model was used
            model_used = (
                self.primary_model if current_llm == self._primary_llm 
                else self.backup_model
            )
            logger.info(f"Response generated using model: {model_used}")
            
            return response_text
            
        except Exception as e:
            logger.exception(f"Error invoking LLM: {e}")
            raise

llm_service = LLMService() 