"""
The `tracing_service` module initializes the Langfuse client for
observability. It provides methods to trace interactions and log
events during API calls. Context-managed traces are implemented,
allowing for detailed monitoring of system interactions, including
quality assessments and scoring.
"""
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import langfuse_context, observe
from loguru import logger
from typing import Optional, Dict, Any, ContextManager, List
from contextlib import contextmanager
from src.utils.config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST
)

class TracingService:
    def __init__(self):
        try:
            self.langfuse = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST
            )
            self.handler = CallbackHandler()
            logger.info("Tracing service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize tracing service: {e}")
            raise

    @contextmanager
    def trace_interaction(self, 
                         interaction_type: str, 
                         metadata: Optional[Dict] = None,
                         tags: Optional[list] = None) -> ContextManager:
        """Start a new trace for an interaction"""
        try:
            # Create a new trace
            trace = self.langfuse.trace(
                name=f"{interaction_type}_interaction",
                metadata=metadata or {},
                tags=tags or []
            )
            
            # Create a span
            span = trace.span(name=interaction_type)
            if metadata:
                span.metadata = metadata
            if tags:
                span.tags = tags
            
            yield span
            
        except Exception as e:
            logger.error(f"Error in trace context: {e}")
            yield None

    def get_handler(self) -> CallbackHandler:
        """Get the Langfuse callback handler for LangChain"""
        return self.handler

    def add_score(self, 
                 name: str, 
                 value: float, 
                 comment: Optional[str] = None,
                 trace_id: Optional[str] = None) -> None:
        """Add a score to a trace"""
        try:
            if trace_id:
                self.langfuse.score(
                    trace_id=trace_id,
                    name=name,
                    value=value,
                    comment=comment
                )
            else:
                # Create a new trace for the score
                trace = self.langfuse.trace(
                    name=f"score_{name}",
                    metadata={"score_name": name}
                )
                span = trace.span(name="add_score")
                span.score(
                    name=name,
                    value=value,
                    comment=comment
                )
                    
            logger.debug(f"Added score: {name}={value}")
        except Exception as e:
            logger.error(f"Error adding score: {e}")

    def log_memory_access(self, thread_id: str, message_count: int, tags: Optional[List[str]] = None) -> None:
        """Log memory access"""
        try:
            trace = self.langfuse.trace(
                name="memory_access",
                metadata={
                    "thread_id": thread_id,
                    "message_count": message_count
                },
                tags=tags or ["memory", "history", "conversation"]
            )
            span = trace.span(name="memory_access")
            span.score(
                name="memory_access",
                value=1.0,
                comment=f"Retrieved {message_count} messages"
            )
                
        except Exception as e:
            logger.error(f"Error logging memory access: {e}")

    def flush(self):
        """Flush any pending traces"""
        try:
            self.langfuse.flush()
        except Exception as e:
            logger.error(f"Error flushing traces: {e}")

    def shutdown(self):
        """Shutdown the tracing service"""
        try:
            self.flush()
            self.langfuse.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down tracing service: {e}")

tracing_service = TracingService() 