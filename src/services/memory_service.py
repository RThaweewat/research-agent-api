"""
The `memory_service` module is responsible for managing conversation history.
It allows messages to be added to a thread, retrieves conversation messages
when needed, and provides functionality to clear conversation threads
entirely.
"""

from typing import Dict, List
from langchain_core.messages import BaseMessage
from loguru import logger

class MemoryService:
    def __init__(self):
        self.conversations: Dict[str, List[BaseMessage]] = {}
        
    def add_message(self, thread_id: str, message: BaseMessage):
        """Add a message to a conversation thread"""
        if thread_id not in self.conversations:
            self.conversations[thread_id] = []
        self.conversations[thread_id].append(message)
        logger.debug(f"Added message to thread {thread_id}. Total messages: {len(self.conversations[thread_id])}")
        
    def get_messages(self, thread_id: str, last_k: int = None) -> List[BaseMessage]:
        """Get messages from a conversation thread"""
        messages = self.conversations.get(thread_id, [])
        logger.debug(f"Retrieved {len(messages)} messages from thread {thread_id}")
        if last_k:
            messages = messages[-last_k:]
        return messages
        
    def clear_thread(self, thread_id: str):
        """Clear a conversation thread"""
        if thread_id in self.conversations:
            del self.conversations[thread_id]
            logger.info(f"Cleared thread {thread_id}")
            
    def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists"""
        return thread_id in self.conversations

memory_service = MemoryService()
