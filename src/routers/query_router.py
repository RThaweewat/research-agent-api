"""
The `query_router` defines endpoints for query processing, conversation
thread management, and resetting the vector database.
The `POST /api/query` endpoint dynamically routes user queries
to the memory service, vector store, or general LLM handler based on
 the characteristics of the question. It uses `src/services/memory_service.py`
  to maintain and manage conversation context. Additionally,
  it includes the `POST /api/thread/reset` endpoint for clearing conversation
  history and the `POST /api/vectordb/reset` endpoint to clear all indexed
  documents from the vector store.
"""

from fastapi import APIRouter, Query, HTTPException, Depends
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from src.services.memory_service import memory_service
from src.services.llm_service import llm_service
from src.services.graph_service import graph_service
from src.services.retrieval_service import RetrievalPipeline, retrieval_pipeline
import uuid
from loguru import logger
from langchain.schema import HumanMessage, AIMessage
from enum import Enum
from src.services.tracing_service import tracing_service
from src.models.request_models import QueryRequest
from src.models.response_models import QueryResponse, ErrorResponse, DocumentReference

router = APIRouter(prefix="/api")

class QueryType(str, Enum):
    MEMORY = "memory"
    VECTORSTORE = "vectorstore"
    GENERAL = "general"

class RouteQuery(BaseModel):
    """Route a user query to the most appropriate processing method."""
    query_type: Literal["memory", "vectorstore", "general"] = Field(
        ...,
        description="Type of query processing needed"
    )
    reason: str = Field(
        ...,
        description="Explanation for routing decision"
    )

class MemoryResponse(BaseModel):
    """Structure the response for memory-related queries."""
    response_type: Literal["count", "list", "history"] = Field(
        ...,
        description="Type of memory response"
    )
    content: str = Field(
        ...,
        description="The formatted response content"
    )

# Setup LLM and Chains
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    callbacks=[tracing_service.get_handler()]
)
structured_router = llm.with_structured_output(RouteQuery)
memory_chain = llm.with_structured_output(MemoryResponse)

# Router Prompt
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced AI assistant specialized in routing user questions to appropriate handlers within a complex information system.

Your goal is to route this question to one of the following handlers:
1. 'memory': For questions about conversation history, previous questions, or interaction counts.
2. 'vectorstore': For questions requiring research paper knowledge, especially those about:
   - Technical topics (LLMs, Deep Learning, NLP)
   - Research findings or conclusions
   - Specific papers or studies
   - State-of-the-art methods or models
   - Technical comparisons or benchmarks
3. 'general': For basic questions not requiring special context, including:
   - Questions about your capabilities
   - Basic factual questions
   - Non-technical queries
   - Questions about general concepts

IMPORTANT: Any question asking about research findings, papers, or technical details MUST go to 'vectorstore'."""),
    ("human", "{question}")
])

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at analyzing conversation history questions.
    Determine the appropriate response type:
    - 'count' for questions about number of interactions
    - 'list' for requests to see previous questions
    - 'history' for general conversation history requests
    
    Format the response appropriately based on the type."""),
    ("human", """Question: {question}
    Conversation History: {history}
    
    Determine response type and format appropriate response.""")
])

router_chain = router_prompt | structured_router
memory_chain = memory_prompt | memory_chain

def handle_memory_question(question: str, thread_id: str, tags: Optional[List[str]] = None) -> dict:
    """Handle questions about conversation history"""
    messages = memory_service.get_messages(thread_id)
    history = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" 
                        for m in messages])

    memory_result = memory_chain.invoke(
        {
            "question": question,
            "history": history
        },
        config={
            "callbacks": [tracing_service.get_handler()],
            "tags": tags or ["memory", "history", "conversation"]
        }
    )

    memory_service.add_message(thread_id, HumanMessage(content=question))
    memory_service.add_message(thread_id, AIMessage(content=memory_result.content))
    
    return {
        "response": memory_result.content,
        "thread_id": thread_id,
        "source": "memory",
        "response_type": memory_result.response_type
    }

def handle_general_question(question: str, thread_id: str, tags: Optional[List[str]] = None) -> dict:
    """Handle general questions using LLM"""
    response = llm_service.invoke(
        question,
        config={
            "callbacks": [tracing_service.get_handler()],
            "tags": tags or ["general", "llm", "direct"]
        }
    )
    memory_service.add_message(thread_id, HumanMessage(content=question))
    memory_service.add_message(thread_id, AIMessage(content=response))
    
    return {
        "response": response,
        "thread_id": thread_id,
        "source": "general"
    }

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query with document references"""
    try:
        thread_id = request.thread_id or str(uuid.uuid4())
        
        # Route the question
        route = router_chain.invoke(
            {"question": request.question},
            config={"callbacks": [tracing_service.get_handler()]}
        )
        
        logger.info(f"Question routed to: {route.query_type}")
        
        if route.query_type == "memory":
            result = handle_memory_question(request.question, thread_id)
            return QueryResponse(
                answer=result["response"],
                references=[],
                thread_id=thread_id
            )
            
        elif route.query_type == "general":
            result = handle_general_question(request.question, thread_id)
            return QueryResponse(
                answer=result["response"],
                references=[],
                thread_id=thread_id
            )
            
        elif route.query_type == "vectorstore":
            if not retrieval_pipeline.has_documents():
                return QueryResponse(
                    answer="I cannot answer research questions as no documents have been loaded. Please upload some research papers first.",
                    references=[],
                    thread_id=thread_id
                )
                
            try:
                # Initialize graph state
                state = {
                    "question": request.question,
                    "documents": [],
                    "rewritten_question": None,
                    "graded_docs": None,
                    "answer": None,
                    "thread_id": thread_id
                }
                
                # Get initial documents
                retrieved_docs = retrieval_pipeline.retrieve(request.question)
                if not retrieved_docs:
                    return QueryResponse(
                        answer="I couldn't find any relevant information in the documents.",
                        references=[],
                        thread_id=thread_id
                    )
                
                # Add retrieved docs to state
                state["documents"] = [
                    {
                        "content": doc["content"],
                        "source": doc["source"],
                        "score": doc["score"]
                    } for doc in retrieved_docs
                ]
                
                # Process through graph service
                try:
                    graph_result = graph_service.process_state(state)
                    
                    # Format references from graded documents
                    references = []
                    if graph_result.get("relevant_docs"):
                        references = [
                            DocumentReference(
                                source=doc["source"].split('/')[-1],
                                relevance_score=doc["grade"],
                                snippet=doc["content"][:500]
                            )
                            for doc in graph_result["relevant_docs"]
                        ]
                    
                    # Get final answer
                    answer = graph_result.get("answer", "Could not generate an answer from the documents.")
                    
                    # Add to conversation history
                    memory_service.add_message(thread_id, HumanMessage(content=request.question))
                    memory_service.add_message(thread_id, AIMessage(content=answer))
                    
                    return QueryResponse(
                        answer=answer,
                        references=sorted(references, key=lambda x: x.relevance_score, reverse=True),
                        thread_id=thread_id
                    )
                    
                except Exception as graph_error:
                    logger.error(f"Graph processing error: {graph_error}")
                    # Fallback to simpler processing
                    return handle_research_fallback(request.question, retrieved_docs, thread_id)
                
            except Exception as e:
                logger.error(f"Error processing research question: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def handle_research_fallback(question: str, docs: List[dict], thread_id: str) -> QueryResponse:
    """Fallback handler for research questions when graph processing fails"""
    try:
        context = "\n\n".join([
            f"From {doc['source']}:\n{doc['content']}"
            for doc in docs
        ])
        
        focused_question = f"""Based ONLY on the provided research papers, answer the following question: {question}
        Please provide a clear and concise answer, focusing on key findings and conclusions."""
        
        answer = llm_service.invoke(
            f"Context from papers:\n{context}\n\nQuestion: {focused_question}",
            config={"temperature": 0.2}
        )
        
        references = [
            DocumentReference(
                source=doc['source'].split('/')[-1],
                relevance_score=doc['score'],
                snippet=doc['content'][:500]
            )
            for doc in docs
        ]
        
        memory_service.add_message(thread_id, HumanMessage(content=question))
        memory_service.add_message(thread_id, AIMessage(content=answer))
        
        return QueryResponse(
            answer=answer,
            references=references,
            thread_id=thread_id
        )
        
    except Exception as e:
        logger.error(f"Error in research fallback: {e}")
        raise

@router.post("/thread/reset")
async def reset_thread(thread_id: str = Query(..., description="Thread ID to reset")):
    """Reset a conversation thread"""
    try:
        if not memory_service.thread_exists(thread_id):
            raise HTTPException(
                status_code=404, 
                detail=f"Thread {thread_id} not found"
            )
            
        memory_service.clear_thread(thread_id)
        logger.info(f"Thread {thread_id} reset successfully")
        return {
            "status": "success",
            "message": "Thread reset successfully",
            "thread_id": thread_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in reset_thread")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vectordb/reset")
async def reset_vectordb():
    """Reset the vector database"""
    try:
        # Reset the existing pipeline instead of creating new one
        retrieval_pipeline.reset()
        
        logger.info("Vector database reset successfully")
        return {
            "status": "success",
            "message": "Vector database reset successfully",
            "document_count": 0,
            "has_documents": False
        }
    except Exception as e:
        logger.exception("Error in reset_vectordb")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/thread/status/{thread_id}")
async def get_thread_status(thread_id: str):
    """Get the status of a conversation thread"""
    try:
        messages = memory_service.get_messages(thread_id)
        return {
            "thread_id": thread_id,
            "message_count": len(messages),
            "has_history": len(messages) > 0,
            "messages": [
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content
                }
                for msg in messages
            ]
        }
    except Exception as e:
        logger.exception("Error getting thread status")
        raise HTTPException(status_code=500, detail=str(e))
