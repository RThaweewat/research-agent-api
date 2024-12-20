"""
The `graph_service` module implements the core logic for processing
research questions. It manages the overall flow, including question
rewriting, document grading, and answer generation. It leverages
`src/services/llm_service.py` for LLM interactions and
`src/services/retrieval_service.py` for document retrieval,
ensuring a seamless question-answering pipeline.
"""

from typing import Dict, Any, List
from loguru import logger
from langchain.schema import HumanMessage, AIMessage
from src.services.llm_service import llm_service

class GraphService:
    def __init__(self, recursion_limit: int = 15):
        self.recursion_limit = recursion_limit
        logger.info(f"Initializing GraphService with recursion limit: {recursion_limit}")

    def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state through the graph workflow"""
        try:
            # 1. Rewrite/improve the question
            state = self.rewrite_question(state)
            
            # 2. Grade and filter documents
            state = self.grade_documents(state)
            
            # 3. Generate answer
            state = self.generate_answer(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Error in graph processing: {e}")
            raise

    def rewrite_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Improve the question for better document retrieval"""
        try:
            prompt = f"""Rewrite this research question to be more specific and focused: "{state['question']}"
            
            Rules:
            1. Keep it concise
            2. Use technical terminology
            3. Focus on specific aspects
            4. Maintain clarity
            
            Return ONLY the rewritten question, nothing else."""
            
            rewritten = llm_service.invoke(prompt, config={"temperature": 0.3})
            # Clean up the response - remove quotes and extra whitespace
            rewritten = rewritten.strip().strip('"').strip()
            state["rewritten_question"] = rewritten
            logger.info(f"Rewrote question: {rewritten}")
            
            return state
            
        except Exception as e:
            logger.error(f"Error rewriting question: {e}")
            state["rewritten_question"] = state["question"]
            return state

    def grade_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Grade documents for relevance to the question"""
        try:
            graded_docs = []
            question = state["rewritten_question"] or state["question"]
            
            for doc in state["documents"]:
                grade_prompt = f"""Rate the relevance of this document excerpt to the question: "{question}"

                Document content:
                {doc['content'][:1000]}...

                Rate from 0-1 where:
                0: Not relevant at all
                0.5: Somewhat relevant
                1: Highly relevant

                Return ONLY the numerical score (e.g., 0.7)."""
                
                response = llm_service.invoke(grade_prompt, config={"temperature": 0.1})
                
                try:
                    # Clean up and extract numerical score
                    grade = float(response.strip())
                except:
                    grade = doc['score']
                
                graded_docs.append({
                    "content": doc["content"],
                    "source": doc["source"],
                    "grade": grade,
                    "original_score": doc["score"]
                })
            
            # Sort by grade
            graded_docs.sort(key=lambda x: x["grade"], reverse=True)
            state["graded_docs"] = graded_docs
            
            return state
            
        except Exception as e:
            logger.error(f"Error grading documents: {e}")
            state["graded_docs"] = [{
                **doc,
                "grade": doc["score"]
            } for doc in state["documents"]]
            return state

    def generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final answer using graded documents"""
        try:
            # Use only highly relevant documents
            relevant_docs = [
                doc for doc in state["graded_docs"]
                if doc["grade"] > 0.5
            ]
            
            if not relevant_docs:
                relevant_docs = state["graded_docs"][:2]  # Use top 2 if none are highly relevant
            
            context = "\n\n".join([
                f"From {doc['source']} (relevance: {doc['grade']:.2f}):\n{doc['content']}"
                for doc in relevant_docs
            ])
            
            prompt = f"""Based on these research paper excerpts, answer this question: "{state['question']}"

            Context from papers:
            {context}

            Instructions:
            1. Only use information from the provided excerpts
            2. Cite specific papers when making claims
            3. If information is incomplete or unclear, acknowledge it
            4. Focus on key findings and conclusions
            5. Be concise but thorough
            6. Include clear references to papers

            Answer:"""
            
            answer = llm_service.invoke(
                prompt,
                config={
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )
            
            # Store both answer and relevant docs in state
            state["answer"] = answer
            state["relevant_docs"] = relevant_docs
            return state
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

graph_service = GraphService()
