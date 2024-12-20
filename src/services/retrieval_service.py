"""
This module manages the document retrieval pipeline.
It sets up and indexes documents using both BM25 keyword-based retrieval
 and FAISS for semantic search. The `rebuild` method allows the indexes
 to be updated with new documents. The module also includes functionality
 for extracting relevant document snippets to improve the relevance of
 retrieved content.
"""
import os
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever

from src.utils.config import OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]  # Try to split on paragraph breaks first
)

class RetrievalPipeline:
    def __init__(self):
        self.vectorstore = None
        self.keyword_retriever = None
        self.ensemble_retriever = None
        self.documents = []
        self.embeddings = None
        
    def reset(self):
        """Completely reset the pipeline"""
        self.vectorstore = None
        self.keyword_retriever = None
        self.ensemble_retriever = None
        self.documents = []
        self.embeddings = None
        logger.info("Pipeline completely reset")
        
    def rebuild(self, docs):
        """Rebuild the pipeline with new documents"""
        self.reset()

        logger.info(f"Rebuilding retrieval pipeline with {len(docs)} documents")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i}: {doc.metadata.get('source', 'No source')} - First 100 chars: {doc.page_content[:100]}...")

        doc_splits = text_splitter.split_documents(docs)
        logger.info(f"Split into {len(doc_splits)} chunks")

        # Build FAISS vectorstore
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=self.embeddings
        )

        self.keyword_retriever = BM25Retriever.from_documents(doc_splits, similarity_top_k=6)
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",  
            search_kwargs={"k": 6},
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, self.keyword_retriever], 
            weights=[0.7, 0.3]  
        )
        self.documents = docs

    def has_documents(self):
        """Check if documents are loaded and retrievable"""
        return (
            len(self.documents) > 0 and 
            self.vectorstore is not None and 
            self.ensemble_retriever is not None
        )

    def retrieve(self, question: str) -> List[str]:
        """Enhanced retrieve method with logging"""
        if not self.has_documents():
            logger.warning("No documents loaded in retrieval pipeline")
            raise ValueError("No documents are currently loaded. Please upload documents first.")
        
        logger.info(f"Retrieving documents for question: {question}")
        docs = self.ensemble_retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} documents")
        
        # Extract relevant snippets and their sources
        relevant_content = []
        for doc in docs:
            score = 1.0
            logger.info(f"Processing document with score: {score}")

            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown source')
            logger.info(f"Document content preview: {content[:100]}...")

            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            question_words = set(question.lower().split())
            relevant_parts = []
            
            for paragraph in paragraphs:
                if (len(question_words & set(paragraph.lower().split())) > 1 or
                    any(term in paragraph.lower() for term in [
                        'study', 'research', 'method', 'result', 'conclusion',
                        'analysis', 'finding', 'data', 'experiment'
                    ])):
                    relevant_parts.append(paragraph)
            
            if relevant_parts:
                # Join with newlines to maintain formatting
                snippet = '\n\n'.join(relevant_parts)
                relevant_content.append({
                    'content': snippet,
                    'source': source,
                    'score': score
                })
                logger.info(f"Added relevant snippet: {snippet[:100]}...")
        
        if not relevant_content:
            if docs:
                first_doc = docs[0].page_content
                relevant_content = [{
                    'content': first_doc[:1000],
                    'source': docs[0].metadata.get('source', 'Unknown source'),
                    'score': 1.0
                }]
                logger.info("No specific snippets found, using document introduction")
            else:
                logger.warning("No relevant snippets or documents found")
        
        return relevant_content

    def query(self, question: str) -> List[str]:
        """Alias for retrieve method to maintain compatibility"""
        return self.retrieve(question)

    def save_vectorstore(self, path: str):
        """Save the FAISS vectorstore to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str):
        """Load the FAISS vectorstore from disk"""
        if os.path.exists(path):
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            self.vectorstore = FAISS.load_local(path, embeddings)
            vector_retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 2},
            )
            if self.documents:  # Only recreate ensemble if we have documents
                keyword_retriever = BM25Retriever.from_documents(
                    text_splitter.split_documents(self.documents), 
                    similarity_top_k=2
                )
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, keyword_retriever],
                    weights=[0.2, 0.8]
                )


retrieval_pipeline = RetrievalPipeline()
