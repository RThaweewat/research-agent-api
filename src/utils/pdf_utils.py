"""
The `pdf_utils` module provides utility functions for loading
and processing PDF files. It includes functions like
`load_pdfs_from_directory` to load multiple PDFs from a specified directory
and `load_pdf` to load a single PDF file. Additionally,
it contains asynchronous functions for processing uploaded PDF files.
"""

import os
from typing import List
from fastapi import UploadFile
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from src.utils.config import DOCS_FOLDER

async def process_uploaded_pdf(file: UploadFile) -> List[Document]:
    """
    Process an uploaded PDF file and return the extracted documents
    
    Args:
        file (UploadFile): The uploaded PDF file
        
    Returns:
        List[Document]: List of extracted documents from the PDF
    """
    # Create the docs directory if it doesn't exist
    os.makedirs(DOCS_FOLDER, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(DOCS_FOLDER, file.filename)
    content = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    try:
        # Load and process the PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Concatenate pages belonging to the same document
        if documents:
            combined_doc = documents[0]
            for doc in documents[1:]:
                combined_doc.page_content += "\n\n" + doc.page_content
            return [combined_doc]
        
        return []
        
    except Exception as e:
        # Clean up the file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise Exception(f"Error processing PDF: {str(e)}")

def load_pdfs_from_directory(directory: str) -> List[Document]:
    """
    Load all PDFs from a directory
    
    Args:
        directory (str): Path to directory containing PDFs
        
    Returns:
        List[Document]: List of documents from all PDFs
    """
    if not os.path.exists(directory):
        return []
        
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Combine pages of the same document
                if docs:
                    combined_doc = docs[0]
                    for doc in docs[1:]:
                        combined_doc.page_content += "\n\n" + doc.page_content
                    documents.append(combined_doc)
                    
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue
                
    return documents

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()
