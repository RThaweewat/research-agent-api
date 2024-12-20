"""
This module handles document upload and status retrieval functionality.
It includes the `POST /api/docs/upload` endpoint, which allows users to
 upload PDF documents. Uploaded PDFs are processed and text is extracted
 using utilities from `src/utils/pdf_utils.py`. The router also provides
 the `GET /api/status` endpoint for checking the status of loaded documents.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.services.retrieval_service import retrieval_pipeline
from src.utils.pdf_utils import process_uploaded_pdf
from loguru import logger

router = APIRouter(prefix="/api")

@router.post("/docs/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload multiple PDF documents to be processed and added to the knowledge base"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        results = {
            "successful": [],
            "failed": []
        }
        
        all_docs = []

        for file in files:
            try:
                if not file.filename.endswith('.pdf'):
                    results["failed"].append({
                        "filename": file.filename,
                        "error": "Only PDF files are supported"
                    })
                    continue

                docs = await process_uploaded_pdf(file)
                if docs:
                    all_docs.extend(docs)
                    results["successful"].append({
                        "filename": file.filename,
                        "pages": len(docs)
                    })
                else:
                    results["failed"].append({
                        "filename": file.filename,
                        "error": "No content could be extracted"
                    })
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                results["failed"].append({
                    "filename": file.filename,
                    "error": str(e)
                })

        if all_docs:
            retrieval_pipeline.rebuild(all_docs)
            logger.info(f"Rebuilt retrieval pipeline with {len(all_docs)} documents")

        response = {
            "message": f"Processed {len(results['successful'])} files successfully, {len(results['failed'])} failed",
            "details": results,
            "file_count": len(all_docs)
        }
        
        # If all files failed, return 400
        if not results["successful"] and results["failed"]:
            raise HTTPException(status_code=400, detail=response)
            
        return response
        
    except Exception as e:
        logger.exception(f"Error in upload_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_document_status():
    """Get the status of loaded documents"""
    try:
        has_docs = retrieval_pipeline.has_documents()
        return {
            "has_documents": has_docs,
            "document_count": len(retrieval_pipeline.documents) if has_docs else 0
        }
    except Exception as e:
        logger.exception(f"Error getting document status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
