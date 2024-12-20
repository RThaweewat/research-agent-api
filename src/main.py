"""
The `src/main.py` file serves as the entry point for the FastAPI application.
It initializes the app, setting up essential components such as CORS and routing.
On startup, the application preloads PDF documents from the `docs` directory
to make them immediately accessible. Additionally, it exposes a root path `/` that
can be used for system health checks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import query_router, docs_router
from src.utils.config import DOCS_FOLDER
from src.services.retrieval_service import retrieval_pipeline
from src.utils.pdf_utils import load_pdfs_from_directory
import uvicorn
from loguru import logger

# Create FastAPI app
app = FastAPI(title="Research Paper Agent API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(query_router.router)
app.include_router(docs_router.router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Research Paper Agent API")
    
    # Pre-load documents if any exist
    try:
        initial_docs = load_pdfs_from_directory(DOCS_FOLDER)
        if initial_docs:
            retrieval_pipeline.rebuild(initial_docs)
            logger.info(f"Loaded {len(initial_docs)} documents on startup")
    except Exception as e:
        logger.warning(f"Error loading initial documents: {e}")

@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {
        "status": "running",
        "message": "Research Paper Agent API is running",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }

def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
