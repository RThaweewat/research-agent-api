# 🤖 Research Agent API

## Overview
The Research Agent API is a robust tool for extracting and synthesizing information from research documents. Designed for researchers, academics, and professionals, 
the API simplifies access to research papers, integrating general knowledge with document-specific details to provide clear and efficient responses.

## Key Features
- **Smart Question Routing:** Automatically directs queries to the most suitable processing mechanism, including conversational memory for context-aware responses, vector-based retrieval for document-specific answers, and general LLM-based solutions for broader questions.
- **Hybrid Retrieval System:** Combines keyword search (BM25) with semantic search using FAISS and OpenAI Embeddings for comprehensive and accurate document discovery.
- **Enhanced Query Rewriting:** Optimizes search precision by rephrasing user queries with advanced techniques.
- **Layered Answer Generation:** Produces high-quality responses through a multi-stage process that includes self-assessment and quality assurance.
- **Flexible Retrieval Pipeline:** Allows real-time indexing of newly uploaded documents, seamlessly updating the retrieval process.
- **REST API Accessibility:** Provides easy integration with endpoints for document uploads, query handling, and managing conversational memory.
- **Advanced Monitoring:** Incorporates Langfuse for in-depth logging, debugging, and performance monitoring, ensuring detailed insights into system operations.

## Technical Stack

### Core Components

*   **Language Models**: 🧠
    *   Primary: Llama 3 70B Instruct via TogetherAI API. 🦙
    *   Backup: GPT-4o (via OpenAI API) for guaranteed response fallback in case of primary LLM issues. 🤖
*   **🗄️ Vector Database:** FAISS (Facebook AI Similarity Search) for creating semantic embeddings.

*   **Retrieval Techniques:** 🔎
    *   **🔑 Keyword Search:** BM25 (Best Matching 25) for keyword-based document retrieval.
    *   **✨ Semantic Search:** OpenAI Embeddings for vector-based semantic search.
    *   **🔀 Ensemble Search:** Combines BM25 and semantic search results for enhanced retrieval performance.
*   **🕹️ Orchestration:** FastAPI for API management and efficient handling of web requests.
*   **🚦 Tracing:** Langfuse for observability of application flow and debugging insights.
*   **📄 Document Handling:** PyPDF for processing PDF documents.

### Environment and Dependencies
*   **Runtime:** Python 3.12
*   **Containerization:** Docker (via `docker-compose`) for reproducible deployments.
*   **Key Libraries:**
    *   `langchain`: Framework for developing applications with language models.
    *   `langgraph`: Library for building robust conversational flows using graph architectures.
    *   `faiss-cpu`: Library for creating and searching vector indices efficiently.
    *   `fastapi`:  Modern, high-performance web framework for building APIs.
    *   `python-multipart`: For handling multipart/form-data file uploads.
    *   `langfuse`: Langfuse library for tracing LLM applications.

## API Endpoints

### 1. Upload Research Papers
*   **Endpoint:** `POST /api/docs/upload`
*   **Description:** Uploads multiple PDF documents to be added to the system's knowledge base.
*   **Request Body:** Accepts `multipart/form-data` with files under the key `files`.
*   **Response:** JSON object detailing each file's upload status (successful, failed, pages processed), along with the total number of files processed.

### 2. Query Documents
*   **Endpoint:** `POST /api/query`
*   **Description:** Handles user queries against loaded documents or general knowledge. Supports conversation continuity via `thread_id`.
*   **Request Body:**
    ```json
    {
        "question": "Your question here",
        "thread_id": "Optional thread ID for conversation continuity",
        "config": {
           //  Optional  custom config options  like temperature or any other config
        }
    }
    ```
*   **Response:** A JSON object containing the answer, a list of document references, and the thread ID.
```json
    {
        "answer": "The answer to your question",
        "references": [
            {
                "source": "paper1.pdf",
                "relevance_score": 0.95,
                "snippet": "A relevant excerpt from the paper"
            },
            ...
        ],
        "thread_id": "UUID of the conversation thread"
    }
```

### 3. Reset Thread
*   **Endpoint:** `POST /api/thread/reset`
*   **Description:** Clears the memory for a specific conversation thread.
*   **Request Params:** Accepts `thread_id` as a query parameter.
*   **Response:** Confirmation message and thread ID of the reset thread.
```json
    {
      "status": "success",
      "message": "Thread reset successfully",
      "thread_id": "your-thread-id"
    }
```

### 4. Reset Vector Database
*   **Endpoint:** `POST /api/vectordb/reset`
*   **Description:** Clears the FAISS vector database, removing all document indexes, and resetting the retrieval pipeline.
*   **Response:** Confirmation message of the database reset, along with document status.
```json
    {
      "status": "success",
      "message": "Vector database reset successfully",
      "document_count": 0,
      "has_documents": false
    }
```
### 5. Get Document Status
*   **Endpoint:** `GET /api/status`
*   **Description:** Gets the status of documents currently loaded in the vector database.
*   **Response:** JSON object containing a boolean to check if document available and the document count.
```json
{
  "has_documents": true,
  "document_count": 4
}
```
### 6. Get Thread Status
*   **Endpoint:** `GET /api/thread/status/{thread_id}`
*   **Description:** Returns the status and history of a specific thread, including all messages exchanged.
*   **Response:** JSON object showing the number of messages, a boolean to check is history available and a list of the messages exchanged.
```json
{
  "thread_id": "your-thread-id",
  "message_count": 3,
  "has_history": true,
  "messages": [
    {
      "type": "human",
      "content": "your question ?"
    },
    {
      "type": "ai",
      "content": "Your response to the question"
    },
        {
      "type": "human",
      "content": "second question ?"
    }
  ]
}
```

## Project Structure

```text
├── docker-compose.yml
├── postman_test_cases.json
├── requirements.txt
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_graph_service.py
│   │   ├── test_llm_service.py
│   │   ├── test_memory_service.py
│   │   └── test_retrieval_service.py
│   └── fixtures/
├── docs/
├── papers/    <--- PDF documents are stored here
├── scripts/
│   └── export_codebase.py
├── src/
│   ├── main.py
│   ├── routers/
│   │   ├── docs_router.py
│   │   └── query_router.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── pdf_utils.py
│   ├── models/
│   │   ├── request_models.py
│   │   └── response_models.py
│   ├── services/
│   │   ├── graph_service.py
│   │   ├── llm_service.py
│   │   ├── memory_service.py
│   │   ├── retrieval_service.py
│   │   └── tracing_service.py

```

## Detailed Component Explanation

<details>
<summary><b>src/main.py</b></summary>
The `src/main.py` file serves as the entry point for the FastAPI application. It initializes the app, setting up essential components such as CORS and routing. On startup, the application preloads PDF documents from the `docs` directory to make them immediately accessible. Additionally, it exposes a root path `/` that can be used for system health checks.
</details>

<details>
<summary><b>src/routers/docs_router.py</b></summary>
This module handles document upload and status retrieval functionality. It includes the `POST /api/docs/upload` endpoint, which allows users to upload PDF documents. Uploaded PDFs are processed and text is extracted using utilities from `src/utils/pdf_utils.py`. The router also provides the `GET /api/status` endpoint for checking the status of loaded documents.
</details>

<details>
<summary><b>src/routers/query_router.py</b></summary>
The `query_router` defines endpoints for query processing, conversation thread management, and resetting the vector database. The `POST /api/query` endpoint dynamically routes user queries to the memory service, vector store, or general LLM handler based on the characteristics of the question. It uses `src/services/memory_service.py` to maintain and manage conversation context. Additionally, it includes the `POST /api/thread/reset` endpoint for clearing conversation history and the `POST /api/vectordb/reset` endpoint to clear all indexed documents from the vector store.
</details>

<details>
<summary><b>src/utils/config.py</b></summary>
This module manages configuration settings for the application by loading environment variables from `.env` files. It defines key settings for OpenAI and Langfuse integrations, as well as paths for default directories used throughout the application.
</details>

<details>
<summary><b>src/utils/logging.py</b></summary>
The `logging` module configures the `loguru` library for logging. It sets up both console and file-based logging to ensure that system activity is appropriately recorded for debugging and monitoring purposes.
</details>

<details>
<summary><b>src/utils/pdf_utils.py</b></summary>
The `pdf_utils` module provides utility functions for loading and processing PDF files. It includes functions like `load_pdfs_from_directory` to load multiple PDFs from a specified directory and `load_pdf` to load a single PDF file. Additionally, it contains asynchronous functions for processing uploaded PDF files.
</details>

<details>
<summary><b>src/models/request_models.py</b></summary>
This module defines Pydantic models used for validating request payloads. It includes the `QueryRequest` model for queries, which incorporates fields for the question, user ID, and thread ID. Additionally, the `UploadRequest` model is used for specifying the file type during document uploads.
</details>

<details>
<summary><b>src/models/response_models.py</b></summary>
Response models for API interactions are defined in this module. It includes the `QueryResponse` model for returning answers to queries, as well as `UploadResponse` and `DocumentReference` models for document-related operations. An `ErrorResponse` model is also included to standardize error handling.
</details>

<details>
<summary><b>src/services/graph_service.py</b></summary>
The `graph_service` module implements the core logic for processing research questions. It manages the overall flow, including question rewriting, document grading, and answer generation. It leverages `src/services/llm_service.py` for LLM interactions and `src/services/retrieval_service.py` for document retrieval, ensuring a seamless question-answering pipeline.
</details>

<details>
<summary><b>src/services/llm_service.py</b></summary>
This module provides methods for interacting with Large Language Models (LLMs). It manages OpenAI and TogetherAI integrations with fallback logic to guarantee responses using `gpt-4o-mini` if `Llama-3.3-70B` models fail. It includes system prompts to define LLM behavior and incorporates error handling and tracing using `langfuse`.
</details>

<details>
<summary><b>src/services/memory_service.py</b></summary>
The `memory_service` module is responsible for managing conversation history. It allows messages to be added to a thread, retrieves conversation messages when needed, and provides functionality to clear conversation threads entirely.
</details>

<details>
<summary><b>src/services/retrieval_service.py</b></summary>
This module manages the document retrieval pipeline. It sets up and indexes documents using both BM25 keyword-based retrieval and FAISS for semantic search. The `rebuild` method allows the indexes to be updated with new documents. The module also includes functionality for extracting relevant document snippets to improve the relevance of retrieved content.
</details>

<details>
<summary><b>src/services/tracing_service.py</b></summary>
The `tracing_service` module initializes the Langfuse client for observability. It provides methods to trace interactions and log events during API calls. Context-managed traces are implemented, allowing for detailed monitoring of system interactions, including quality assessments and scoring.
</details>

## Installation

### 1. Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-repo/research-agent-api.git
cd research-agent-api
```

---

### 2. Set Up the Environment

Copy the environment variables file and add your API keys:

```bash
cp .env.example .env
# Add your OpenAI, Together, and Langfuse API keys in the .env file
```

---

### 3. Install Dependencies

#### Option 1: Using pip (for local development)

Install the required Python packages:

```bash
pip install -r requirements.txt
```

#### Option 2: Using Docker (recommended for consistent deployments)

- **Install Docker:** If not already installed, follow instructions for your OS at [Docker Installation Guide](https://docs.docker.com/get-docker/).
- **Install Docker Compose:** Usually included with Docker Desktop. For standalone installation, refer to [Docker Compose Installation Guide](https://docs.docker.com/compose/install/).

---

### 4. Run the API

#### Option 1: Without Docker

Run the application using Uvicorn:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 2: With Docker

1. Navigate to the project root where `docker-compose.yml` is located.
2. Build and run the Docker containers in detached mode:

   ```bash
   docker-compose up -d --build
   ```

3. To stop the Docker instance:

   ```bash
   docker-compose down
   ```

---

### 5. Access the API

Open your browser and navigate to `http://localhost:8000/docs` to view the interactive API documentation.


## Testing
### Unit Tests
Run unit tests using Pytest:
```bash
pytest tests/
```
These tests focus on the core logic of each module in the `src/services` directory.
* `test_graph_service.py`: Tests the end-to-end flow and error handling within the graph architecture.
* `test_llm_service.py`: Tests that the LLM is initialized correctly and handles prompts.
* `test_memory_service.py`: Tests the storage and retrieval functionality of the conversational memory system.
* `test_retrieval_service.py`: Tests that the document retrieval pipeline works correctly.
These tests are crucial for ensuring the reliability of individual components.
### API Tests
The `postman_test_cases.json` file provides comprehensive API tests which you can use with the Postman application. Import this collection into Postman to check the API functionality and confirm proper integration.