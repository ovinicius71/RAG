

# Retrieval-Augmented Generation (RAG) Pipeline with GPT-3.5 Turbo

This repository contains a complete RAG (Retrieval-Augmented Generation) pipeline built from scratch using PyTorch, FAISS, FastAPI, and OpenAI’s GPT-3.5 Turbo. You can ingest local text documents, generate embeddings, index them with FAISS, and expose a RESTful API that answers user queries by retrieving relevant document chunks and generating context-aware answers.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features & Techniques](#features--techniques)  
3. [Prerequisites](#prerequisites)  
4. [Directory Structure](#directory-structure)  
5. [Installation](#installation)  
6. [Prepare Documents](#prepare-documents)  
7. [Generate Embeddings & Build FAISS Index](#generate-embeddings--build-faiss-index)  
8. [Configure OpenAI API Key](#configure-openai-api-key)  
9. [Run the FastAPI Server](#run-the-fastapi-server)  
10. [Test the `/ask` Endpoint](#test-the-ask-endpoint)  
11. [Adjust Semantic Validation](#adjust-semantic-validation)  
12. [Project Files & Code Snippets](#project-files--code-snippets)  
13. [Suggested Visuals for Documentation](#suggested-visuals-for-documentation)  
14. [License](#license)  

---

## Project Overview

A Retrieval-Augmented Generation (RAG) system combines:
1. **Retrieval**: Finding the most relevant pieces of information (chunks) from a large document collection.  
2. **Generation**: Using a large language model (LLM) to generate answers based on the retrieved chunks.  

This project implements an end-to-end RAG pipeline that:
- Loads and chunks local text files.  
- Generates dense vector embeddings for each chunk using a Sentence-Transformers model.  
- Indexes embeddings with FAISS for ultra-fast nearest-neighbor search.  
- Exposes a FastAPI endpoint (`/ask`) that:  
  1. Receives a user question.  
  2. Computes the question embedding.  
  3. Retrieves the top-k most similar document chunks.  
  4. Builds a prompt containing those chunks plus the question.  
  5. Sends the prompt to GPT-3.5 Turbo (via OpenAI’s ChatCompletion API).  
  6. Validates the generated answer semantically (cosine similarity between answer embedding and chunk embeddings).  
  7. Returns the final, validated answer along with which chunks were used.

This RAG pipeline can be adapted to any scenario where you need context-aware answers from a large local knowledge base (e.g., FAQs, manuals, legal documents, research papers, course materials, etc.).

---

## Features & Techniques

- **Document Ingestion & Chunking**  
  - Splits large text files into smaller “chunks” (≈500–1000 characters) while respecting paragraph boundaries.  

- **Embeddings with Sentence-Transformers**  
  - Uses `all-MiniLM-L6-v2` (384-dimensional) to generate high-quality embeddings for each chunk.  
  - Embeddings saved as NumPy arrays for further processing.

- **FAISS Indexing & Similarity Search**  
  - Builds a FAISS `IndexFlatL2` index in memory.  
  - Supports extremely fast k-nearest-neighbor (k-NN) search on hundreds of thousands of chunks.  

- **FastAPI RESTful API**  
  - Exposes a `/ask` endpoint that accepts JSON payloads with a `"question"` field.  
  - Retrieves top-k chunks, builds a prompt, calls GPT-3.5 Turbo, and returns the answer.

- **GPT-3.5 Turbo for Generation**  
  - Integrates with OpenAI’s `ChatCompletion` to generate answers.  
  - Uses `temperature=0.0` for more deterministic responses.

- **Semantic Validation Agent**  
  - After GPT-3.5 Turbo returns an answer, re-embeds the answer and computes cosine similarity against the retrieved chunks.  
  - Ensures `average_similarity ≥ SIMILARITY_THRESHOLD` (default `0.15`) to confirm the answer is aligned with the source context.  
  - Prevents hallucinations and off-topic responses.

- **Pydantic + Error Handling**  
  - Validates incoming JSON requests.  
  - Returns HTTP 400 for empty questions, HTTP 502 if semantic validation fails, etc.

---

## Prerequisites

- Python 3.8+  
- Basic familiarity with:
  - Virtual environments (`venv` or `conda`)  
  - Command-line interface (bash, PowerShell, etc.)  
  - Editing `.env` or setting environment variables  

---

## Directory Structure

```
your-rag-repo/
├── docs/                          # Place your .txt (or PDF→TXT) files here
│   ├── document1.txt
│   ├── document2.txt
│   └── ...
├── src/                           # Source code for RAG pipeline
│   ├── ingestion.py               # Document loader & chunking functions
│   ├── embed_and_index.py         # Embedding generation & FAISS index creation
│   ├── validate_agent.py          # Semantic validation of LLM answers
│   └── api.py                     # FastAPI app exposing /ask endpoint
├── faiss_index.bin                # Auto-generated FAISS index (binary)
├── doc_metadata.pkl               # Auto-generated metadata (pickle)
├── requirements.txt               # (Optional) pip requirements snapshot
└── README.md                      # This file
```

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/your-rag-repo.git
   cd your-rag-repo
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   # On Windows (PowerShell):
   # .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**  
   ```bash
   pip install --upgrade pip
   pip install torch sentence-transformers faiss-cpu fastapi uvicorn openai
   ```
   If you plan to use a GPU for faster embedding inference, install the appropriate torch + CUDA version (e.g., `pip install torch==1.13.1+cu117`).

---

## Prepare Documents

1. Create a `docs/` folder at the project root (if it doesn’t already exist).  
2. Add your `.txt` files (or convert PDFs to TXT) into `docs/`.  
   Example:  
   ```
   docs/
   ├── finance_manual.txt
   ├── engineering_specs.txt
   └── research_paper.txt
   ```
3. Ensure each file is UTF-8 encoded and properly formatted—i.e., paragraphs separated by blank lines (`\n\n`).

---

## Generate Embeddings & Build FAISS Index

1. Navigate to `src/`  
   ```bash
   cd src
   ```

2. Run the embedding & indexing script  
   ```bash
   python embed_and_index.py
   ```

   **What this does:**  
   - Loads all `.txt` files under `../docs/`.  
   - Splits each document into chunks of up to ~1000 characters (preserving paragraph boundaries).  
   - Uses `SentenceTransformer("all-MiniLM-L6-v2")` to encode each chunk into a 384-dimensional vector.  
   - Builds a FAISS index (`IndexFlatL2`) containing all chunk embeddings.  
   - Saves:  
     - `faiss_index.bin` (FAISS index file)  
     - `doc_metadata.pkl` (pickle containing a list of all chunk texts and their `(document_name, chunk_id)` metadata)  

3. Confirm `faiss_index.bin` and `doc_metadata.pkl` exist in the `src/` directory.

---

## Configure OpenAI API Key

1. Obtain your OpenAI API key from [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).  
2. Set the environment variable (so that `api.py` can read it):  
   - **macOS/Linux**  
     ```bash
     export OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXXXXXX"
     ```  
   - **Windows (PowerShell)**  
     ```powershell
     $Env:OPENAI_API_KEY="sk-XXXXXXXXXXXXXXXXXXXXXXXX"
     ```  
   Replace `sk-XXXXXXXXXXXXXXXXXXXXXXXX` with your actual key.

---

## Run the FastAPI Server

From the project root (or from within `src/`), run:  
```bash
uvicorn src.api:app --reload --port 8000
```

You should see logs similar to:  
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
Carregando modelo de embeddings...
Carregando índice FAISS e metadata...
INFO:     Application startup complete.
```

The server will watch for changes in the `src/` directory (because of `--reload`). Do not press any keys in this terminal—Uvicorn “blocks” it to keep the server running.

---

## Test the `/ask` Endpoint

Once the FastAPI server is running, open a second terminal (or tab/PowerShell) and use one of the methods below:

- **Option A: curl (macOS/Linux)**  
  ```bash
  curl -X POST "http://localhost:8000/ask" \
       -H "Content-Type: application/json" \
       -d '{"question":"What is the summary of document1.txt?"}'
  ```

- **Option B: curl.exe (Windows PowerShell)**  
  ```powershell
  curl.exe -X POST "http://localhost:8000/ask" `
           -H "Content-Type: application/json" `
           -d '{"question":"What is the summary of document1.txt?"}'
  ```

- **Option C: Invoke-RestMethod (Windows PowerShell)**  
  ```powershell
  Invoke-RestMethod `
    -Uri "http://localhost:8000/ask" `
    -Method POST `
    -Headers @{ "Content-Type" = "application/json" } `
    -Body '{"question":"What is the summary of document1.txt?"}'
  ```

- **Option D: Swagger UI in Browser**  
  Open your browser and navigate to:  
  ```
  http://127.0.0.1:8000/docs
  ```  
  - Expand `POST /ask`.  
  - Click **Try it out**, paste the JSON body:  
    ```json
    {
      "question": "What is the summary of document1.txt?"
    }
    ```  
  - Click **Execute**.  
  - Examine the response section to see:  
    - `"answer"`: the generated answer from GPT-3.5 Turbo.  
    - `"retrieved_chunks"`: the list of chunks (text snippets) used to build the prompt.

---

## Adjust Semantic Validation

The semantic validation threshold can be adjusted in `src/validate_agent.py`. The default `SIMILARITY_THRESHOLD` is `0.15`. If you find that too many answers are being rejected (HTTP 502), you can lower this value. Conversely, if you want stricter validation, increase it.

---

## Project Files & Code Snippets

For a detailed look at the code, refer to the individual files in the `src/` directory:
- `ingestion.py`: Functions for loading and chunking documents.
- `embed_and_index.py`: Script to generate embeddings and build the FAISS index.
- `validate_agent.py`: Logic for semantic validation of the generated answers.
- `api.py`: FastAPI application code, including the `/ask` endpoint.

---

## Suggested Visuals for Documentation

To enhance the documentation, consider adding the following visuals:
- A flowchart illustrating the RAG pipeline process.
- Screenshots of the FastAPI Swagger UI showing how to use the `/ask` endpoint.
- A diagram of the directory structure for clarity.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```

Este bloco de código Markdown contém todo o conteúdo fornecido no seu pedido, incluindo todas as seções, cabeçalhos, listas, snippets de código e formatação. Você pode copiá-lo e colá-lo diretamente em um arquivo como `README.md` ou outro documento do seu projeto. A estrutura foi mantida intacta, e os snippets de código foram devidamente destacados para facilitar a leitura e o uso.