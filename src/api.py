
import os
import pickle
import faiss
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from validate_agent import validate_llm_output

# == CONFIGURAÇÕES ==

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY com sua chave da OpenAI.")

client = OpenAI(api_key=openai_api_key)

INDEX_PATH = "faiss_index.bin"
META_PATH = "doc_metadata.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

print("Carregando modelo de embeddings...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Carregando índice FAISS e metadata...")
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise RuntimeError("Índice FAISS ou metadata não encontrados. Execute embed_and_index.py primeiro.")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta_data = pickle.load(f)
chunks_texts: List[str] = meta_data["chunks"]
chunks_meta: List[tuple[str, int]] = meta_data["metadata"]

app = FastAPI(
    title="RAG com OpenAI 1.x",
    version="1.0",
    description="API RAG usando FAISS + sentence-transformers + OpenAI v1.x"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str]

def retrieve_top_k_chunks(query: str, k: int = TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    indices = indices[0]
    retrieved_texts = []
    retrieved_meta = []
    for idx in indices:
        if 0 <= idx < len(chunks_texts):
            retrieved_texts.append(chunks_texts[idx])
            retrieved_meta.append(chunks_meta[idx])
    return retrieved_texts, retrieved_meta

def build_prompt(question: str, retrieved_texts: List[str]) -> str:
    prompt = "Você é um assistente que responde com base no contexto fornecido.\n"
    prompt += "====== CONTEXTO RECUPERADO ======\n"
    for i, chunk in enumerate(retrieved_texts):
        prompt += f"CHUNK {i+1}:\n{chunk}\n\n"
    prompt += "====== PERGUNTA ======\n"
    prompt += question + "\n\n"
    prompt += "Com base no contexto acima, responda de forma objetiva:"
    return prompt

def call_openai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # ou “gpt-4”
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Erro ao chamar OpenAI (nova interface): {e}")

@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A pergunta não pode ser vazia.")

    # 1. Recuperar top-K chunks
    retrieved_texts, retrieved_meta = retrieve_top_k_chunks(question)

    # 2. Montar prompt
    prompt = build_prompt(question, retrieved_texts)

    # 3. Chamar OpenAI
    llm_output = call_openai(prompt)

    # 4. Validar saída
    is_valid, validation_info = validate_llm_output(llm_output, question, retrieved_texts)
    if not is_valid:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Saída da LLM não passou pela validação interna.",
                "validation_info": validation_info
            }
        )

    # 5. Retornar resposta
    return QueryResponse(answer=llm_output, retrieved_chunks=retrieved_texts)
