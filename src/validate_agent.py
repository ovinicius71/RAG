from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Parâmetros de validação
MAX_CHARS = 5000   # tamanho máximo da resposta em caracteres
SIMILARITY_THRESHOLD = 0.1  
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

#Carregar modelo de embeddings
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

#calcula similaridade do cosseno entre dois vetores a e b.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

#Valida a resposta da LLM
def validate_llm_output(
    llm_output: str, 
    question: str, 
    retrieved_texts: List[str]
) -> Tuple[bool, dict]:
   
    info = {}

    if len(llm_output) > MAX_CHARS:
        info["reason"] = f"Resposta excede {MAX_CHARS} caracteres (tamanho={len(llm_output)})."
        return False, info

    # 2. Similaridade semântica
    resp_emb = embed_model.encode([llm_output], convert_to_numpy=True)[0]  # shape (dim,)
    # Para cada chunk recuperado, gerar embedding
    chunk_embs = embed_model.encode(retrieved_texts, convert_to_numpy=True)  # shape (K, dim)

    # Calcular similaridades cosseno
    sims = []
    for emb in chunk_embs:
        sims.append(cosine_similarity(resp_emb, emb))
    avg_sim = float(np.mean(sims)) if sims else 0.0

    info["average_similarity_with_chunks"] = avg_sim
    info["individual_similarities"] = sims

    if avg_sim < SIMILARITY_THRESHOLD:
        info["reason"] = (
            f"Similaridade média baixa ({avg_sim:.3f} < {SIMILARITY_THRESHOLD}). "
            "A resposta pode não estar relacionada ao contexto."
        )
        return False, info

    # Se passou em todas as checagens:
    info["reason"] = "Validação bem‐sucedida."
    return True, info
