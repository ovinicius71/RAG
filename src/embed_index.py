import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ingestion import load_documents, split_into_chunk

MODEL_NAME = "all-MiniLM-L6-v2"

#gera o embedding para cada chunck e indexa o embedding no faiss
def building_embedding_index(
        docs_folder : str,
        index_path: str = "faiss_index.bin",
        meta_path: str = "doc_metadata.pkl",
):
    #carrega os documentos e gera uma chunk para cada um
    documents = load_documents(docs_folder)
    all_chunks = []
    metadata = []
    for doc_name, content in documents.items():
        chunks = split_into_chunk (content)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append((doc_name, idx))
    
    #carrega o modelo que vai gerar o embedding
    model = SentenceTransformer(MODEL_NAME)
    #produz um array de numpy
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)

    #criando indece FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim) #indice de similiaridade L2
    index.add (embeddings)

    faiss.write_index(index,index_path)
    with open (meta_path, "wb") as f :
        pickle.dump({
            "chunks": all_chunks,
            "metadata": metadata
        },f)

    print(f"=== Index criado com {len(all_chunks)} chunks (dim={dim}) ===")
    print(f"indece Faiss salvo em: {index_path}")
    print(f"metadata salvo em: {meta_path}")

if __name__ == "__main__":
    building_embedding_index(docs_folder="C:\\RAG\\Docs")