import os 

#Lê os arquivos .txt em um diretório e retorna um dicionário
def load_documents(folder_path: str) ->dict:             

    docs = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".txt"):
            file_path = os.path.join(folder_path,file_name)
            with open (file_path, "r", encoding="utf-8") as f:
                text = f.read()
            docs[file_name] = text
    return docs

#Divide o texto em chunks de até max_chars preservando paragrafo e retorna um lista de chunks
def split_into_chunk(text: str, max_chars: int = 100) -> list[str]:    

    #separa paragrafo
    raw_paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in raw_paragraphs:
        #se o paragrafo atual for maior que max_chars fais split
        if len(para)> max_chars:
            for i in range (0,len(para), max_chars):
                chunks_piece = para[i : i + max_chars]
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.append(chunks_piece)

        else:
            #verifica se adicioanar esse paragrafo ao chunk atual exede o limite, se sim salva em current_chunk e inicia um novo
            if len(current_chunk) + len(para) + 2 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

    #adicionando o resto dos chunks 
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

if __name__ == "__main__":
    folder = "C:\\RAG\\Docs"
    docs = load_documents(folder)
    for name, text in docs.items():
        chunks = split_into_chunk(text)
        print(f"{name}: {len(chunks)} chunks")
