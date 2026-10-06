import os
import glob
import json
import hashlib
from typing import List, Dict, Any, Optional
import pypdf
from sentence_transformers import SentenceTransformer
import chromadb
import requests

DEFAULT_DOCS_DIR = "/home/lucas/workspace/valhalla/documentos"
DEFAULT_DB_DIR = "/home/lucas/workspace/valhalla/vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434"

class PDFDocumentLoader:
    def __init__(self, docs_dir: str = DEFAULT_DOCS_DIR):
        self.docs_dir = docs_dir

    def extract_chunks(self, chunk_size: int = 700, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
        pdf_paths = glob.glob(os.path.join(self.docs_dir, "**/*.pdf"), recursive=True)
        chunks = []

        for pdf_path in pdf_paths:
            rel_path = os.path.relpath(pdf_path, self.docs_dir)
            path_parts = rel_path.split(os.sep)
            category = path_parts[0] if len(path_parts) > 1 else "geral"
            file_name = os.path.basename(pdf_path)

            try:
                reader = pypdf.PdfReader(pdf_path)
                for page_idx, page in enumerate(reader.pages):
                    page_num = page_idx + 1
                    text = page.extract_text() or ""
                    text = text.strip()
                    if not text:
                        continue

                    # Process text into chunks with overlap
                    start = 0
                    text_len = len(text)
                    c_idx = 0
                    while start < text_len:
                        end = min(start + chunk_size, text_len)
                        chunk_text = text[start:end].strip()

                        if len(chunk_text) > 30: # Ignore tiny noise
                            chunk_id = hashlib.md5(f"{rel_path}_{page_num}_{start}_{c_idx}".encode()).hexdigest()
                            chunks.append({
                                "id": chunk_id,
                                "text": chunk_text,
                                "metadata": {
                                    "file_path": pdf_path,
                                    "rel_path": rel_path,
                                    "file_name": file_name,
                                    "category": category,
                                    "page_number": page_num,
                                    "chunk_index": c_idx
                                }
                            })
                            c_idx += 1
                        
                        if end == text_len:
                            break
                        start += chunk_size - chunk_overlap
            except Exception as e:
                print(f"Erro ao ler PDF {pdf_path}: {e}")

        return chunks

class VectorStoreManager:
    def __init__(self, db_dir: str = DEFAULT_DB_DIR, model_name: str = EMBEDDING_MODEL_NAME):
        self.db_dir = db_dir
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name="normas_valhalla",
            metadata={"hnsw:space": "cosine"}
        )

    def build_index(self, chunks: List[Dict[str, Any]], force: bool = False):
        if not force and self.collection.count() > 0:
            print(f"Banco vetorial já existe com {self.collection.count()} chunks. Use force=True para reconstruir.")
            return

        if force:
            try:
                self.client.delete_collection("normas_valhalla")
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name="normas_valhalla",
                metadata={"hnsw:space": "cosine"}
            )

        if not chunks:
            print("Nenhum chunk para indexar.")
            return

        print(f"Gerando embeddings para {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Batch insert into ChromaDB
        batch_size = 200
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        print(f"Indexação concluída! {self.collection.count()} chunks armazenados.")

    def search(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query]).tolist()[0]
        
        where_clause = None
        if category_filter and category_filter != "Todas":
            where_clause = {"category": category_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause
        )

        search_results = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0]*len(docs)
            ids = results["ids"][0]

            for doc, meta, dist, cid in zip(docs, metas, distances, ids):
                similarity = 1 - dist if dist is not None else 1.0
                search_results.append({
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    "similarity": round(float(similarity), 4)
                })

        return search_results

class NormasRAG:
    def __init__(self, db_dir: str = DEFAULT_DB_DIR, ollama_model: str = OLLAMA_MODEL):
        self.vector_store = VectorStoreManager(db_dir=db_dir)
        self.ollama_model = ollama_model
        self.ollama_url = OLLAMA_URL

    def check_ollama_status(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def query(self, question: str, top_k: int = 5, category_filter: Optional[str] = None) -> Dict[str, Any]:
        retrieved_chunks = self.vector_store.search(question, top_k=top_k, category_filter=category_filter)

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "Não foram encontrados trechos de documentos relevantes para responder à sua pergunta.",
                "sources": []
            }

        # Build context
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk["metadata"]
            source_info = f"[Fonte {idx}: {meta['file_name']} (Página {meta['page_number']}) - Categoria: {meta['category']}]"
            context_blocks.append(f"{source_info}\n{chunk['text']}")

        context_str = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "Você é um assistente especializado em consultar normas institucionais, estatutos, regimentos e resoluções.\n"
            "Responda à pergunta do usuário utilizando ESTRITAMENTE as informações fornecidas no CONTEXTO abaixo.\n"
            "Diretrizes:\n"
            "1. Responda em português claro, conciso e objetivo.\n"
            "2. Sempre cite a fonte (Nome do arquivo e Número da Página) quando mencionar uma regra ou artigo.\n"
            "3. Se o contexto fornecido não contiver a resposta para a pergunta, diga honestamente que a informação não foi encontrada nas normas consultadas.\n"
            "4. Não invente informações que não estejam nos trechos do contexto."
        )

        user_prompt = f"CONTEXTO DOS DOCUMENTOS:\n{context_str}\n\nPERGUNTA DO USUÁRIO:\n{question}"

        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }

        try:
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=60)
            if response.status_code == 200:
                answer = response.json()["message"]["content"]
            else:
                answer = f"Erro na comunicação com o Ollama (Status {response.status_code}): {response.text}"
        except Exception as e:
            answer = f"Erro de conexão com o servidor Ollama: {str(e)}"

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks
        }
