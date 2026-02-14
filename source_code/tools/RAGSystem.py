import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


class AdvancedRAGSystem:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def retrieve_relevant_chunks(self, query: str, documents: List[str], top_k: int = 3) -> List[str]:
        valid_documents = [doc for doc in documents if doc and str(doc).strip()]
        if not valid_documents:
            return ["No documents available for retrieval."]
        all_chunks = []
        all_embeddings = []
        chunk_to_text = {}
        for i, doc_text in enumerate(valid_documents):
            try:
                chunks = self.text_splitter.split_text(str(doc_text))
                for chunk in chunks:
                    if chunk.strip():  # if not empty chunks
                        chunk_id = hashlib.md5(chunk.encode()).hexdigest()[:10]
                        embedding = self.embedding_model.encode(chunk)
                        all_chunks.append(chunk_id)
                        all_embeddings.append(embedding)
                        chunk_to_text[chunk_id] = chunk
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                continue

        if not all_embeddings:
            return ["No valid content found in documents."]
        
        try:
            all_embeddings = np.array(all_embeddings)
            query_embedding = self.embedding_model.encode(query).reshape(1, -1)
            if all_embeddings.shape[1] != query_embedding.shape[1]:
                print(f"Embedding dimension mismatch: doc={all_embeddings.shape[1]}, query={query_embedding.shape[1]}")
                return [chunk_to_text[all_chunks[0]]] if all_chunks else ["Content retrieval failed."]
            similarities = np.dot(all_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            retrieved_chunks = []
            for idx in top_indices:
                if idx < len(all_chunks):
                    chunk_id = all_chunks[idx]
                    retrieved_chunks.append(chunk_to_text[chunk_id])
            return retrieved_chunks if retrieved_chunks else ["No relevant content found."]
            
        #in case of any exception    
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            # Return first 3 chunks as fallback
            fallback_chunks = []
            for i in range(min(3, len(all_chunks))):
                chunk_id = all_chunks[i]
                fallback_chunks.append(chunk_to_text[chunk_id])
            return fallback_chunks if fallback_chunks else ["Retrieval system error."]