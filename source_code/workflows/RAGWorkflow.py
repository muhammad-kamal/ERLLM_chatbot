"""RAG workflow implementation with Conversation History"""
import time
from langchain_community.llms import Ollama
from typing import List, Dict, Any, Tuple, Optional
from tools.RAGSystem import AdvancedRAGSystem


class RAGWorkflow:
    def __init__(self, model_name: str = "phi3.5", embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.rag_system = AdvancedRAGSystem(embedding_model)
    
    def format_conversation_history(self, history: Optional[List[Dict]]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return ""
        formatted = "Previous conversation history:\n"
        for i, pair in enumerate(history, 1):
            question = pair['question']
            answer = pair['answer'][:100] + "..." if len(pair['answer']) > 100 else pair['answer']
            formatted += f"\n{i}. User: {question}"
            formatted += f"\n   Assistant: {answer}"
        return formatted
    
    def run(self, question: str, relevant_docs: Optional[List[str]] = None, 
            context: Optional[str] = None, conversation_history: List[Dict] = None,
            **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        start_time = time.time()
        if relevant_docs is None:
            relevant_docs = []
        relevant_docs = [doc for doc in relevant_docs if doc and str(doc).strip()]
        history_context = self.format_conversation_history(conversation_history)
        # If no valid documents provided, use context if available
        if not relevant_docs and context and str(context).strip():
            relevant_docs = [context]
        # Add history as a document for retrieval
        if history_context:
            relevant_docs.append(f"CONVERSATION HISTORY:\n{history_context}")
        try:
            # Retrieve relevant chunks from the external class RAGSystem
            retrieved_chunks = self.rag_system.retrieve_relevant_chunks(question, relevant_docs, top_k=3)
            # Build the prompt
            prompt_parts = []
            if history_context:
                prompt_parts.append(history_context)
            if retrieved_chunks and retrieved_chunks[0].strip():
                context_text = "\n\n".join([f"[Information {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
                prompt_parts.append("Use ONLY the information provided below to answer the question.")
                prompt_parts.append(context_text)
                prompt_parts.append(f"Question: {question}")
                prompt_parts.append("Provide a direct answer based only on the information above:")
            else:
                # No relevant chunks found
                prompt_parts.append(f"Question: {question}")
                prompt_parts.append("Note: No relevant information was found in the provided documents.")
                prompt_parts.append("Provide an answer based on your general knowledge:")
            
            prompt = "\n\n".join(prompt_parts)
            response = self.llm.invoke(prompt)
            response_time = time.time() - start_time
            metadata = {
                "response_time": response_time,
                "model": self.model_name,
                "rag_chunks": len(retrieved_chunks),
                "cot_used": False,
                "iterations": 1,
                "workflow": "RAG",
                "history_used": len(conversation_history) if conversation_history else 0
            }
            return response, prompt, metadata
            
        except Exception as e:
            # in case of any exception
            print(f"RAG error: {e}. Falling back to simple LLM.")
            fallback_parts = []
            if history_context:
                fallback_parts.append(history_context)
            fallback_parts.append(f"Question: {question}")
            fallback_parts.append("Provide an answer based on your general knowledge:")
            fallback_prompt = "\n\n".join(fallback_parts)
            response = self.llm.invoke(fallback_prompt)
            response_time = time.time() - start_time
            metadata = {
                "response_time": response_time,
                "model": self.model_name,
                "rag_chunks": 0,
                "cot_used": False,
                "iterations": 1,
                "workflow": "RAG",
                "error": str(e),
                "history_used": len(conversation_history) if conversation_history else 0
            }
            return response, fallback_prompt, metadata