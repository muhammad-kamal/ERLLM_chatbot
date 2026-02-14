import time
from langchain_community.llms import Ollama
from typing import Dict, Any, Tuple, List


class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict] = []
    
    def add_interaction(self, question: str, answer: str, metadata: Dict = None):
        self.history.append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_relevant_history(self, current_question: str, top_k: int = 2) -> List[Dict]:
        """Get relevant conversation history"""
        relevant = []
        current_lower = current_question.lower()
        for item in reversed(self.history):
            item_lower = item["question"].lower()
            # Check for keyword overlap
            if any(word in current_lower for word in item_lower.split()[:5]):
                relevant.append(item)
                if len(relevant) >= top_k:
                    break
        if not relevant:
            relevant = self.history[-top_k:] if len(self.history) >= top_k else self.history
        return relevant
    
    def format_history_for_prompt(self, history: List[Dict]) -> str:
        if not history:
            return ""
        formatted = "Previous conversations that might be relevant:\n"
        for i, item in enumerate(history, 1):
            formatted += f"\n{i}. Q: {item['question']}\n   A: {item['answer'][:200]}...\n"
        return formatted


class MemoryAugmentedWorkflow:
    def __init__(self, model_name: str = "phi3.5"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.memory = ConversationMemory(max_history=10)
    
    def run(self, question: str, context: str = None, 
            conversation_history: List[Dict] = None, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        start_time = time.time()
        relevant_history = self.memory.get_relevant_history(question, top_k=2)
        history_context = self.memory.format_history_for_prompt(relevant_history)
        prompt = f"""{history_context}

Current Question: {question}

{('Additional Context: ' + context) if context else ''}

Consider previous similar questions and their answers when responding.
Provide a comprehensive answer:"""
        response = self.llm.invoke(prompt)
        response_time = time.time() - start_time
        # Add to memory
        self.memory.add_interaction(question, response, {"context": context})
        metadata = {
            "response_time": response_time,
            "model": self.model_name,
            "cot_used": False,
            "memory_used": len(relevant_history),
            "iterations": 1,
            "workflow": "Memory_Augmented",
            "history_used": len(relevant_history)
        }
        return response, prompt, metadata