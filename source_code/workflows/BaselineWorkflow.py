import time
from langchain_community.llms import Ollama
from typing import Dict, Any, Tuple, List, Optional


class BaselineWorkflow:
    def __init__(self, model_name: str = "phi3.5", use_cot: bool = True):
        self.model_name = model_name
        self.use_cot = use_cot
        self.llm = Ollama(model=model_name, temperature=0.1)
    
    def format_conversation_history(self, history: Optional[List[Dict]]) -> str:
        if not history:
            return ""
        formatted = "Previous conversation history:\n"
        for i, pair in enumerate(history, 1):
            question = pair['question']
            answer = pair['answer'][:100] + "..." if len(pair['answer']) > 100 else pair['answer']
            formatted += f"\n{i}. User: {question}"
            formatted += f"\n   Assistant: {answer}"
        return formatted
    
    #Create prompt with cot
    def enhance_with_cot(self, question: str, context: str = None, history: List[Dict] = None) -> str:
        history_context = self.format_conversation_history(history)
        # Build prompt
        prompt_parts = []
        if history_context:
            prompt_parts.append(history_context)
        if context:
            prompt_parts.append(f"Context Information:\n{context}")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("I'll reason through this carefully:")
        prompt_parts.append("1. First, I need to parse and understand exactly what the question is asking. Let me identify the key entities and relationships.")
        prompt_parts.append("2. Next, I'll examine the context provided to find relevant information.")
        prompt_parts.append("3. I need to connect different pieces of information if this is a multi-hop question.")
        prompt_parts.append("4. I'll synthesize the information to form a coherent answer.")
        prompt_parts.append("5. Finally, I'll verify that my answer addresses all parts of the question.")
        prompt_parts.append("Step-by-step reasoning:")
        return "\n\n".join(prompt_parts)
    
    #create simple prompt without cot
    def simple_prompt(self, question: str, context: str = None, history: List[Dict] = None) -> str:
        history_context = self.format_conversation_history(history)
        prompt_parts = []
        if history_context:
            prompt_parts.append(history_context)
        if context:
            prompt_parts.append(f"Context Information:\n{context}")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Provide a direct answer based on the available information:")
        return "\n\n".join(prompt_parts)
    
    def run(self, question: str, context: str = None, 
            conversation_history: List[Dict] = None, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        start_time = time.time()
        if self.use_cot:
            prompt = self.enhance_with_cot(question, context, conversation_history)
        else:
            prompt = self.simple_prompt(question, context, conversation_history)
        response = self.llm.invoke(prompt)
        response_time = time.time() - start_time
        metadata = {
            "response_time": response_time,
            "model": self.model_name,
            "cot_used": self.use_cot,
            "iterations": 1,
            "workflow": "Baseline",
            "history_used": len(conversation_history) if conversation_history else 0
        }
        return response, prompt, metadata