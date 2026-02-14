import time
import re
from langchain_community.llms import Ollama
from typing import Dict, Any, Tuple, List, Optional
from tools.ReActTools import  DocumentSearchTool, CalculatorTool

class ReActAgent:
    def __init__(self, model_name: str = "phi3.5", max_iterations: int = 5, documents: Dict = None):
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.max_iterations = max_iterations
        self.model_name = model_name
        
        # define tools
        self.doc_tool = DocumentSearchTool(documents or {})
        self.calc_tool = CalculatorTool()
        
        self.tools = {
            "document_search": self.doc_tool,
            "calculator": self.calc_tool,
        }
    
    #get the conversation history pairs if exist
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
    
    
    #get the (thought, action, action input) from the llm response
    def parse_llm_output(self, text: str) -> Tuple[str, str, str]:
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*(.*?)(?=Action Input:|$)", text, re.DOTALL | re.IGNORECASE)
        #ENDAI is the end of action input string
        action_input_match = re.search(r"Action Input:\s*(.*?)(?=ENDAI:|$)", text, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        action_input = action_input_match.group(1).strip() if action_input_match else ""
        return thought, action, action_input
    
    #excute action based on the extracted (action, action input) from llm response 
    def execute_action(self, action: str, action_input: str, query: str) -> str:
        action_lower = action.lower().strip()
        if action_lower == "document_search":
            return self.doc_tool.execute(action_input)
        elif action_lower == "calculator":
            return self.calc_tool.execute(query, action_input)
        elif action_lower == "final_answer":
            return "FINAL_ANSWER"
        else:
            tool_names = ', '.join(self.tools.keys())
            return f"Unknown action: {action}. Available actions: {tool_names}, final_answer"
    
    #The full agent running flow
    def run(self, query: str,  conversation_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        full_conversation = []
        history_context = self.format_conversation_history(conversation_history)
        # Build  prompt
        prompt_parts = []
        prompt_parts.append("You are a reasoning agent using the ReAct (Reasoning + Acting) framework.")
        if history_context:
            prompt_parts.append(history_context)
        prompt_parts.append("Available tools:")
        prompt_parts.append("1. document_search: Search documents")
        prompt_parts.append("2. calculator: Perform calculations")
        prompt_parts.append("3. final_answer: Provide final answer when ready")
        prompt_parts.append(" . ")
        prompt_parts.append("Always use this EXACT format:")
        prompt_parts.append("Thought: [Your reasoning about what to do next]")
        prompt_parts.append("Action: [Tool name from above list]")
        prompt_parts.append("Action Input: [Input for the tool]")
        prompt_parts.append("")
        prompt_parts.append("Don't start the observation until you finish the search for information")
        prompt_parts.append("")
        prompt_parts.append("--Important: add a single string 'ENDAI' right after define Action Input--")
        prompt_parts.append("")
        prompt_parts.append("After receiving Observation, continue with another Thought-Action cycle.")
        prompt_parts.append(" . ")
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append(" . ")
        prompt_parts.append("Let's think step by step:")
        system_prompt = "\n".join(prompt_parts)
        conversation = system_prompt

        final_answer = None
        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1
            try:
                # Get LLM response
                response = self.llm.invoke(conversation)
                # Parse response
                thought, action, action_input = self.parse_llm_output(response)
                # Log this step
                step_info = {
                    "iteration": iterations,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "raw_response": response
                }
                full_conversation.append(step_info)
                # Check for final answer
                if action and action.lower() == "final_answer":
                    final_answer = action_input if action_input else "No answer provided"
                    step_info["observation"] = "Providing final answer"
                    break
                # Execute action if we have one
                if action:
                    observation = self.execute_action(action, action_input, query)
                    step_info["observation"] = observation
                    # Check if we got final answer signal
                    if observation == "FINAL_ANSWER":
                        final_answer = action_input if action_input else "Answer determined"
                        break
                    # Update conversation for next iteration
                    conversation += f"\n\n{response}\nObservation: {observation}"
                else:
                    # No action parsed, break to avoid infinite loop
                    step_info["observation"] = "No valid action parsed"
                    break
                    
            except Exception as e:
                step_info = {
                    "iteration": iterations,
                    "thought": f"Error: {str(e)}",
                    "action": "error",
                    "action_input": "",
                    "observation": f"Exception occurred: {str(e)}"
                }
                full_conversation.append(step_info)
                break
        
        # If loop finish without any final answer, try to extract one
        if not final_answer and full_conversation:
            last_step = full_conversation[-1]
            if last_step.get("action", "").lower() == "final_answer":
                final_answer = last_step.get("action_input", "No answer")
            elif last_step.get("thought"):
                thought_text = last_step["thought"]
                sentences = thought_text.split('.')
                if sentences:
                    final_answer = sentences[0].strip()
        
        return final_answer or "Unable to determine answer", full_conversation


class AgenticWorkflow:
    def __init__(self, model_name: str = "phi3.5", documents: Dict = None):
        self.model_name = model_name
        self.documents = documents or {}

    def run(self, question: str, conversation_history: List[Dict] = None, **kwargs) -> Tuple[str, str, Dict[str, Any]]:
        start_time = time.time()
        # Initialize ReAct agent
        agent = ReActAgent(
            model_name=self.model_name,
            max_iterations=5,
            documents=self.documents
        )
        # Run agent
        answer, agent_conversation = agent.run(question,  conversation_history)
        response_time = time.time() - start_time
        
        # Format the full response showing agent's reasoning process
        full_response = "REACT AGENT EXECUTION:\n\n"
        for i, step in enumerate(agent_conversation):
            full_response += f"=== Iteration {i+1} ===\n"
            full_response += f"Thought: {step.get('thought', 'N/A')}\n"
            full_response += f"Action: {step.get('action', 'N/A')}\n"
            full_response += f"Action Input: {step.get('action_input', 'N/A')}\n"
            if 'observation' in step:
                full_response += f"Observation: {step.get('observation', 'N/A')}\n\n"
        
        full_response += f"\nFINAL ANSWER: {answer}"
        prompt = f"ReAct Agent System Prompt:\nYou are a reasoning agent with access to tools.\nQuestion: {question}"
        if conversation_history:
            prompt += f"\nHistory: {len(conversation_history)} conversation pairs"
        
        # Extract tools used
        tools_used = []
        for step in agent_conversation:
            action = step.get('action', '').lower()
            if action and action != 'final_answer' and action != 'error':
                tools_used.append(action)
        
        metadata = {
            "response_time": response_time,
            "model": self.model_name,
            "iterations": len(agent_conversation),
            "cot_used": False,
            "agent_technique": "ReAct",
            "tools_used": list(set(tools_used)),
            "final_action": agent_conversation[-1].get('action', '') if agent_conversation else 'N/A',
            "workflow": "Agentic",
            "history_used": len(conversation_history) if conversation_history else 0
        }
        
        return full_response, prompt, metadata