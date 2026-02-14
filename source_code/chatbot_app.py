import streamlit as st
import pandas as pd
import time
import json
import sys
import os
from typing import Dict, Any, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import workflow modules
from workflows.BaselineWorkflow import BaselineWorkflow
from workflows.RAGWorkflow import RAGWorkflow
from workflows.AgenticWorkflow import AgenticWorkflow
from workflows.MemoryAugmentedWorkflow import MemoryAugmentedWorkflow


class ChatbotInterface:
    def __init__(self):
        self.initialize_session_state()
        
        # Initialize workflows
        self.workflows = {}
        self.available_workflows = {
            "Baseline (with Chain-of-Thought)": "baseline_cot",
            "Baseline (without Chain-of-Thought)": "baseline_simple",
            "RAG (Retrieval-Augmented)": "rag", 
            "Agentic (ReAct with Tools)": "agentic",
            "Memory-Augmented": "memory_augmented"
        }
    
    def initialize_session_state(self):
        """Initialize session state variables - Simplified"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "uploaded_docs" not in st.session_state:
            st.session_state.uploaded_docs = {}
        
        if "selected_workflow" not in st.session_state:
            st.session_state.selected_workflow = "baseline_cot"
        
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
    
    def get_last_n_conversation_pairs(self, n: int = 5) -> list:
        if not st.session_state.conversation_history:
            return []
        
        return st.session_state.conversation_history[-n:]
    
    def initialize_workflows(self):
        try:
            documents = st.session_state.uploaded_docs 
            
            self.workflows = {
                "baseline_cot": BaselineWorkflow(model_name="phi3.5", use_cot=True),
                "baseline_simple": BaselineWorkflow(model_name="phi3.5", use_cot=False),
                "rag": RAGWorkflow(model_name="phi3.5"),
                "agentic": AgenticWorkflow(
                    model_name="phi3.5",
                    documents=documents
                ),
                "memory_augmented": MemoryAugmentedWorkflow(model_name="phi3.5")
            }
            return True
        except Exception as e:
            st.error(f"Error initializing workflows: {str(e)}")
            return False
    
    def add_to_conversation_history(self, question: str, answer: str):
        st.session_state.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
            "workflow": st.session_state.selected_workflow
        })
        if len(st.session_state.conversation_history) > 20:
            st.session_state.conversation_history = st.session_state.conversation_history[-20:]
    
    def process_uploaded_file(self, uploaded_file):
        try:
            # if uploaded_file.name.endswith('.txt'):
            #     content = uploaded_file.read().decode('utf-8')
            #     doc_name = uploaded_file.name.replace('.txt', '')
            #     st.session_state.uploaded_docs[doc_name] = content
            #     return f"✓ Added document '{doc_name}' ({len(content)} chars)"
            
            # elif uploaded_file.name.endswith('.json'):
            #     content = json.loads(uploaded_file.read())
            #     if isinstance(content, dict):
            #         st.session_state.uploaded_docs.update(content)
            #         return f"✓ Added {len(content)} documents from JSON"
            #     else:
            #         return "✗ JSON must contain a dictionary of documents"
            
            # elif uploaded_file.name.endswith('.csv'):
            #     df = pd.read_csv(uploaded_file)
            #     for idx, row in df.iterrows():
            #         doc_name = f"doc_{idx}"
            #         content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
            #         st.session_state.uploaded_docs[doc_name] = content
            #     return f"✓ Added {len(df)} documents from CSV"
            
            # else:
            #     return f"✗ Unsupported file type: {uploaded_file.name}"


            content = uploaded_file.read().decode('utf-8')
            doc_name = uploaded_file.name.replace('.txt', '')
            st.session_state.uploaded_docs[doc_name] = content
            return f"✓ Added document '{doc_name}' ({len(content)} chars)"
        except Exception as e:
            return f"✗ Error processing file: {str(e)}"
    
    def get_rag_documents(self) -> List[str]:
        documents = []
        for content in st.session_state.uploaded_docs.items():
            if content and str(content).strip():
                documents.append(str(content))
        
        # If no uploaded documents, use conversation history as fallback
        if not documents and st.session_state.conversation_history:
            history_text = "Previous conversation history:\n"
            for i, pair in enumerate(st.session_state.conversation_history[-5:], 1):
                history_text += f"\n{i}. Q: {pair['question']}\n   A: {pair['answer'][:100]}..."
            documents.append(history_text)
        
        return documents
    
    def estimate_tokens(self, text: str) -> int:
        #suppose that the word is 4 chars long in avrage
        return max(1, len(text) // 4)
    
    def run_workflow(self, workflow_id: str, question: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self.workflows:
                self.initialize_workflows()
            workflow = self.workflows[workflow_id]
            # Get last 5 conversation pairs for history
            conversation_history = self.get_last_n_conversation_pairs(5)
            # Prepare context based on workflow type
            context = None
            relevant_docs = []
            if workflow_id == "rag":
                relevant_docs = self.get_rag_documents()
                if not relevant_docs or len(relevant_docs) == 0:
                    st.warning("⚠️ No documents uploaded. RAG workflow will use only conversation history.")
                response, prompt, metadata = workflow.run(
                    question=question,
                    relevant_docs=relevant_docs,
                    conversation_history=conversation_history
                )
            elif workflow_id == "memory_augmented":
                response, prompt, metadata = workflow.run(
                    question=question,
                    context=None,
                    conversation_history=conversation_history
                )
            else:
                # For Baseline and Agentic: Prepare context from uploaded docs
                if st.session_state.uploaded_docs:
                    context = "\n".join([
                        f"Document '{name}': {content}"
                        for name, content in st.session_state.uploaded_docs.items()
                    ])
                
                response, prompt, metadata = workflow.run(
                    question=question,
                    context=context,
                    conversation_history=conversation_history
                )
            
            # Calculate performance metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            # Estimate tokens
            prompt_tokens = self.estimate_tokens(prompt)
            response_tokens = self.estimate_tokens(response)
            total_tokens = prompt_tokens + response_tokens
            
            # Add to conversation history
            self.add_to_conversation_history(question, response)
            
            return {
                "success": True,
                "response": response,
                "prompt": prompt,
                "metadata": metadata,
                "performance": {
                    "response_time": round(response_time, 2),
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "latency_ms": round(response_time * 1000, 0),
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"Error: {str(e)}",
                "performance": {
                    "response_time": round(time.time() - start_time, 2),
                    "total_tokens": 0,
                    "latency_ms": 0,
                }
            }
    
    def display_conversation_history_panel(self):
        if st.session_state.conversation_history:
            with st.sidebar.expander(f"💬 Conversation History ({len(st.session_state.conversation_history)} pairs)", expanded=False):
                for i, pair in enumerate(reversed(st.session_state.conversation_history[-10:]), 1):
                    st.markdown(f"**Q{i}:** {pair['question'][:80]}..." if len(pair['question']) > 80 else f"**Q{i}:** {pair['question']}")
                    st.caption(f"*Workflow: {pair['workflow']}*")
                    st.markdown(f"**A{i}:** {pair['answer'][:100]}..." if len(pair['answer']) > 100 else f"**A{i}:** {pair['answer']}")
                    st.markdown("---")
                
                if st.button("Clear History", key="clear_history_btn", use_container_width=True):
                    st.session_state.conversation_history = []
                    st.rerun()
    
    def render_chat_interface(self):
        st.title("AI Assistant")
        st.markdown("Choose between 5 different reasoning workflows!")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Workflow selection
            workflow_options = list(self.available_workflows.keys())
            selected_display = st.selectbox(
                "Select Workflow Strategy:",
                workflow_options,
                index=0,
                help="Choose how the AI should process your questions"
            )
            
            st.session_state.selected_workflow = self.available_workflows[selected_display]
            
            st.markdown(f"**Selected:** `{selected_display}`")
            
            with st.expander("Workflow Descriptions", expanded=False):
                st.markdown("""
                **Baseline (with CoT):** Step-by-step reasoning with explicit thinking  
                **Baseline (without CoT):** Direct answer without reasoning steps  
                **RAG (Retrieval-Augmented):** Searches uploaded documents for answers  
                **Agentic (ReAct):** Uses tools for search, calculation, and reasoning  
                **Memory-Augmented:** Learns from conversation history
                """)
            
            # Document upload section
            st.markdown("---")
            st.markdown("### Upload Documents")
            
            uploaded_file = st.file_uploader(
                "Upload documents",
                type=["txt"],
                help="Workflow will search these documents for answers"
            )
            
            if uploaded_file is not None:
                result = self.process_uploaded_file(uploaded_file)
                st.success(result)
            
            if st.session_state.uploaded_docs:
                with st.expander(f"Uploaded Documents ({len(st.session_state.uploaded_docs)})", expanded=False):
                    for doc_name, content in st.session_state.uploaded_docs.items():
                        st.caption(f"**{doc_name}**")
                        st.text(content[:200] + "..." if len(content) > 200 else content)
            
            self.display_conversation_history_panel()
            
        
        # Main chat area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    if message["role"] == "assistant" and "metadata" in message:
                        with st.expander("Response Details", expanded=False):
                            metadata = message["metadata"]
                            st.caption(f"**Workflow:** {metadata.get('workflow', 'N/A')}")
                            st.caption(f"**Response Time:** {metadata.get('response_time', 0):.2f}s")
                            st.caption(f"**Tokens Used:** {metadata.get('total_tokens', 0)}")
                            
                            # Show RAG-specific info
                            if st.session_state.selected_workflow == "rag" and "rag_chunks" in metadata:
                                st.caption(f"**Chunks Retrieved:** {metadata['rag_chunks']}")
                            
                            # Show if CoT was used
                            if "cot_used" in metadata:
                                st.caption(f"**Chain-of-Thought:** {'Yes' if metadata['cot_used'] else 'No'}")
        
        with col2:
            # Quick stats
            st.metric("Uploaded Docs", len(st.session_state.uploaded_docs))
            
            if st.session_state.conversation_history:
                st.metric("History Pairs", len(st.session_state.conversation_history))
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                with st.spinner(f"Thinking with {selected_display}..."):
                    result = self.run_workflow(st.session_state.selected_workflow, prompt)
                
                if result["success"]:
                    message_placeholder.markdown(result["response"])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "metadata": {
                            "workflow": st.session_state.selected_workflow,
                            **result["performance"],
                            "cot_used": result["metadata"].get("cot_used", False) if "metadata" in result else False,
                            **{k: v for k, v in result["metadata"].items() if k not in ["cot_used", "response_time", "model"]}
                        }
                    })
                    
                    # Show performance metrics inline
                    perf = result["performance"]
                    with st.expander("📈 Performance Metrics", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Response Time", f"{perf['response_time']}s")
                        
                        with col2:
                            st.metric("Tokens", perf['total_tokens'])
                        
                        with col3:
                            st.metric("Latency", f"{perf['latency_ms']}ms")
                        
                        st.caption(f"Prompt: {perf['prompt_tokens']} tokens | Response: {perf['response_tokens']} tokens")
                        
                        # Show workflow-specific details
                        if result["metadata"].get("cot_used", False):
                            st.caption("🔍 Chain-of-Thought reasoning was used")
                        elif st.session_state.selected_workflow == "memory_augmented":
                            st.caption("🧠 Using conversation memory")
                        elif st.session_state.selected_workflow == "rag":
                            st.caption("🔍 Searching document chunks for answers")
                        elif st.session_state.selected_workflow == "agentic":
                            st.caption("⚙️ Using tools for reasoning")
                        
                        # Show full prompt and response
                        with st.expander("🔍 View Full Prompt & Response", expanded=False):
                            st.text_area("Prompt Content", result["prompt"], height=200, disabled=True)
                            st.text_area("Full Response", result["response"], height=200, disabled=True)
                else:
                    message_placeholder.error(result["response"])
        
        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Multi-Workflow AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage.user {
        background-color: rgba(0, 123, 255, 0.1);
    }
    .stChatMessage.assistant {
        background-color: rgba(40, 167, 69, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    chatbot = ChatbotInterface()
    chatbot.render_chat_interface()


if __name__ == "__main__":
    main()