==== Code setup requirements ====
1- Download ollama desktop application: https://ollama.com/download/windows
2- Download Python programming language: https://www.python.org/downloads/   needed version = Python 3.11.7
3- install important python libraries : 
   pip install langchain==0.2.12 langchain-core==1.2.4 langchain-community==0.4.1 langchain-ollama==0.1.1 streamlit==1.51.0 pandas==2.3.3 sentence-transformers==5.1.2 numpy==1.26.4 ollama==0.6.0
4- install LLM models using Ollama : 
   ollama pull phi3.5:latest
   ollama pull gemma2:2b

==== Run the chatbot interface ====
open two sparated terminals and run the following :
terminal 1 : 
    ollama start
terminal 2 : 
    streamlit run chatbot_app.py
    CTRL + left click on the link to open the interface locally.

==== Notes ==== 
- Checking ollama installed models: ollama list
- Remove ollama model: ollama rm model-name
- Checking Python version: python --version
- Upload txt file in the input field in the interface bottom left(example here data.txt).
- Choose the workflow from the drop down list in the left side of the interface.

