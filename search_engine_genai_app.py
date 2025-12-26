import sys
sys.modules["tensorflow"] = None
sys.modules["keras"] = None
sys.modules["tf_keras"] = None

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
if not hasattr(st.session_state, "messages"):
    st.session_state.messages = None

# st.write(sys.executable)

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

#ArxivQueryRun: In-built tool by langchain to query the Arxiv's repository
#WikipediaQueryRun: In-built tool by langchain to query the Wikipedia's repository
#DuckDuckGoSearchResults: Package for web search on DuckDuckGo browser
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_classic.agents import initialize_agent, AgentType

# 'StreamlitCallbackHandler' is an important library that allows us to communicate with all kinds of tools [arxiv_tool, wikipedia_tool, retriever_tool] within themselves
from langchain_classic.callbacks import StreamlitCallbackHandler

# Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Arxiv & Wikipedia In-built Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

duckduckgo_search = DuckDuckGoSearchResults(name="Search")

if st.session_state.messages is None:
    st.session_state.messages = [
        {"role": "assistant","content":"Hi, I am a ChatBot who can search the web. How can I help you?"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# System Propmt
SYSTEM_PROMPT = """
You are a research-oriented search assistant.

INSTRUCTIONS:
- Always give DETAILED and WELL-EXPLAINED answers.
- Do NOT be concise unless explicitly asked.
- Explain concepts step-by-step.
- Use headings, bullet points, and examples.
- If the question is conceptual, explain:
  - definition
  - background
  - how it works
  - real-world examples
- If tools are used, SYNTHESIZE the information instead of summarizing it.
- Target response length: 300-600 words unless the user asks otherwise.
"""

# Prompt from User
if prompt := st.chat_input(placeholder="What is Generative AI?"):

    # append user role and prompt to messages
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)

    groq_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", streaming=True)

    tools = [duckduckgo_search,arxiv_tool,wikipedia_tool]

    #Convert 'tools' into Agent
    '''
    'AgentType.ZERO_SHOT_REACT_DESCRIPTION' in LangChain is a stateless agent that uses the ReAct (Reasoning and Acting) framework 
    to decide which tool to use based only on the current input and tool descriptions, without relying on chat history (zero-shot)
    '''
    search_agent = initialize_agent(
        tools,
        groq_llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "system_message": SYSTEM_PROMPT
        },
        handling_parsing_errors=True,
        verbose=True
    )
    
    # When the chatbot is giving response, it should communicate with itself using the tools
    with st.chat_message("assistant"):
        # st.container: Insert a multi-element container. Inserts an invisible container into your app that can be used to hold multiple elements. This allows you to, for example, insert multiple elements into your app out of order. To add elements to the returned container, you can use the with notation (preferred) or just call commands directly on the returned object. See examples below.
        # expand_new_thoughts: shows the thoughts of the chatbot
        streamlit_callback = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)

        # invoke search_agent
        response = search_agent.run(st.session_state.messages,callbacks=[streamlit_callback])

        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)
