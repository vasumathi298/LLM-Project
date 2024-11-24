import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from typing_extensions import Annotated
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
 
 
load_dotenv()
 
# Set the OpenAI API key
openai_api_key = os.getenv("OPEN_API_KEY")
 
# Check if the API key is loaded
if openai_api_key:
    print("API Key loaded successfully:", openai_api_key)
else:
    print("Failed to load API Key.")
    st.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    exit()
 
# Initialize ChromaDB and retriever
# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
embeddings = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
 
vectorstore = Chroma(
    persist_directory="chroma_vector_db",
    embedding_function=embeddings
)
 
vectordb = None
 
# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
 
template = """As Srila Prabhupada, answer this question based on the Bhagavad Gita teachings and lectures:
 
Context from Bhagavad Gita and lectures: {context}
 
Devotee's Question: {question}
 
My dear devotee, let me explain this point according to the Bhagavad Gita's teachings:"""
 
rag_prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
 
# Define the state
class State(TypedDict):
    messages: Annotated[list, "The messages in the conversation"]
    context: Annotated[str, "Retrieved context"]
    question: Annotated[str, "User question"]
 
# Create a Streamlit callback handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
   
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
 
# Set up the LLM
llm = ChatOpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key)
 
# Create the graph
graph = StateGraph(State)
 
# Define the nodes
def retrieve_context(state: State):
    question = state["messages"][-1]["content"]
    docs = retriever.get_relevant_documents(question)
    context = format_docs(docs)
    return {"messages": state["messages"], "context": context, "question": question}
 
def generate_response(state: State):
    response = llm.invoke(
        rag_prompt.format(
            context=state["context"],
            question=state["question"]
        )
    )
    return {"messages": state["messages"] + [response]}
 
# Set up the graph
graph.add_node("retriever", retrieve_context)
graph.add_node("generator", generate_response)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", END)
 
# Compile the graph
app = graph.compile()
 
# Streamlit UI
st.title("LangChain + LangGraph + Streamlit RAG Chatbot")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    with st.chat_message("assistant"):
        stream_container = st.empty()
        with stream_container:
            callback = StreamlitCallbackHandler(stream_container)
            response = app.invoke(
                {"messages": st.session_state.messages},
                {"callbacks": [callback]}
            )
        assistant_message = response["messages"][-1]
        st.session_state.messages.append({"role": "assistant", "content": assistant_message.content})
has context menu