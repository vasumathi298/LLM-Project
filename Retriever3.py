import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain.callbacks.base import BaseCallbackHandler

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPEN_API_KEY")
if not openai_api_key:
    st.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Embedding model for the vector database
embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')

# Load the vector database
vector_db_directory = "chroma_vector_db"
vectordb = None
try:
    vectordb = Chroma(persist_directory=vector_db_directory, embedding_function=embedding_model)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
except Exception as e:
    st.error(f"Error initializing vector database: {e}")
    st.stop()

# OpenAI LLM
try:
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1500)
    streaming_llm = ChatOpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key)
except Exception as e:
    st.error(f"Error initializing OpenAI LLM: {e}")
    st.stop()

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# LangGraph chatbot
class State(TypedDict):
    messages: Annotated[list, "The messages in the conversation"]

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

graph = StateGraph(State)
def chatbot(state: State):
    messages = state["messages"]
    response = streaming_llm.invoke(messages)
    return {"messages": messages + [response]}
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()

# Streamlit UI
st.title("Multi-Functional AI Application")

# Sidebar for navigation
menu = st.sidebar.selectbox("Choose an application:", ["Bhagavad Gita Q&A", "LangGraph Chatbot"])

if menu == "Bhagavad Gita Q&A":
    st.header("Bhagavad Gita Question Answering")
    query = st.text_input("Enter your question:")
    if st.button('Get Answer'):
        if query:
            try:
                response = qa_chain.run(query)
                st.write("### Response:")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.warning("Please enter a question.")
elif menu == "LangGraph Chatbot":
    st.header("LangGraph Chatbot")
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
