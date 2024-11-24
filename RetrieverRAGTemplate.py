import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize ChromaDB and retriever
embeddings = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')

vectorstore = Chroma(
    persist_directory="chroma_vector_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

# Define the prompt template
template = """As Srila Prabhupada, answer this question based on the Bhagavad Gita teachings and lectures:

Context from Bhagavad Gita and lectures: {context}

Previous conversation:
{chat_history}

Devotee's Question: {question}

Instructions:
1. If the question asks for a specific verse, always quote the exact verse in Sanskrit and provide its English translation.
2. Include the chapter and verse number (e.g., BG 16.7) when referencing a verse.
3. After quoting the verse, provide an explanation based on Srila Prabhupada's teachings.

My dear devotee, let me explain this point according to the Bhagavad Gita's teachings:"""

rag_prompt = PromptTemplate(
    template=template,
    input_variables=["context", "chat_history", "question"]
)
def format_docs(docs):
    return "\n\n".join(f"Chapter {doc.metadata.get('chapter', 'N/A')}, Verse {doc.metadata.get('verse', 'N/A')}: {doc.page_content}" for doc in docs)

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
    new_context = format_docs(docs)

    combined_context = f"{state.get('context', '')}\n\n{new_context}".strip()

    return {
        "messages": state["messages"],
        "context": combined_context,
        "question": question
    }

def generate_response(state: State):
    chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in state["messages"][:-1]])
    
    response = llm.invoke(
        rag_prompt.format(
            context=state["context"],
            chat_history=chat_history,
            question=state["question"]
        )
    )

    updated_messages = state["messages"] + [{"role": "assistant", "content": response.content}]
    
    return {
        "messages": updated_messages,
        "context": state["context"],
        "question": state["question"]
    }

# Set up the graph
graph.add_node("retriever", retrieve_context)
graph.add_node("generator", generate_response)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", END)

# Compile the graph
app = graph.compile()

# Streamlit UI
st.title("Bhagavad Gita RAG Chatbot")

if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "context": "",
        "question": ""
    }

for message in st.session_state.state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question about the Bhagavad Gita:"):
    st.session_state.state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_container = st.empty()
        with stream_container:
            callback = StreamlitCallbackHandler(stream_container)

            new_state = app.invoke(
                st.session_state.state,
                {"callbacks": [callback]}
            )

        st.session_state.state = new_state