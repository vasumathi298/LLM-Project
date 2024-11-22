import os
import time
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch the OpenAI API key securely from environment variables
openai_api_key = os.getenv("OPEN_API_KEY")

if not openai_api_key:
    st.error("API key is not set. Please set the OPENAI_API_KEY environment variable.")
    exit()

# Initialize OpenAI LLM with the API key
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1500)

# Initialize memory to store conversation history, explicitly setting the memory key to 'history'
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create the ConversationChain
conversation = ConversationChain(llm=llm, memory=memory)

# Streamlit UI improvements
st.set_page_config(page_title="Memory-enabled Chatbot", page_icon="🤖", layout="wide")

# Custom CSS to enhance the UI
st.markdown("""
    <style>
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background-color: #f7f7f7;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .user-msg, .bot-msg {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            max-width: 80%;
        }
        .user-msg {
            background-color: #DCF8C6;
            margin-left: auto;
        }
        .bot-msg {
            background-color: #E5E5E5;
        }
        .stTextInput input {
            border-radius: 15px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton button {
            border-radius: 15px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Check if the messages history exists in session_state, and initialize it if not
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Chat UI container
with st.container():
    st.title("Memory-enabled Chatbot 🤖")
    st.write("""
        Start a conversation, and I will remember what we talked about!
    """)

    # Display the chat history in a scrollable container
    for message in st.session_state.messages:
        if "User:" in message:
            st.markdown(f'<div class="user-msg">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{message}</div>', unsafe_allow_html=True)

    # Input box for the user message
    query = st.text_input("Enter your message:", key="input_text", placeholder="Type something...")

    # Define the function to handle message submission
    def handle_message_submission():
        query = st.session_state.input_text
        if query:
            # Display the user's message
            st.session_state.messages.append(f"User: {query}")

            # Simulate a slight delay to make it feel more conversational
            with st.spinner("Thinking..."):
                time.sleep(1)  # Add delay before response

            # Get the response from the chatbot
            response = conversation.predict(input=query)
            st.session_state.messages.append(f"Bot: {response}")

            # Display the bot's response
            st.markdown(f'<div class="bot-msg">{response}</div>', unsafe_allow_html=True)

            # Scroll to the bottom after each new message
            st.rerun()

        else:
            st.warning("Please enter a message.")

    # Button to submit the message and get the answer
    if st.button('Send'):
        handle_message_submission()

    # # Handle Enter key press by triggering the same function
    # if query:
    #     handle_message_submission()