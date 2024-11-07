# Bhagavad Gita Question Answering System

This project is a question-answering system focused on the Bhagavad Gita. It processes text files and PDFs to generate a searchable vector database for retrieving contextually relevant answers from the text. Users can ask questions, and the system leverages LangChain and OpenAI's models to provide responses based on the embedded documents.

## Features

- **Web Scraping**: Automatically fetches lecture transcripts from an online source.
- **PDF Processing**: Processes the Bhagavad Gita PDF into manageable text chunks.
- **Embedding & Vector Storage**: Converts text into embeddings stored in a Chroma vector database.
- **Question Answering**: Uses LangChain's `RetrievalQA` chain with OpenAI's language model to answer questions.
- **Streamlit UI**: A user-friendly interface for entering questions and viewing answers.

## Technologies Used

- **Python**: Main programming language.
- **LangChain**: Framework for building applications with language models, specifically for embedding and retrieval.
- **Chroma**: Vector database to store and retrieve document embeddings.
- **SentenceTransformer**: Converts text chunks into embeddings using `paraphrase-MiniLM-L6-v2`.
- **OpenAI API**: For generating answers with OpenAI's language models.
- **PyMuPDF (fitz)**: Library to process and extract text from PDF files.
- **BeautifulSoup**: Library for web scraping to fetch lecture transcripts.
- **Streamlit**: Framework for building the web-based user interface.

## Setup Instructions

### Prerequisites

- **Python 3.7+** installed on your system.
- **Environment Variables**: Set `OPENAI_API_KEY` in your environment for the OpenAI integration.
