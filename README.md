# Bhagavad Gita Question Answering System

This project is an intelligent question-answering system centered around the teachings of the Bhagavad Gita, designed to offer insights and references that can support mental well-being. By quoting and referring to passages from this ancient text, the system helps users explore valuable perspectives from the Gita. It processes text files and PDFs to create a searchable vector database, enabling contextually relevant answers based on user queries. With the integration of LangChain and OpenAI models, it provides insightful responses grounded in the embedded content of the Bhagavad Gita.

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

## To Dos
- Maintain chat History In UI.
- Fine-tuning Model Responses
- How to handle unexpected questions?
- Host it on web using Docker
- Build an App in Swift

## Sample Q&A
![image](https://github.com/user-attachments/assets/05764c5b-9627-48c6-8f70-356a8559549c)
![image](https://github.com/user-attachments/assets/1bf83e3e-8609-4814-9a96-00fbc7bacf9d)

