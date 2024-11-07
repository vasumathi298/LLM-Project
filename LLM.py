import os
import fitz  # PyMuPDF for reading PDF
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embedding model with LangChain's SentenceTransformerEmbeddings wrapper
embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Prepare transcript data
transcript_data = []
if not os.path.exists("transcripts"):
    print("The transcripts directory does not exist!")
    exit()

# Read and process transcript text files
print("Reading transcript files...")
for filename in os.listdir("transcripts"):
    if filename.endswith(".txt"):
        with open(os.path.join("transcripts", filename), 'r') as file:
            text = file.read()
            chunks = text_splitter.split_text(text)
            transcript_data.extend([Document(page_content=chunk) for chunk in chunks])
            print(f"Processed file: {filename}, Number of chunks: {len(chunks)}")

# Read Bhagavad Gita PDF and convert it to chunks
def process_pdf(pdf_path):
    print(f"Processing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)  # Open the PDF file
    pdf_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pdf_text += page.get_text("text")  # Extract text from each page
    
    # Split the extracted text into chunks
    chunks = text_splitter.split_text(pdf_text)
    return [Document(page_content=chunk) for chunk in chunks]

# Add Bhagavad Gita PDF to transcript data
pdf_path = "book/bhagavad-gita.pdf"  # Path to the Bhagavad Gita PDF
if os.path.exists(pdf_path):
    print("Processing Bhagavad Gita PDF...")
    gita_chunks = process_pdf(pdf_path)
    transcript_data.extend(gita_chunks)
    print(f"Processed Bhagavad Gita PDF, Number of chunks: {len(gita_chunks)}")

# Function to split the documents into smaller batches
def batch_documents(documents, batch_size=100):
    return [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

# Check if the vector database already exists
vector_db_directory = "chroma_vector_db"
vectordb = None
try:
    if os.path.exists(vector_db_directory):
        print(f"Vector database already loaded from {vector_db_directory}.")
        vectordb = Chroma(persist_directory=vector_db_directory, embedding_function=embedding_model)
    else:
        print(f"Total number of documents to process: {len(transcript_data)}")
        # Process all documents in batches
        batched_data = batch_documents(transcript_data, batch_size=100)

        total_batches = len(batched_data)
        all_documents = []

        for batch_index, batch in enumerate(batched_data, start=1):
            print(f"Processing batch {batch_index}/{total_batches} with {len(batch)} documents...")
            all_documents.extend(batch)
            print(f"Successfully processed batch {batch_index}. Total documents processed: {len(all_documents)}")

        # Create Chroma vector store with the embedding model
        vectordb = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            persist_directory=vector_db_directory  # Specify the directory to persist
        )
        vectordb.persist()  # Persist the database to the specified directory
        print(f"Vector database saved to {vector_db_directory}.")

except ImportError as e:
    print("Error: Could not import `chromadb` package. Please install it with `pip install chromadb`.")

except Exception as e:
    print(f"Error occurred while processing or storing embeddings: {e}")

# Check if vectordb was successfully created before proceeding
if vectordb:
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    query = "How to be happy in life as mentioned by Bhagavad Gita?"
    retrieved_docs = retriever.invoke(query)
    print(f"Number of documents retrieved for query '{query}': {len(retrieved_docs)}")
else:
    print("Vector database was not created. Please check for errors and try again.")
