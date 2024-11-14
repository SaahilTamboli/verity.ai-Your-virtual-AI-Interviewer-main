import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def extract_text_from_resume(file_path):
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a PDF file.")
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def store_resume_in_pinecone(texts, index_name):
    pinecone_api_key = "key"
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists, if not create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=384, metric="cosine")

    # Initialize the embedding model
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Pinecone vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embedding_function,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key
    )

    print(f"Resume stored in Pinecone index: {index_name}")

def main():
    resume_path = "test_resume.pdf"
    index_name = "resume-job-description"

    if not os.path.exists(resume_path):
        print("Error: The specified file does not exist.")
        return

    if not resume_path.lower().endswith('.pdf'):
        print("Error: Please provide a PDF file.")
        return

    # Check if PINECONE_API_KEY is set
    if not os.getenv('PINECONE_API_KEY'):
        print("Error: PINECONE_API_KEY environment variable is not set.")
        return

    try:
        texts = extract_text_from_resume(resume_path)
        store_resume_in_pinecone(texts, index_name)
        print("Resume successfully stored in vector form in Pinecone.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()