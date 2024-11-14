import os
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

def process_and_vectorize_company_document(file_content: bytes, document_url: str, index_name: str, company_id: str):
    # Extract text from PDF
    pdf_reader = PdfReader(BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    # Create Document objects
    documents = [Document(page_content=t, metadata={"document_url": document_url, "company_id": company_id}) for t in texts]

    # Initialize Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")

    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists, if not create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=384, metric="cosine")

    # Initialize the embedding model
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Pinecone vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding_function,
        index_name=index_name,
        namespace=company_id,
        pinecone_api_key=pinecone_api_key
    )

    logger.info(f"Company document vectorized and stored in Pinecone index: {index_name}, namespace: {company_id}, document_url: {document_url}")