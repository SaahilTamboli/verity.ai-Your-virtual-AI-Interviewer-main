import os
import logging
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from io import BytesIO

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Pinecone
PINECONE_API_KEY = ""
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
pc = Pinecone(api_key=PINECONE_API_KEY)

def extract_text_from_pdf(pdf_content: bytes) -> str:
    pdf_file = BytesIO(pdf_content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def store_resume_in_pinecone(texts: List[str], index_name: str, file_url: str):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        documents = [Document(page_content=text, metadata={"source": file_url}) for text in texts]
        
        index = pc.Index(index_name)
        vectorstore = Pinecone(index, embedding_function, "text")
        vectorstore.add_documents(documents)
        
        logger.info(f"Resume stored in Pinecone index: {index_name}")

    except Exception as e:
        logger.error(f"Error storing resume in Pinecone: {str(e)}")
        raise

def process_and_vectorize_resume(file_content: bytes, file_url: str, index_name: str, user_id: str):
    try:
        logger.info(f"Starting resume processing for user: {user_id}")
        
        # Extract text from PDF
        text = extract_text_from_pdf(file_content)
        logger.info(f"Text extracted from PDF for user: {user_id}. Length: {len(text)} characters")

        # Ensure the index exists
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        index = pc.Index(index_name)
        
        # Check for existing resume and delete if found
        existing_vectors = index.query(vector=[0]*384, filter={"user_id": user_id}, top_k=1)
        if existing_vectors.matches:
            logger.info(f"Existing resume found for user: {user_id}. Deleting...")
            index.delete(ids=[match.id for match in existing_vectors.matches])

        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        documents = [Document(page_content=text, metadata={"source": file_url, "user_id": user_id})]
        
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embedding_function,
            index_name=index_name,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        logger.info(f"Resume processed and vectorized in Pinecone index: {index_name} for user: {user_id}")

        # Verify that the document was added
        verification_query = index.query(vector=[0]*384, filter={"user_id": user_id}, top_k=1)
        if verification_query.matches:
            logger.info(f"Resume successfully added to Pinecone index: {index_name} for user: {user_id}")
        else:
            logger.error(f"Failed to verify resume in Pinecone index for user: {user_id}")

    except Exception as e:
        logger.error(f"Error processing and vectorizing resume: {str(e)}", exc_info=True)
        raise
