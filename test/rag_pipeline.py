import os
from groq import Groq
from pinecone import Pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore

def initialize_rag_system():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = "resume-job-description"
    
    pc = Pinecone(api_key=pinecone_api_key)
    docsearch = PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)
    
    return docsearch

def get_relevant_information(query, docsearch):
    relevant_docs = docsearch.similarity_search(query, k=3)
    relevant_info = '\n\n'.join([doc.page_content for doc in relevant_docs])
    return relevant_info

def generate_interview_response(client, model, system_prompt, conversation_history, relevant_info):
    messages = [
        {"role": "system", "content": system_prompt},
    ] + conversation_history + [
        {"role": "user", "content": f"Relevant information from candidate's resume:\n{relevant_info}\n\nBased on this information and the conversation history, generate the next interview question or provide feedback. If the candidate's answer is incorrect or null, address it appropriately."}
    ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    
    return chat_completion.choices[0].message.content