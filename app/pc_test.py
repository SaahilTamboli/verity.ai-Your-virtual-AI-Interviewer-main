from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("resume-job-description")

# Query the index with the namespace
# results = index.query(
#     vector=[0] * 384,  # Replace with an actual query vector
#     top_k=10,
#     namespace="aa7f06be-a26a-415c-b896-ed9951ffb2f9"  # Use the namespace returned from the upload
# )
# print(results)

source = " https://ylomxpsxxeavqkbpsttb.supabase.co/rest/v1/users?user_id=eq.3f684437-71cc-44ad-8c32-e149623bd32d"

results = index.query(
    vector=[0] * 384,  # Replace with an actual query vector
    top_k=10,
    
    filter={"source": source}
)
print(results)