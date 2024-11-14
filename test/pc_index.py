from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="key")

pc.create_index(
    name="resume-job-description",
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)