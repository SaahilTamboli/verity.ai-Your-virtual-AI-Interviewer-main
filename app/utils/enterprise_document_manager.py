import os
import uuid
from supabase import create_client, Client

async def store_enterprise_document(file_content: bytes, filename: str, company_id: str) -> str:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    file_extension = os.path.splitext(filename)[1]
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = f"company_documents/{company_id}/{file_name}"

    try:
        response = supabase.storage.from_("company_documents").upload(file_path, file_content)
        
        # The new response format doesn't have an 'error' attribute
        # Instead, if the upload is successful, it returns the path
        if response:
            public_url = supabase.storage.from_("company_documents").get_public_url(file_path)
            return public_url
        else:
            raise Exception("Failed to upload file: No response received")
    except Exception as e:
        raise Exception(f"Failed to upload file: {str(e)}")