from supabase import create_client, Client
import os 
from dotenv import load_dotenv

load_dotenv()

def get_supabase() -> Client:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def get_supabase_admin() -> Client:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)