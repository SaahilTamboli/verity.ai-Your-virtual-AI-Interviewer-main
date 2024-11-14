import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client
from routes import user_routes, enterprise_routes
from fastapi.templating import Jinja2Templates

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory of the current file (main.py)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(project_root, "static")), name="static")

# Set up templates
templates = Jinja2Templates(directory=os.path.join(project_root, "templates"))

# Include routers
app.include_router(user_routes.router, prefix="/user")
app.include_router(enterprise_routes.router, prefix="/enterprise")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
