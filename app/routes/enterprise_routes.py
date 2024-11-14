from fastapi import APIRouter, Request, Form, HTTPException, Depends, Response, File, UploadFile, BackgroundTasks
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates 
from pydantic import BaseModel
import uuid
from supabase import Client
from utils.database import get_supabase, get_supabase_admin
from utils.enterprise_document_manager import store_enterprise_document
from utils.company_document_processor import process_and_vectorize_company_document
import os
import logging
import string
import random

router = APIRouter()

# Get the directory of the current file (enterprise_routes.py)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(project_root, "templates"))

class InterviewSetup(BaseModel):
    job_role: str
    job_description: str
    interview_type: str

class LoginData(BaseModel):
    email: str
    password: str

class SignupData(BaseModel):
    company_name: str
    email: str
    password: str
    industry: str
    size: str

class ChatMessage(BaseModel):
    message: str

@router.get("/")
async def enterprise_home(request: Request):
    company_id = request.session.get("company_id")
    if company_id:
        return RedirectResponse(url="/enterprise/dashboard")
    return templates.TemplateResponse("enterprise/home.html", {"request": request})

@router.get("/login")
async def enterprise_login_page(request: Request):
    return templates.TemplateResponse("enterprise/login.html", {"request": request})

@router.get("/signup")
async def enterprise_signup_page(request: Request):
    return templates.TemplateResponse("enterprise/signup.html", {"request": request})

@router.post("/signup")
async def signup(request: Request, signup_data: SignupData, supabase: Client = Depends(get_supabase)):
    try:
        # Check if the company already exists
        existing_company = supabase.table('companies').select('*').eq('email', signup_data.email).execute()
        if len(existing_company.data) > 0:
            raise HTTPException(status_code=400, detail="A company with this email already exists")

        # Create the user in Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": signup_data.email,
            "password": signup_data.password
        })
        
        company_id = auth_response.user.id
        
        # Create the company record in the 'companies' table
        new_company = {
            'company_id': company_id,
            'company_name': signup_data.company_name,
            'email': signup_data.email,
            'industry': signup_data.industry,
            'size': signup_data.size
        }
        supabase.table('companies').insert(new_company).execute()
        
        # Set session data
        request.session['company_id'] = company_id
        
        return JSONResponse(content={
            "message": "Signup successful",
            "redirect_url": "/enterprise/dashboard"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(request: Request, login_data: LoginData, supabase: Client = Depends(get_supabase)):
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": login_data.email,
            "password": login_data.password
        })
        
        company_id = auth_response.user.id
        token = auth_response.session.access_token
        
        company_data = supabase.table('companies').select('*').eq('company_id', company_id).execute()
        
        if len(company_data.data) == 0:
            raise HTTPException(status_code=404, detail="Company not found")
        
        company = company_data.data[0]
        
        request.session['company_id'] = company_id
        
        return JSONResponse(content={
            "message": "Login successful",
            "redirect_url": "/enterprise/dashboard",
            "token": token
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dashboard")
async def enterprise_dashboard(request: Request, supabase: Client = Depends(get_supabase)):
    company_id = request.session.get("company_id")
    if not company_id:
        return RedirectResponse(url="/enterprise/login")
    
    interviews = supabase.table('enterprise_interviews').select('*').eq('company_id', company_id).execute()
    company_data = supabase.table('companies').select('*').eq('company_id', company_id).execute()
    company = company_data.data[0] if company_data.data else None
    
    return templates.TemplateResponse("enterprise/dashboard.html", {
        "request": request,
        "interviews": interviews.data,
        "company": company
    })

@router.post("/create-interview")
async def create_interview(
    request: Request, 
    background_tasks: BackgroundTasks,
    job_role: str = Form(...), 
    job_description: str = Form(...), 
    interview_type: str = Form(...), 
    file: UploadFile = File(None), 
    supabase: Client = Depends(get_supabase),
    supabase_admin: Client = Depends(get_supabase_admin)
):
    logging.info(f"Received job_role: {job_role}")
    logging.info(f"Received job_description: {job_description}")
    logging.info(f"Received interview_type: {interview_type}")
    logging.info(f"Received file: {file.filename if file else 'No file uploaded'}")

    company_id = request.session.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    interview_id = str(uuid.uuid4())
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    document_url = None
    document_name = file.filename if file else "No document"

    if file:
        file_content = await file.read()
        document_url = await store_enterprise_document(file_content, file.filename, company_id)
        # Add the vectorization task to background tasks
        background_tasks.add_task(process_and_vectorize_company_document, file_content, document_url, "company-documents", company_id)

    interview_data = {
        "company_id": company_id,
        "document_name": document_name,
        "interview_type": interview_type,
        "position": job_role,
        "status": "active",
        "interview_id": interview_id,
        "job_description": job_description,
        "document_url": document_url,
        "password": password
    }

    logging.info(f"Interview data to be inserted: {interview_data}")

    result = supabase.table('enterprise_interviews').insert(interview_data).execute()

    if len(result.data) == 0:
        raise HTTPException(status_code=500, detail="Failed to create interview")

    return {
        "message": "Interview created successfully",
        "interview_id": interview_id,
        "password": password
    }

@router.get("/interviews")
async def get_interviews(request: Request, supabase: Client = Depends(get_supabase)):
    company_id = request.session.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    interviews = supabase.table('enterprise_interviews').select('interview_id', 'position', 'status').eq('company_id', company_id).execute()
    return JSONResponse(content={"interviews": interviews.data})

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/enterprise/login")

@router.get("/interview-results/{interview_id}")
async def get_interview_results(
    request: Request,
    interview_id: str,
    supabase: Client = Depends(get_supabase)
):
    company_id = request.session.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Fetch the interview results
    results = supabase.table("enterprise_user_interviews").select(
        "enterprise_user_interviews.id",
        "enterprise_user_interviews.start_time",
        "enterprise_user_interviews.end_time",
        "enterprise_user_interviews.status",
        "enterprise_user_interviews.interest_level",
        "enterprise_user_interviews.feedback",
        "users.email"
    ).eq("enterprise_interview_id", interview_id).join(
        "users",
        "enterprise_user_interviews.user_id",
        "users.user_id"
    ).execute()

    if not results.data:
        raise HTTPException(status_code=404, detail="No results found for this interview")

    return JSONResponse(content={"results": results.data})