from asyncio.log import logger
from fastapi import APIRouter, Request, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from supabase import Client
from utils.interview_conductor import InterviewConductor
from utils.interview_evaluator import InterviewEvaluator
from utils.audio_processor import AudioProcessor
from utils.audio_manager import AudioManager
from utils.resume_processor import process_and_vectorize_resume
from utils.database import get_supabase, get_supabase_admin
import os
import uuid
import logging
import json

router = APIRouter()

# Initialize necessary components
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

templates = Jinja2Templates(directory=os.path.join(project_root, "templates"))

audio_manager = AudioManager()
interview_conductor = InterviewConductor()
interview_evaluator = InterviewEvaluator()

class UserCredentials(BaseModel):
    email: str
    password: str

class InterviewData(BaseModel):
    jobRole: str
    jobDescription: str

@router.get("/signup-page")
async def signup_page(request: Request):
    return templates.TemplateResponse("user/signup.html", {"request": request})

@router.post("/signup")
async def signup(user: UserCredentials, supabase: Client = Depends(get_supabase)):
    try:
        auth_response = supabase.auth.sign_up({
            "email": user.email,
            "password": user.password
        })
        
        user_id = auth_response.user.id
        
        # Add user data to the users table
        user_data = {
            'user_id': user_id,
            'email': user.email,
            'resume_uploaded': False
        }
        supabase.table('users').insert(user_data).execute()
        
        return JSONResponse(content={"message": "Signup successful"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/login-page")
async def login_page(request: Request):
    return templates.TemplateResponse("user/login.html", {"request": request})

@router.post("/login")
async def login(request: Request, user: UserCredentials, supabase: Client = Depends(get_supabase)):
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": user.email,
            "password": user.password
        })
        user_id = auth_response.user.id
        request.session['user_id'] = user_id
        request.session['session'] = {
            "access_token": auth_response.session.access_token,
            "refresh_token": auth_response.session.refresh_token
        }
        logger.info(f"User {user_id} logged in successfully")
        return JSONResponse(content={"message": "Login successful", "redirect_url": "/user/dashboard"})
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dashboard")
async def dashboard(request: Request, supabase: Client = Depends(get_supabase)):
    user_id = request.session.get('user_id')
    if not user_id:
        return RedirectResponse(url="/user/login-page")
    
    user_data = supabase.table('users').select('*').eq('user_id', user_id).execute()
    interviews = supabase.table('interviews').select('*').eq('user_id', user_id).execute()
    
    return templates.TemplateResponse("user/dashboard.html", {
        "request": request,
        "user": user_data.data[0] if user_data.data else None,
        "interviews": interviews.data
    })

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/user/login-page")

@router.get("/new-interview")
async def new_interview_page(request: Request):
    return templates.TemplateResponse("user/new_interview.html", {"request": request})

@router.post("/start-interview")
async def start_interview(request: Request, jobRole: str = Form(...), jobDescription: str = Form(...)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    request.session['job_role'] = jobRole
    request.session['job_description'] = jobDescription
    
    return RedirectResponse(url="/user/student-interview")

@router.get("/student-interview/{interview_id}")
async def student_interview(request: Request, interview_id: str):
    user_id = request.session.get('user_id')
    if not user_id:
        return RedirectResponse(url="/user/login-page")
    
    # Fetch interview data from the database
    # You'll need to implement this part
    
    return templates.TemplateResponse("user/interview.html", {
        "request": request,
        "interview_id": interview_id,
        "job_role": "Fetched Job Role",  # Replace with actual data
        "job_description": "Fetched Job Description",  # Replace with actual data
        "user_id": user_id
    })

@router.post("/submit-interview")
async def student_submit_interview(request: Request, interview_data: InterviewData, supabase_admin: Client = Depends(get_supabase_admin)):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        interview_data_dict = {
            'user_id': user_id,
            'job_role': interview_data.jobRole,
            'job_description': interview_data.jobDescription,
            'status': 'pending',
            "interview_id": str(uuid.uuid4())
        }
        result = supabase_admin.table('interviews').insert(interview_data_dict).execute()

        if len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create interview")

        interview_id = result.data[0]['id']

        return JSONResponse(content={
            "message": "Interview created successfully",
            "interview_id": interview_id
        })
    except Exception as e:
        logging.error(f"Error in student_submit_interview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/interview-ws/{interview_id}")
async def interview_websocket(
    websocket: WebSocket, 
    interview_id: str,
    supabase: Client = Depends(get_supabase)
):
    logger.info(f"Starting interview WebSocket for interview_id: {interview_id}")
    await websocket.accept()
    conductor = InterviewConductor()
    await conductor.initialize()
    audio_manager = AudioManager()

    try:
        # Fetch interview data
        logger.info(f"Fetching interview data for interview_id: {interview_id}")
        interview_data = supabase.table('interviews').select('*').eq('interview_id', interview_id).execute()
        
        if len(interview_data.data) == 0:
            logger.warning(f"Interview not found for interview_id: {interview_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        interview = interview_data.data[0]
        user_id = interview['user_id']
        job_role = interview['job_role']
        job_description = interview['job_description']

        # Initialize conversation and question_analysis lists
        conversation = []
        question_analysis = []

        logger.info(f"Starting interview for user_id: {user_id}, job_role: {job_role}")
        initial_response = await conductor.start_interview(user_id, job_role, job_description, None, None, is_enterprise=False)
        
        # Add initial question to conversation
        conversation.append({"role": "assistant", "content": initial_response['text']})

        # Update the database with the initial question
        supabase.table("interviews").update({
            "conversation": json.dumps(conversation),
            "question_analysis": json.dumps(question_analysis),
            "status": "in_progress",
            "start_time": "now()"
        }).eq("interview_id", interview_id).execute()

        await stream_audio_response(websocket, initial_response['text'])
        
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message: {data}")
            
            if data['type'] == 'start_recording':
                logger.info("Starting audio recording")
                audio_manager.start_recording()
                await websocket.send_json({"type": "info", "content": "Recording started"})
            elif data['type'] == 'stop_recording':
                transcript = await audio_manager.stop_recording()
                logger.info(f"Candidate's response: {transcript}")
                if transcript:
                    await websocket.send_json({"type": "text", "content": f"You said: {transcript}"})
                    
                    # Add user response to conversation
                    conversation.append({"role": "user", "content": transcript})
                    
                    response = await conductor.process_candidate_response(transcript)
                    
                    # Add assistant response to conversation
                    conversation.append({"role": "assistant", "content": response})
                    
                    # Add question analysis
                    evaluation = await conductor.evaluator.evaluate(conductor.current_question, transcript)
                    question_analysis.append({
                        "question": conductor.current_question,
                        "user_response": transcript,
                        "analysis": evaluation['analysis'],
                        "interest_change": evaluation['interest_change']
                    })
                    
                    # Update the database with new conversation and question analysis
                    supabase.table("interviews").update({
                        "conversation": json.dumps(conversation),
                        "question_analysis": json.dumps(question_analysis),
                        "status": "in_progress",
                        "interest_level": conductor.current_interest_level
                    }).eq("interview_id", interview_id).execute()
                    
                    await stream_audio_response(websocket, response)
                else:
                    await websocket.send_json({"type": "error", "content": "No speech detected. Please try again."})
            elif data['type'] == 'end_interview':
                logger.info("Ending interview")
                # Update the interviews table
                logger.info(f"Updating interviews table for interview_id: {interview_id}")
                supabase.table("interviews").update({
                    "end_time": "now()",
                    "status": "completed",
                    "interest_level": conductor.current_interest_level,
                    "feedback": "Interview completed successfully",
                    "conversation": json.dumps(conversation),
                    "question_analysis": json.dumps(question_analysis)
                }).eq("interview_id", interview_id).execute()

                await websocket.send_json({"type": "info", "content": "Interview ended"})
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for interview_id: {interview_id}")
        # Update the interviews table in case of disconnection
        logger.info(f"Updating interviews table for disconnection: interview_id: {interview_id}")
        supabase.table("interviews").update({
            "end_time": "now()",
            "status": "disconnected",
            "feedback": "Interview disconnected unexpectedly",
            "conversation": json.dumps(conversation),
            "question_analysis": json.dumps(question_analysis)
        }).eq("interview_id", interview_id).execute()
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}", exc_info=True)
        await websocket.send_json({"type": "error", "content": "An unexpected error occurred"})
        # Update the interviews table in case of error
        logger.info(f"Updating interviews table for error: interview_id: {interview_id}")
        supabase.table("interviews").update({
            "end_time": "now()",
            "status": "error",
            "feedback": f"Error occurred: {str(e)}",
            "conversation": json.dumps(conversation),
            "question_analysis": json.dumps(question_analysis)
        }).eq("interview_id", interview_id).execute()
    finally:
        logger.info(f"Closing interview WebSocket for interview_id: {interview_id}")
        await conductor.close()
        await audio_manager.terminate()

async def stream_audio_response(websocket: WebSocket, response: str):
    logger.info("Starting audio response streaming")
    audio_processor = AudioProcessor()
    await audio_processor.initialize()
    try:
        logger.info("Processing audio response")
        async for audio_chunk in audio_processor.process_response(response):
            logger.debug(f"Sending audio chunk of size: {len(audio_chunk)}")
            await websocket.send_bytes(audio_chunk)
        logger.info("Audio streaming completed")
        await websocket.send_json({"type": "audio_end"})
    except Exception as e:
        logger.error(f"Error in stream_audio_response: {str(e)}", exc_info=True)
    finally:
        logger.info("Terminating audio processor")
        await audio_processor.terminate()

@router.websocket("/enterprise-interview-ws/{interview_id}")
async def enterprise_interview_websocket(
    websocket: WebSocket, 
    interview_id: str,
    supabase: Client = Depends(get_supabase)
):
    logger.info(f"Starting enterprise interview WebSocket for interview_id: {interview_id}")
    await websocket.accept()
    conductor = InterviewConductor()
    await conductor.initialize()
    audio_manager = AudioManager()

    try:
        # Get user_id from query parameters
        user_id = websocket.query_params.get("user_id")
        if not user_id:
            logger.error(f"No user_id provided for interview_id: {interview_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        # Fetch interview details
        logger.info(f"Fetching enterprise interview details for interview_id: {interview_id}")
        interview = supabase.table("enterprise_interviews").select("*").eq("interview_id", interview_id).execute()
        if not interview.data:
            logger.error(f"No interview found for interview_id: {interview_id}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        interview_data = interview.data[0]
        job_role = interview_data["position"]
        job_description = interview_data["job_description"]
        document_url = interview_data["document_url"]
        company_id = interview_data["company_id"]

        logger.info(f"Starting enterprise interview for user_id: {user_id}, job_role: {job_role}")
        initial_response = await conductor.start_interview(
            user_id, 
            job_role, 
            job_description, 
            document_url, 
            company_id, 
            is_enterprise=True
        )

        # Initialize conversation and question_analysis lists
        conversation = []
        question_analysis = []

        # Add initial question to conversation
        conversation.append({"role": "assistant", "content": initial_response['text']})

        # Update the database with the initial question
        supabase.table("enterprise_user_interviews").update({
            "conversations": json.dumps(conversation),
            "question_analysis": json.dumps(question_analysis)
        }).eq("enterprise_interview_id", interview_id).eq("user_id", user_id).execute()

        await stream_audio_response(websocket, initial_response['text'])
        
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received WebSocket message: {data}")
            
            if data['type'] == 'start_recording':
                logger.info("Starting audio recording")
                audio_manager.start_recording()
                await websocket.send_json({"type": "info", "content": "Recording started"})
            elif data['type'] == 'stop_recording':
                transcript = await audio_manager.stop_recording()
                logger.info(f"Candidate's response: {transcript}")
                if transcript:
                    await websocket.send_json({"type": "text", "content": f"You said: {transcript}"})
                    
                    # Add user response to conversation
                    conversation.append({"role": "user", "content": transcript})
                    
                    response = await conductor.process_candidate_response(transcript)
                    
                    # Add assistant response to conversation
                    conversation.append({"role": "assistant", "content": response})
                    
                    # Add question analysis
                    evaluation = await conductor.evaluator.evaluate(conductor.current_question, transcript)
                    question_analysis.append({
                        "question": conductor.current_question,
                        "user_response": transcript,
                        "analysis": evaluation['analysis'],
                        "interest_change": evaluation['interest_change']
                    })
                    
                    # Update the database with new conversation and question analysis
                    supabase.table("enterprise_user_interviews").update({
                        "conversations": json.dumps(conversation),
                        "question_analysis": json.dumps(question_analysis),
                        "interest_level": conductor.current_interest_level
                    }).eq("enterprise_interview_id", interview_id).eq("user_id", user_id).execute()
                    
                    await stream_audio_response(websocket, response)
                else:
                    await websocket.send_json({"type": "error", "content": "No speech detected. Please try again."})
            elif data['type'] == 'end_interview':
                logger.info("Ending interview")
                # Update the enterprise_user_interviews table
                logger.info(f"Updating enterprise_user_interviews table for interview_id: {interview_id}, user_id: {user_id}")
                supabase.table("enterprise_user_interviews").update({
                    "end_time": "now()",
                    "status": "completed",
                    "interest_level": conductor.current_interest_level,
                    "feedback": "Interview completed successfully",
                    "conversations": json.dumps(conversation),
                    "question_analysis": json.dumps(question_analysis)
                }).eq("enterprise_interview_id", interview_id).eq("user_id", user_id).execute()

                await websocket.send_json({"type": "info", "content": "Interview ended"})
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for interview_id: {interview_id}")
        # Update the enterprise_user_interviews table in case of disconnection
        logger.info(f"Updating enterprise_user_interviews table for disconnection: interview_id: {interview_id}, user_id: {user_id}")
        supabase.table("enterprise_user_interviews").update({
            "end_time": "now()",
            "status": "disconnected",
            "feedback": "Interview disconnected unexpectedly",
            "conversations": json.dumps(conversation),
            "question_analysis": json.dumps(question_analysis)
        }).eq("enterprise_interview_id", interview_id).eq("user_id", user_id).execute()
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}", exc_info=True)
        await websocket.send_json({"type": "error", "content": "An unexpected error occurred"})
        # Update the enterprise_user_interviews table in case of error
        logger.info(f"Updating enterprise_user_interviews table for error: interview_id: {interview_id}, user_id: {user_id}")
        supabase.table("enterprise_user_interviews").update({
            "end_time": "now()",
            "status": "error",
            "feedback": f"Error occurred: {str(e)}",
            "conversations": json.dumps(conversation),
            "question_analysis": json.dumps(question_analysis)
        }).eq("enterprise_interview_id", interview_id).eq("user_id", user_id).execute()
    finally:
        logger.info(f"Closing enterprise interview WebSocket for interview_id: {interview_id}")
        await conductor.close()
        await audio_manager.terminate()

@router.post("/upload-file")
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    supabase_admin: Client = Depends(get_supabase_admin)
):
    user_id = request.session.get("user_id")
    logger.info(f"Attempting file upload for user: {user_id}")
    if not user_id:
        logger.error("Unauthorized access attempt: No user_id in session")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        bucket_name = "resumes"
        file_path = f"{user_id}/{unique_filename}"
        file_content = await file.read()
        logger.info(f"Prepared file for upload: {file_path}")

        # Upload file to Supabase storage
        logger.info(f"Attempting to upload file to Supabase storage: {bucket_name}/{file_path}")
        response = supabase_admin.storage.from_(bucket_name).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type, "upsert": "true"}
        )

        if hasattr(response, 'error') and response.error:
            logger.error(f"Supabase storage upload error: {response.error}")
            raise Exception(response.error)

        # Get the public URL of the uploaded file
        file_url = supabase_admin.storage.from_(bucket_name).get_public_url(file_path)
        logger.info(f"File uploaded successfully. Public URL: {file_url}")

        # Update user's resume status in the database
        logger.info(f"Updating user resume status in database for user: {user_id}")
        update_response = supabase_admin.table('users').update({
            'resume_uploaded': True,
            'resume_location': file_url
        }).eq('user_id', user_id).execute()
        logger.info(f"Database update response: {update_response}")

        # Add the vectorization task to background tasks
        logger.info(f"Adding vectorization task to background for user: {user_id}")
        background_tasks.add_task(process_and_vectorize_resume, file_content, file_url, "resume-job-description", user_id)

        logger.info(f"File upload process completed successfully for user: {user_id}")
        return JSONResponse(content={
            "message": "Resume uploaded successfully and processing started",
            "filename": unique_filename,
            "file_url": file_url
        })
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-interview")
async def create_interview(request: Request, interview_data: InterviewData, supabase: Client = Depends(get_supabase)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Check if user has uploaded a resume
    user_data = supabase.table('users').select('resume_uploaded').eq('user_id', user_id).execute()
    if not user_data.data or not user_data.data[0].get('resume_uploaded'):
        raise HTTPException(status_code=400, detail="Please upload your resume before creating an interview")
    
    try:
        interview_id = uuid.uuid4()
        interview_data_dict = {
            'user_id': user_id,
            'job_role': interview_data.jobRole,
            'job_description': interview_data.jobDescription,
            'status': 'pending',
            'interview_id': str(interview_id)
        }
        result = supabase.table('interviews').insert(interview_data_dict).execute()

        if len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create interview")

        return JSONResponse(content={
            "message": "Interview created successfully",
            "interview_id": str(interview_id)
        })
    except Exception as e:
        logger.error(f"Error in create_interview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/interview/{interview_id}")
async def get_interview(request: Request, interview_id: str, supabase: Client = Depends(get_supabase)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        interview_data = supabase.table('interviews').select('*').eq('interview_id', interview_id).eq('user_id', user_id).execute()
        
        if len(interview_data.data) == 0:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        return templates.TemplateResponse("user/interview.html", {
            "request": request,
            "interview_id": interview_id,
            "job_role": interview_data.data[0]['job_role'],
            "job_description": interview_data.data[0]['job_description'],
            "user_id": user_id
        })
    except Exception as e:
        logger.error(f"Error in get_interview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/resume-status")
async def get_resume_status(request: Request, supabase: Client = Depends(get_supabase)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user_data = supabase.table('users').select('resume_uploaded').eq('user_id', user_id).execute()
    if not user_data.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    return JSONResponse(content={"resume_uploaded": user_data.data[0].get('resume_uploaded', False)})

class EnterpriseInterviewRequest(BaseModel):
    interview_code: str
    password: str

@router.post("/start-enterprise-interview")
async def start_enterprise_interview(
    request: Request,
    interview_data: EnterpriseInterviewRequest,
    supabase: Client = Depends(get_supabase)
):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    interview_id = interview_data.interview_code
    password = interview_data.password

    # Verify the interview code and password
    interview = supabase.table("enterprise_interviews").select("*").eq("interview_id", interview_id).execute()
    
    if not interview.data or interview.data[0]["password"] != password:
        raise HTTPException(status_code=400, detail="Invalid interview code or password")

    # Create a new entry in enterprise_user_interviews
    new_entry = {
        "enterprise_interview_id": interview_id,
        "user_id": str(user_id),
        "status": "in_progress"
    }
    result = supabase.table("enterprise_user_interviews").insert(new_entry).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to start interview")

    # If valid, redirect to the enterprise interview page
    return JSONResponse(content={
        "redirect_url": f"/user/enterprise-interview/{interview_id}/{user_id}"
    })

@router.get("/enterprise-interview/{interview_code}/{user_id}")
async def enterprise_interview(
    request: Request,
    interview_code: str,
    user_id: str,
    supabase: Client = Depends(get_supabase)
):
    # Fetch interview details
    interview = supabase.table("enterprise_interviews").select("*").eq("interview_id", interview_code).execute()
    
    if not interview.data:
        raise HTTPException(status_code=404, detail="Interview not found")

    interview_data = interview.data[0]

    return templates.TemplateResponse("user/interview.html", {
        "request": request,
        "interview_id": interview_code,
        "job_role": interview_data["position"],
        "job_description": interview_data["job_description"],
        "document_url": interview_data["document_url"],
        "user_id": user_id,
        "is_enterprise": True
    })

# Add any other user-related routes here

@router.on_event("startup")
async def startup_event():
    try:
        global interview_conductor
        interview_conductor = InterviewConductor()
        await interview_conductor.initialize()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise

@router.on_event("shutdown")
async def shutdown_event():
    await interview_conductor.close()
    audio_manager.terminate()