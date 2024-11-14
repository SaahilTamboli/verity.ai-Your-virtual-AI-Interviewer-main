import logging
from .resume_evaluator import ResumeEvaluator
from .groq_client import GroqClient
from pinecone import Pinecone
import os

logger = logging.getLogger(__name__)

class InterviewInitializer:
    def __init__(self):
        self.resume_evaluator = ResumeEvaluator()
        self.groq_client = GroqClient()
        self.pinecone_client = None
        self.resume_index = None
        self.company_documents_index = None

    async def initialize(self):
        await self.resume_evaluator.initialize()
        await self.groq_client.initialize()
        
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not set in the environment variables")
        
        try:
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            self.resume_index = self.pinecone_client.Index("resume-job-description")
            self.company_documents_index = self.pinecone_client.Index("company-documents")
            logger.info("Pinecone indexes initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise

    async def setup_interview(self, user_id, job_role, job_description, resume_content, company_document_content, is_enterprise):
        analysis, self.current_interest_level = await self.resume_evaluator.evaluate(resume_content, job_role, job_description, company_document_content)
        
        interview_context = f"""
        You are an experienced professional interviewer named Verity conducting an interview for the position of {job_role}. Your goal is to assess the candidate's qualifications, experience, and fit for the role through a natural, conversational interview.

        Job Description: {job_description}

        Candidate's Resume: {resume_content}

        Initial Interest Level: {self.current_interest_level}

        Interview Guidelines:
        1. Ask one question at a time, waiting for the candidate's response before proceeding.
        2. Use information from the resume to personalize questions and validate the candidate's claims.
        3. Maintain a professional and courteous tone throughout the interview.
        4. Adapt the difficulty of questions based on the candidate's responses.
        5. The interview will have a maximum of 20 questions.
        6. Do not mention or allude to any evaluation processes or AI involvement.

        Conduct the interview by asking one question at a time. Wait for the candidate's response before asking the next question.
        Use the information from the resume to tailor your questions and validate the candidate's claims.
        Be professional and courteous throughout the interview.
        Do not evaluate or grade the candidate's responses. Your role is solely to conduct the interview.

        This is an adaptive interview:
        - The difficulty of questions may increase based on the candidate's performance.
        - The interview may end early if the candidate's performance is not meeting expectations.
        - The interview will have a maximum of 20 questions.

        Format your response as a single question, including a brief conversational prefix.
        Your responses shouldn't let the candidate know that you are an AI or that you are evaluating their responses.

        The conversational prefix should be natural and appropriate to the context. It can be:
        - Positive (e.g., "That's interesting.", "I appreciate your detailed response.")
        - Neutral (e.g., "I understand.", "Let's move on to the next topic.")
        - Probing (e.g., "Could you elaborate on that?", "Can you provide a specific example?")

        Choose the prefix based on the context of the interview and your previous question.
        """
        
        return interview_context, self.current_interest_level

    async def fetch_resume_content(self, user_id):
        try:
            resume_vector = self.resume_index.query(
                vector=[0]*384,
                filter={"user_id": user_id},
                top_k=1,
                include_metadata=True
            )

            if not resume_vector.matches:
                logger.warning(f"Resume not found in Pinecone for user_id: {user_id}")
                return ""
            else:
                resume_content = resume_vector.matches[0].metadata.get('text', '')
                logger.info(f"Resume found for user_id: {user_id}")
                return resume_content
        except Exception as e:
            logger.error(f"Error fetching resume content: {str(e)}")
            return ""

    async def fetch_company_document(self, document_url, company_id):
        try:
            company_doc_vector = self.company_documents_index.query(
                vector=[0]*384,
                filter={"document_url": document_url},
                top_k=1,
                include_metadata=True,
                namespace=company_id
            )

            if not company_doc_vector.matches:
                logger.warning(f"Company document not found in Pinecone for document_url: {document_url}")
                return ""
            else:
                company_document_content = company_doc_vector.matches[0].metadata.get('text', '')
                logger.info(f"Company document found for document_url: {document_url}")
                return company_document_content
        except Exception as e:
            logger.error(f"Error fetching company document: {str(e)}")
            return ""

    async def generate_initial_question(self, interview_context):
        prompt = f"{interview_context}\n\nAs Verity, start the interview with a warm welcome and ask an initial, open-ended question to begin the conversation naturally. You might ask the candidate to introduce themselves, share their background, or explain why they're interested in the role."
        response = await self.groq_client.get_response(prompt)
        return response
