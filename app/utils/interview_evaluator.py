from .resume_evaluator import ResumeEvaluator
from .response_evaluator import ResponseEvaluator
import logging
import asyncio
from groq import Groq
import os

logger = logging.getLogger(__name__)

class InterviewEvaluator:
    def __init__(self):
        self.resume_evaluator = ResumeEvaluator()
        self.response_evaluator = ResponseEvaluator()
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.job_description = ""
        self.job_role = ""
        self.company_document_content = ""

    async def initialize(self):
        await self.resume_evaluator.initialize()
        await self.response_evaluator.initialize()

    async def evaluate_resume(self, resume_content: str):
        return await self.resume_evaluator.evaluate(resume_content, self.job_role, self.job_description, self.company_document_content)

    async def evaluate(self, question: str, answer: str):
        return await self.response_evaluator.evaluate(question, answer, self.job_role, self.job_description, self.company_document_content)

    async def analyze_response(self, question: str, answer: str):
        evaluation = await self.evaluate(question, answer)
        return {
            "question": evaluation["question"],
            "user_response": evaluation["answer"],
            "analysis": evaluation["analysis"],
            "interest_change": evaluation["interest_change"]
        }

    async def get_groq_response(self, prompt):
        try:
            chat_completion = await asyncio.to_thread(
                self.groq_client.chat.completions.create,
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping to evaluate interview responses. Provide a brief, objective analysis of the candidate's answer based on relevance, depth, and alignment with the job requirements."},
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.1-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting response from Groq: {str(e)}")
            raise

    def set_job_details(self, job_role: str, job_description: str, company_document_content: str = ""):
        self.job_role = job_role
        self.job_description = job_description
        self.company_document_content = company_document_content
        logger.info(f"Job details set - Role: {job_role}, Description: {job_description}, Company Document Length: {len(company_document_content)}")
