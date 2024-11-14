import logging
from .interview_initializer import InterviewInitializer
from .interview_processor import InterviewProcessor
from .interview_evaluator import InterviewEvaluator
from .audio_processor import AudioProcessor
from .groq_client import GroqClient
from .audio_manager import AudioManager

logger = logging.getLogger(__name__)

class InterviewConductor:
    def __init__(self):
        self.initializer = InterviewInitializer()
        self.processor = InterviewProcessor()
        self.evaluator = InterviewEvaluator()
        self.audio_processor = AudioProcessor()
        self.audio_manager = AudioManager()
        self.groq_client = GroqClient()
        self.current_interest_level = 0
        self.interview_context = ""
        self.current_question = ""
        self.job_role = ""
        self.job_description = ""
        self.resume_content = ""
        self.company_document_content = ""

    async def initialize(self):
        logger.info("Initializing InterviewConductor")
        try:
            await self.initializer.initialize()
            await self.evaluator.initialize()
            await self.groq_client.initialize()
            await self.audio_processor.initialize()
            logger.info("InterviewConductor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing InterviewConductor: {str(e)}", exc_info=True)
            raise

    async def close(self):
        logger.info("Closing InterviewConductor")
        await self.audio_processor.terminate()
        # ... (close other components if necessary)

    async def start_interview(self, user_id: str, job_role: str, job_description: str, document_url: str = None, company_id: str = None, is_enterprise: bool = False):
        logger.info(f"Starting interview for user_id: {user_id}, job_role: {job_role}, is_enterprise: {is_enterprise}")
        try:
            self.job_role = job_role
            self.job_description = job_description
            
            self.resume_content = await self.initializer.fetch_resume_content(user_id)
            if not self.resume_content:
                logger.warning(f"No resume content found for user_id: {user_id}. Using a placeholder.")
                self.resume_content = "No resume content available."

            self.company_document_content = ""
            if is_enterprise and document_url:
                self.company_document_content = await self.initializer.fetch_company_document(document_url, company_id)
                if not self.company_document_content:
                    logger.warning(f"No company document content found for document_url: {document_url}. Using a placeholder.")
                    self.company_document_content = "No company document content available."

            self.interview_context, self.current_interest_level = await self.initializer.setup_interview(
                user_id, job_role, job_description, self.resume_content, self.company_document_content, is_enterprise
            )
            
            self.evaluator.set_job_details(job_role, job_description, self.company_document_content)
            self.processor.set_current_interest_level(self.current_interest_level)
            
            initial_question = await self.initializer.generate_initial_question(self.interview_context)
            self.processor.set_current_question(initial_question)
            self.current_question = initial_question
            
            logger.info("Interview started successfully")
            return await self.process_response(self.current_question)
        except Exception as e:
            logger.error(f"Error starting interview: {str(e)}", exc_info=True)
            raise

    async def process_message(self, message):
        if message == "start":
            return {"type": "text", "content": self.current_question}
        else:
            response = await self.processor.process_message(message, self.evaluator)
            self.current_interest_level = self.processor.get_current_interest_level()
            self.current_question = response["content"] if isinstance(response, dict) and "content" in response else response
            self.processor.set_current_question(self.current_question)
            return await self.process_response(self.current_question)

    async def process_candidate_response(self, transcript):
        logger.info("Processing candidate response")
        try:
            evaluation = await self.evaluator.evaluate(self.current_question, transcript)
            self.current_interest_level += evaluation['interest_change']
            self.current_interest_level = max(0, min(100, self.current_interest_level))  # Ensure it's between 0 and 100

            logger.info(f"Generating next question. Current interest level: {self.current_interest_level}")
            next_question = await self.groq_client.get_response(
                f"{self.interview_context}\n\nCandidate's response: {transcript}\n\nEvaluation: {evaluation['analysis']}\n\nCurrent interest level: {self.current_interest_level}\n\nAs Verity, provide a brief comment on the candidate's response and ask the next question."
            )
            self.current_question = next_question
            logger.info("Candidate response processed successfully")
            return next_question
        except Exception as e:
            logger.error(f"Error processing candidate response: {str(e)}", exc_info=True)
            raise

    async def process_response(self, response):
        logger.info(f"Processing response: {response}")
        try:
            audio_chunks = []
            async for chunk in self.audio_processor.process_response(response):
                audio_chunks.append(chunk)
            return {"type": "audio", "content": audio_chunks, "text": response}
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}", exc_info=True)
            raise

    async def get_groq_response(self, user_prompt):
        return await self.groq_client.get_response(self.interview_context, user_prompt)

    async def start_recording(self):
        self.audio_manager.start_recording()

    async def stop_recording(self):
        return await self.audio_manager.stop_recording()
