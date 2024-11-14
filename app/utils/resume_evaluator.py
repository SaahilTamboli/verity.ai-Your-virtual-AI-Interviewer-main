from .groq_client import GroqClient
import logging

logger = logging.getLogger(__name__)

class ResumeEvaluator:
    def __init__(self):
        self.groq_client = GroqClient()

    async def initialize(self):
        # Add any initialization code if needed
        pass

    async def evaluate(self, resume_content: str, job_role: str, job_description: str, company_document_content: str):
        prompt = f"""
        You are an AI resume evaluator. Your task is to analyze the candidate's resume and determine an initial interest level based on how well it matches the job requirements.

        Job Role: {job_role}
        Job Description: {job_description}
        Company Document: {company_document_content}

        Resume Content: {resume_content}

        Evaluate the resume based on the following criteria:
        1. Relevance of skills and experience to the job requirements
        2. Educational background
        3. Project experience
        4. Overall impression
        5. Alignment with the company document (if provided)

        Provide a brief analysis (2-3 sentences) of the resume and determine an initial interest level.
        Express the initial interest level as a number between 60 and 100, where:
        60-70: Low interest
        71-80: Moderate interest
        81-90: High interest
        91-100: Very high interest

        Format your response as follows:
        Analysis: [Your analysis here]
        Initial Interest: [Number between 60 and 100]
        """

        logger.info("Sending resume evaluation prompt to Groq")
        response = await self.groq_client.get_response(prompt)
        logger.info(f"Received response from Groq: {response}")

        try:
            parts = response.split('\n')
            analysis = next((part.replace('Analysis:', '').strip() for part in parts if part.startswith('Analysis:')), '')
            interest_level_str = next((part.replace('Initial Interest:', '').strip() for part in parts if part.startswith('Initial Interest:')), '')
            
            logger.info(f"Parsed analysis: {analysis}")
            logger.info(f"Parsed interest level string: {interest_level_str}")

            if interest_level_str:
                interest_level = int(interest_level_str)
                interest_level = max(60, min(100, interest_level))  # Ensure it's between 60 and 100
            else:
                logger.warning("Failed to parse initial interest level. Using default value of 75.")
                interest_level = 75
            
            logger.info(f"Final interest level: {interest_level}")
        except Exception as e:
            logger.error(f"Error parsing Groq response: {str(e)}")
            logger.warning(f"Failed to parse initial interest level. Using default value of 75. Response was: {response}")
            interest_level = 75
            analysis = "Unable to analyze the resume due to an error."

        return analysis, interest_level
