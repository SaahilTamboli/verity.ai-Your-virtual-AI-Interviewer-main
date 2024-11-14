from .groq_client import GroqClient
import logging

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    def __init__(self):
        self.groq_client = GroqClient()

    async def initialize(self):
        # Add any initialization code if needed
        pass

    async def evaluate(self, question: str, answer: str, job_role: str, job_description: str, company_document_content: str):
        prompt = f"""
        You are an AI interview evaluator. Your task is to analyze the candidate's response and assess how it affects the interviewer's interest level.

        Job Role: {job_role}
        Job Description: {job_description}
        Company Document: {company_document_content}

        Question: {question}
        Candidate's Answer: {answer}

        Evaluate the candidate's response based on the following criteria:
        1. Relevance to the question and job requirements
        2. Depth and quality of the answer
        3. Communication skills
        4. Enthusiasm and engagement
        5. Alignment with the company document (if provided)

        Provide a brief analysis (2-3 sentences) of the response and indicate how it affects the interviewer's interest.
        Express the change in interest as a number between -15 and 15, where:
        -15 to -8: Very Negative
        -7 to -1: Negative
        0: Neutral
        1 to 7: Positive
        8 to 15: Very Positive

        Format your response as follows:
        Analysis: [Your analysis here]
        Interest Change: [Number between -15 and 15. Provide the number only, no other text.]
        """

        logger.info("Sending response evaluation prompt to Groq")
        response = await self.groq_client.get_response(prompt)
        logger.info(f"Received response from Groq: {response}")

        try:
            parts = response.split('\n')
            analysis = next((part.replace('Analysis:', '').strip() for part in parts if part.startswith('Analysis:')), '')
            interest_change_str = next((part.replace('Interest Change:', '').strip() for part in parts if part.startswith('Interest Change:')), '')
            
            logger.info(f"Parsed analysis: {analysis}")
            logger.info(f"Parsed interest change string: {interest_change_str}")

            interest_change = int(interest_change_str)
            interest_change = max(-15, min(15, interest_change))  # Ensure it's between -15 and 15
            
            logger.info(f"Final interest change: {interest_change}")
        except Exception as e:
            logger.error(f"Error parsing Groq response: {str(e)}")
            logger.warning(f"Failed to parse interest change. Using 0. Response was: {response}")
            interest_change = 0

        return {
            "analysis": analysis,
            "interest_change": interest_change,
            "question": question,
            "answer": answer
        }
