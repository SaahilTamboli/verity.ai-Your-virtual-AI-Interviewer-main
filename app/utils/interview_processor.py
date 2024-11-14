import logging
from .groq_client import GroqClient
from .interview_evaluator import InterviewEvaluator

logger = logging.getLogger(__name__)

class InterviewProcessor:
    def __init__(self):
        self.current_interest_level = 75  # Default interest level
        self.question_count = 0
        self.difficulty_level = "normal"
        self.current_question = ""  # Initialize current_question

    def set_current_interest_level(self, interest_level: int):
        self.current_interest_level = interest_level
        logger.info(f"Current interest level set to: {self.current_interest_level}")

    def get_current_interest_level(self):
        return self.current_interest_level

    def set_current_question(self, question: str):
        self.current_question = question
        logger.info(f"Current question set to: {self.current_question}")

    async def process_message(self, message: str, evaluator: InterviewEvaluator):
        self.question_count += 1
        
        analysis, interest_change = await evaluator.evaluate_response(self.current_question, message)
        self.current_interest_level += interest_change
        self.current_interest_level = max(0, min(100, self.current_interest_level))

        logger.info(f"Updated Interest Level: {self.current_interest_level}")

        if self.should_end_interview():
            return {"type": "text", "content": "Thank you for your time. We've covered all the necessary topics for this interview."}

        self.adjust_difficulty()

        next_question = await self.generate_next_question(evaluator)
        self.set_current_question(next_question)

        return {"type": "text", "content": next_question}

    def should_end_interview(self):
        if self.current_interest_level <= 0:
            logger.info("Ending interview due to zero or negative interest level")
            return True
        if self.question_count == 7 and self.current_interest_level < 50:
            logger.info("Ending interview at question 7 due to low interest level")
            return True
        if self.question_count == 14 and self.current_interest_level < 50:
            logger.info("Ending interview at question 14 due to low interest level")
            return True
        if self.question_count >= 20:
            logger.info("Ending interview after reaching maximum questions")
            return True
        return False

    def adjust_difficulty(self):
        if self.question_count == 7 and self.current_interest_level >= 50:
            self.difficulty_level = "slightly harder"
            logger.info("Adjusting difficulty to slightly harder")
        elif self.question_count == 14 and self.current_interest_level >= 50:
            self.difficulty_level = "challenging"
            logger.info("Adjusting difficulty to challenging")

    async def generate_next_question(self, evaluator: InterviewEvaluator):
        prompt = f"""
        As an experienced interviewer for the {evaluator.job_role} position, generate the next relevant question based on the job requirements and previous discussions. The question should be at a {self.difficulty_level} difficulty level. Ensure the question is natural, conversational, and doesn't reveal any backend evaluation processes. Start with a brief, context-appropriate comment or transition phrase before asking the question.

        Job Description: {evaluator.job_description}
        Current Question Count: {self.question_count}
        """

        return await evaluator.get_groq_response(prompt)

    async def process_candidate_response(self, transcript, evaluator):
        return await self.process_message(transcript, evaluator)

    async def get_groq_response(self, prompt):
        return await self.groq_client.get_response(prompt)
