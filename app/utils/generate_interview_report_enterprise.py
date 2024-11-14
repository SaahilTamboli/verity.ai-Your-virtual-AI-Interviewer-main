import os
from supabase import create_client, Client
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import json
import logging
import dotenv
import matplotlib.pyplot as plt
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from io import BytesIO

dotenv.load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(temperature=0.1, groq_api_key=groq_api_key)

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """add page info to each page (page x of y)"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 8)
        self.drawRightString(letter[0] - 0.5*inch, 0.5*inch, 
            f"Page {self._pageNumber} of {page_count}")

def fetch_interview_data(interview_id: str, user_id: str):
    # Fetch data from enterprise_user_interviews
    user_interview = supabase.table("enterprise_user_interviews").select("*").eq("enterprise_interview_id", interview_id).eq("user_id", user_id).execute()
    
    if not user_interview.data:
        raise ValueError(f"No interview data found for interview_id: {interview_id} and user_id: {user_id}")
    
    user_interview_data = user_interview.data[0]
    
    # Fetch data from enterprise_interviews
    enterprise_interview = supabase.table("enterprise_interviews").select("*").eq("interview_id", interview_id).execute()
    
    if not enterprise_interview.data:
        raise ValueError(f"No enterprise interview data found for interview_id: {interview_id}")
    
    enterprise_interview_data = enterprise_interview.data[0]
    
    # Combine the data
    combined_data = {**user_interview_data, **enterprise_interview_data}
    return combined_data

def analyze_interview(interview_data):
    conversations = interview_data['conversations']
    question_analysis = interview_data['question_analysis']
    
    prompt = PromptTemplate(
        input_variables=["company_id", "document_name", "interview_type", "position", "job_description", "conversations", "question_analysis"],
        template="""
        As an AI interview analyst, your task is to provide a comprehensive evaluation of a candidate's interview performance.

        Company ID: {company_id}
        Document Name: {document_name}
        Interview Type: {interview_type}
        Position: {position}
        Job Description: {job_description}

        Interview Conversation:
        {conversations}

        Question-by-Question Analysis:
        {question_analysis}

        Please provide a detailed analysis covering the following aspects:
        1. Overall impression of the candidate
        2. Strengths demonstrated during the interview
        3. Areas for improvement
        4. Alignment with the job role and requirements
        5. Communication skills
        6. Technical competence (if applicable)
        7. Cultural fit
        8. Recommendation (Fit / Unfit) with justification

        Your analysis should be thorough, balanced, and provide specific examples from the interview to support your points.
        Consider the interview type and any specific requirements mentioned in the job description.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.run(
        company_id=interview_data['company_id'],
        document_name=interview_data['document_name'],
        interview_type=interview_data['interview_type'],
        position=interview_data['position'],
        job_description=interview_data['job_description'],
        conversations=json.dumps(conversations, indent=2),
        question_analysis=json.dumps(question_analysis, indent=2)
    )
    
    return response

def generate_pdf_report(interview_data, analysis):
    pdf_filename = f"interview_report_{interview_data['interview_id']}_{interview_data['user_id']}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    # Modify existing styles
    styles['Title'].fontSize = 18
    styles['Title'].textColor = colors.darkblue
    styles['Title'].spaceAfter = 12

    styles['Heading1'].fontSize = 16
    styles['Heading1'].textColor = colors.darkblue
    styles['Heading1'].spaceAfter = 10

    styles['Heading2'].fontSize = 14
    styles['Heading2'].textColor = colors.darkgreen
    styles['Heading2'].spaceBefore = 12
    styles['Heading2'].spaceAfter = 6

    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].textColor = colors.black

    # Add a new style for highlights
    styles.add(ParagraphStyle(name='Highlight', 
                              parent=styles['BodyText'],
                              textColor=colors.red))

    story = []

    # Header
    def header(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica-Bold', 12)
        canvas.drawString(inch, letter[1] - 0.5*inch, "Confidential Interview Report")
        canvas.restoreState()

    # Footer
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.drawString(inch, 0.5*inch, f"Page {doc.page} of {doc.pageCount()}")
        canvas.restoreState()

    # Title
    story.append(Paragraph("Interview Feedback Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Company and Interview Details
    story.append(Paragraph("Company and Interview Details", styles['Heading1']))
    details = [
        ["Company ID", interview_data['company_id']],
        ["Document Name", interview_data['document_name']],
        ["Interview Type", interview_data['interview_type']],
        ["Position", interview_data['position']],
    ]
    t = Table(details, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (0,-1), colors.darkblue),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (1,1), (-1,-1), colors.beige),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Job Description
    story.append(Paragraph("Job Description", styles['Heading2']))
    story.append(Paragraph(interview_data['job_description'], styles['BodyText']))
    story.append(Spacer(1, 12))

    # Candidate Details
    story.append(Paragraph("Candidate Details", styles['Heading2']))
    story.append(Paragraph(f"Candidate ID: {interview_data['user_id']}", styles['BodyText']))
    story.append(Paragraph(f"Interview Date: {interview_data['start_time']}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Analysis
    story.append(Paragraph("Interview Analysis", styles['Heading2']))
    for paragraph in analysis.split('\n\n'):
        story.append(Paragraph(paragraph, styles['BodyText']))
        story.append(Spacer(1, 6))

    # Interest Level
    story.append(Paragraph("Interest Level", styles['Heading2']))
    story.append(Paragraph(f"Final Interest Level: {interview_data['interest_level']}/100", styles['Highlight']))
    story.append(Spacer(1, 12))

    # Interest Level Chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Interest Level'], [interview_data['interest_level']], color='skyblue')
    plt.title('Candidate Interest Level')
    plt.ylim(0, 100)
    plt.ylabel('Interest Level')
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    story.append(Image(img_buffer, width=4*inch, height=3*inch))
    plt.close()

    story.append(PageBreak())

    # Question-by-Question Analysis
    story.append(Paragraph("Question-by-Question Analysis", styles['Heading2']))
    question_analysis = json.loads(interview_data['question_analysis']) if isinstance(interview_data['question_analysis'], str) else interview_data['question_analysis']
    
    # Prepare data for chart
    questions = []
    interest_changes = []
    
    for qa in question_analysis:
        story.append(Paragraph(f"Question: {qa['question']}", styles['Heading2']))
        story.append(Paragraph(f"Answer: {qa['user_response']}", styles['BodyText']))
        story.append(Paragraph(f"Analysis: {qa['analysis']}", styles['BodyText']))
        story.append(Paragraph(f"Interest Change: {qa['interest_change']}", styles['Highlight']))
        story.append(Spacer(1, 6))
        
        questions.append(f"Q{len(questions)+1}")
        interest_changes.append(float(qa['interest_change']))

    # Interest Change Chart
    plt.figure(figsize=(8, 4))
    plt.bar(questions, interest_changes, color='lightgreen')
    plt.title('Interest Change by Question')
    plt.xlabel('Questions')
    plt.ylabel('Interest Change')
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    story.append(Image(img_buffer, width=6*inch, height=3*inch))
    plt.close()

    doc.build(story, onFirstPage=header, onLaterPages=header, canvasmaker=NumberedCanvas)
    logger.info(f"PDF report generated: {pdf_filename}")
    return pdf_filename

def main(interview_id: str, user_id: str):
    try:
        interview_data = fetch_interview_data(interview_id, user_id)
        analysis = analyze_interview(interview_data)
        pdf_filename = generate_pdf_report(interview_data, analysis)
        logger.info(f"Interview report generated successfully: {pdf_filename}")
    except Exception as e:
        logger.error(f"Error generating interview report: {str(e)}")
        raise  # Re-raise the exception to see the full traceback

if __name__ == "__main__":
    interview_id = input("Enter the interview ID: ")
    user_id = input("Enter the user ID: ")
    main(interview_id, user_id)
