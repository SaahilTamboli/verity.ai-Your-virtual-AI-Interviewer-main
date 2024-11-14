# verity.ai-Your-virtual-AI-Interviewer

![alt text](https://github.com/VedantDeshmukh1/verity.ai-Your-virtual-AI-Interviewer/blob/main/app/mermaid-diagram-2024-08-29-234916.svg)

The Virtual Interviewer is an innovative project that combines the power of large language models (LLMs) and audio transcription to create an interactive and intelligent interviewing system. This project aims to streamline the interview process by generating relevant questions based on a candidate's resume and evaluating the candidate's responses using advanced natural language processing techniques.

## Features

- Automatic generation of interview questions based on the candidate's resume
- Audio recording and transcription of the candidate's responses
- Real-time evaluation of the candidate's answers using LLMs
- Interactive and user-friendly interface
- Integration with Google's Gemini language model for accurate question generation and response evaluation

## Technologies Used

- Python
- LangChain: A framework for building applications with LLMs
- AssemblyAI: Audio transcription API
- Google Generative AI: Language model for question generation and response evaluation
- PyPDF: Library for loading and processing PDF documents
- FFmpeg: Library for handling audio and video files

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/virtual-interviewer.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the necessary API keys:
   - Obtain an API key from AssemblyAI and update the `aai.settings.api_key` variable in the code.
   - Obtain a Google API key and provide it as the `api_key` variable in the code.

4. Prepare the candidate's resume:
   - Save the candidate's resume as a PDF file.
   - Update the `pdf_path` variable in the code with the path to the resume file.

5. Run the Virtual Interviewer:
   ```
   python virtual_interviewer.py
   ```

## Usage

1. The Virtual Interviewer will load the candidate's resume and generate relevant interview questions based on the content.

2. The candidate will be prompted to answer each question by speaking into the microphone.

3. The audio responses will be recorded, transcribed, and evaluated in real-time using the LLMs.

4. If a response is deemed unsatisfactory or irrelevant, the candidate will be asked to provide a more relevant answer.

5. The interview will continue until all generated questions have been answered satisfactorily.

## Future Enhancements

- Integration with video recording for a more immersive interview experience
- Sentiment analysis of the candidate's responses
- Customizable question generation based on specific job requirements
- Multi-language support for global recruitment

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We would like to express our gratitude to the developers and contributors of the various libraries and frameworks used in this project. Their hard work and dedication have made this project possible.

## Contact

For any inquiries or feedback, please contact us at [email protected]

---

Feel free to customize and expand upon this README file based on your specific project details and requirements.
