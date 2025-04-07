import pdfplumber
import pytesseract
import spacy
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load NLP model
nlp = spacy.load("en_core_web_sm")
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')  # Using GPT for question generation

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_image(image_path):
    """Extracts text from an image using OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def generate_flashcards(text):
    """Generates flashcard-style questions from text."""
    prompt = f"Generate a set of question-answer flashcards based on the following text:\n{text[:1000]}..."
    response = llm.predict(prompt)
    return response

def extract_named_entities(text):
    """Extracts named entities using spaCy."""
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

pdf_path = "C:/Users/Ayush yadav/Downloads/Adarsh_Yadav_Resume.pdf.pdf"
text = extract_text_from_pdf(pdf_path)
response = generate_flashcards(text)
print(response)