import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import fitz  # PyMuPDF
import logging
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock
from langchain_aws import BedrockEmbeddings
from flask import Flask, request, jsonify, render_template
import warnings
import io

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Initialize BedrockEmbeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# OCR Function
def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        if not text.strip():
            logger.warning("OCR produced empty result")
        return text
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        return ""

# PDF Processing Function
def process_pdf(file):
    text = ""
    try:
        # Extract text using PyMuPDF
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        
        # Perform OCR on all pages
        file.seek(0)  # Reset file pointer
        pdf_bytes = file.read()
        images = convert_from_bytes(pdf_bytes)
        ocr_text = ""
        for image in images:
            ocr_text += perform_ocr(image) + "\n"
        
        # Combine PyMuPDF text and OCR text
        combined_text = text + "\n" + ocr_text
        
        if not combined_text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        
        return combined_text
    
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise

# Data Ingestion Implementation
def data_ingestion(file):
    if file.filename.endswith('.pdf'):
        text = process_pdf(file)
    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(file)
        text = perform_ocr(image)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image file.")

    if not text.strip():
        raise ValueError("No text could be extracted from the document. Please check your uploaded file.")

    logger.info(f"Extracted text (first 100 chars): {text[:100]}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    return [Document(page_content=t) for t in docs]

# Vector Embedding and Vector Store Implementation
def get_vector_store(docs, bedrock_embeddings):
    try:
        embeddings = bedrock_embeddings.embed_documents([doc.page_content for doc in docs])
        if not embeddings:
            raise ValueError("No embeddings were generated. Please check your documents and embedding model.")
        
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        return vectorstore_faiss
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

# Summarization function
def summarize_documents(docs, llm):
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)
    return summary

prompt_template = """
Human: You are a medical coding assistant. Your task is to analyze the given medical text and provide the following information:

1. Under the subheading "CODINGS":
   Create a table with the following columns:
   - Provider Name (Mentioned Doctor)
   - Disease Name
   - ICD-10 Code
   - Chronicity Status (Acute/Chronic)
   
   Include all diseases mentioned in the text.

2. After the table, indicate whether the report satisfies the MEAT criteria in medical coding. (Monitoring, Evaluating, Assessing/Addressing, and Treating)

3. Under the subheading "DISEASE INFORMATION":
   For each disease mentioned:
   - Provide a brief explanation of the disease
   - Describe how it can be treated or managed

Use the following context to inform your response:
<context>
{context}
</context>

Question: {question}
Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

@app.route('/process_medical_coding', methods=['POST'])
def api_process_medical_coding():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file:
            result = process_medical_coding(file, icd_vectorstore, get_llama3_llm, bedrock_embeddings)
            return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in process_medical_coding: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
def process_medical_query(query, icd_vectorstore, get_llama3_llm):
    try:
        llm = get_llama3_llm()
        icd_codes = get_response_llm(llm, icd_vectorstore, query)
        
        return {
            "icd_codes": icd_codes
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {"error": str(e)}

def get_llama3_llm():
    try:
        llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 2000})
        return llm
    except Exception as e:
        print(f"Error initializing LLaMA 3 model: {str(e)}")
        return None

# Load FAISS index for ICD codes
try:
    icd_vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading FAISS index: {str(e)}")
    icd_vectorstore = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_medical_coding', methods=['POST'])
def api_process_medical_coding():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    result = process_medical_coding(file, icd_vectorstore, get_llama3_llm, bedrock_embeddings)
    return jsonify(result)

@app.route('/process_medical_query', methods=['POST'])
def api_process_medical_query():
    data = request.json
    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    result = process_medical_query(data['query'], icd_vectorstore, get_llama3_llm)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)