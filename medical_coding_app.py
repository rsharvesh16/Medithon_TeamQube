import boto3
from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import logging
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import random
import pandas as pd

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Load FAISS index for ICD codes
try:
    icd_vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded successfully.")
except FileNotFoundError:
    logger.error("FAISS index not found. Please ensure the 'faiss_index' directory exists with the necessary files.")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {str(e)}")

# Function to get LLaMA 3 model
def get_llama3_llm():
    try:
        llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 2000})
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLaMA 3 model: {str(e)}")
        return None

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
def data_ingestion(uploaded_file):
    if uploaded_file.filename.endswith('.pdf'):
        text = process_pdf(uploaded_file)
    elif uploaded_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(uploaded_file)
        text = perform_ocr(image)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image file.")

    if not text.strip():
        raise ValueError("No text could be extracted from the document. Please check your uploaded file.")

    logger.info(f"Extracted text (first 100 chars): {text[:100]}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    print("I am here")
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
    summary = chain.run(docs)
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

# Analytics Functions
def parse_llm_response(response):
    # For this example, we'll use mock data. You would parse the response from LLM here.
    age_data = pd.DataFrame({
        'Age Group': ['0-18', '19-40', '41-60', '61+'],
        'Percentage': [random.randint(5, 20), random.randint(20, 40), random.randint(30, 50), random.randint(10, 30)]
    })
    
    gender_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Percentage': [random.randint(40, 60), random.randint(40, 60)]
    })
    
    cost_data = pd.DataFrame({
        'Category': ['Medication', 'Hospital Stay', 'Outpatient Care', 'Tests'],
        'Cost': [random.randint(2000, 7000), random.randint(8000, 20000), random.randint(1000, 4000), random.randint(500, 2000)]
    })
    
    comorbidities_data = pd.DataFrame({
        'Comorbidity': ['Hypertension', 'Diabetes', 'Obesity', 'Heart Disease'],
        'Percentage': [random.randint(20, 60), random.randint(10, 40), random.randint(15, 35), random.randint(10, 25)]
    })
    
    return age_data, gender_data, cost_data, comorbidities_data

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_medical_coding', methods=['POST'])
def process_medical_coding():
    try:
        file = request.files.get('file')
        query = request.form.get('query')

        if file:
            docs = data_ingestion(file)
            print("Hey there")
            vectorstore = get_vector_store(docs, bedrock_embeddings)
            llm = get_llama3_llm()
            summary = summarize_documents(docs, llm)
            icd_codes = get_response_llm(llm, icd_vectorstore, summary)
            return jsonify({'summary': summary, 'icd_codes': icd_codes})
        elif query:
            llm = get_llama3_llm()
            icd_codes = get_response_llm(llm, icd_vectorstore, query)
            return jsonify({'icd_codes': icd_codes})
        else:
            return jsonify({'error': 'No file or query provided'}), 400
    except Exception as e:
        logger.error(f"Error in process_medical_coding: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/process_medical_query', methods=['POST'])
def process_medical_query():
    try:
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        llm = get_llama3_llm()
        icd_codes = get_response_llm(llm, icd_vectorstore, query)
        print("Hellllooooo")
        return jsonify({'icd_codes': icd_codes})
    except Exception as e:
        logger.error(f"Error in process_medical_query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_analytics', methods=['POST'])
def generate_analytics():
    try:
        data = request.json
        disease = data.get('disease')
        if not disease:
            return jsonify({'error': 'No disease name provided'}), 400

        llm = get_llama3_llm()
        analytics_prompt = f"""
        You are an AI assistant specializing in healthcare analytics for insurance companies. Based on the disease {disease}, provide the following information:

        1. Prevalence: Estimate the prevalence of the disease in the general population.
        2. Age Distribution: Describe how the disease affects different age groups.
        3. Gender Distribution: Explain any gender-specific trends for this disease.
        4. Average Treatment Cost: Estimate the average cost of treatment for this disease.
        5. Hospitalization Rate: Provide an estimate of how often this disease requires hospitalization.
        6. Comorbidities: List common comorbidities associated with this disease.
        7. Prevention Strategies: Suggest effective prevention strategies for this disease.
        8. Long-term Prognosis: Describe the long-term outlook for patients with this disease.

        For each point, provide specific numbers or percentages where applicable.
        """
        response = llm(analytics_prompt)
        age_data, gender_data, cost_data, comorbidities_data = parse_llm_response(response)

        return jsonify({
            'analysis': response,
            'age_distribution': {'type': 'bar', 'labels': age_data['Age Group'].tolist(), 'values': age_data['Percentage'].tolist()},
            'gender_distribution': {'type': 'pie', 'labels': gender_data['Gender'].tolist(), 'values': gender_data['Percentage'].tolist()},
            'cost_breakdown': {'type': 'pie', 'labels': cost_data['Category'].tolist(), 'values': cost_data['Cost'].tolist()},
            'comorbidities': {'type': 'bar', 'labels': comorbidities_data['Comorbidity'].tolist(), 'values': comorbidities_data['Percentage'].tolist()}
        })
    except Exception as e:
        logger.error(f"Error in generate_analytics: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)