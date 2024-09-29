import streamlit as st
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
import logging
from medical_coding import process_medical_coding
from analytics import process_analytics

# Ensure set_page_config is the very first Streamlit call
st.set_page_config(page_title="Medical Assistant", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Function to get LLaMA 3 model
def get_llama3_llm():
    try:
        llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 2000})
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLaMA 3 model: {str(e)}")
        st.error(f"Unable to load LLaMA 3 model: {str(e)}")
        return None

def main():
    # HTML/CSS for styling
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Aileron&display=swap');
            .container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                max-width: 800px;
                margin: 0 auto;
            }
            .title {
                text-align: center;
                font-family: 'Aileron', sans-serif;
                color: #333;
            }
            .sidebar-title {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 10px;
            }
            .footer {
                text-align: center;
                margin-top: 20px;
                font-size: 14px;
                color: #888;
            }
        </style>
        <div class="container">
            <h2 class="title">Medical Assistant</h2>
        </div>
    """, unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    # Attempt to load FAISS index for ICD codes
    try:
        icd_vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")
    except FileNotFoundError:
        st.error("FAISS index not found. Please ensure the 'faiss_index' directory exists with the necessary files.")
        return
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        logger.error(f"Failed to load FAISS index: {str(e)}")
        return

    # Sidebar for operation selection
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Select Operation:</div>', unsafe_allow_html=True)
        operation = st.selectbox("Choose operation", ["Medical Coding", "Analytics"])

    # Operations: Medical Coding and Analytics
    if operation == "Medical Coding":
        if get_llama3_llm() is not None:
            process_medical_coding(st, icd_vectorstore, get_llama3_llm, bedrock_embeddings)
    elif operation == "Analytics":
        if get_llama3_llm() is not None:
            process_analytics(st, get_llama3_llm)

    # Footer
    st.markdown('<div class="footer">Made with Love by Team Cube ♥︎</div>', unsafe_allow_html=True)

# Entry point for the application
if __name__ == "__main__":
    main()
