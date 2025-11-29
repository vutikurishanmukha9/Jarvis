import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import hashlib
import logging
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="PDF Chatbot")

st.header("PDF Question Answering Chatbot")

# Helper function to get file hash
def get_file_hash(file):
    """Generate MD5 hash of uploaded file for change detection."""
    return hashlib.md5(file.getvalue()).hexdigest()

# Sidebar for API key and file upload
with st.sidebar:
    st.title("PDF Summarizer")
    
    # API Key Management (Hybrid Approach)
    api_key = None
    
    # Try to get API key from secrets.toml first
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.success("API key loaded from secrets")
            logger.info("API key loaded from secrets.toml")
    except FileNotFoundError:
        pass
    
    # Fallback to environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.info("API key loaded from environment")
            logger.info("API key loaded from environment variable")
    
    # Final fallback to UI input
    if not api_key:
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        if api_key:
            logger.info("API key entered via UI")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    
    # Model Configuration
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        help="GPT-4 provides better accuracy but costs more"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make output more creative, lower values more focused"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum length of the response"
    )
    
    st.markdown("---")
    
    # File uploader
    file = st.file_uploader("Upload your PDF document", type="pdf")
    
    st.markdown("---")
    
    # Advanced Settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=150,
            step=50,
            help="Overlap between chunks for context continuity"
        )
        
        num_chunks = st.slider(
            "Number of Relevant Chunks",
            min_value=2,
            max_value=10,
            value=4,
            help="How many text chunks to use for answering"
        )
    
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Configure your API key")
    st.markdown("2. Upload a PDF file")
    st.markdown("3. Ask questions about the content")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None

# Check if API key is provided
if not api_key:
    st.warning("Please configure your OpenAI API key to continue.")
    st.info("""
    **Three ways to configure your API key:**
    1. Create `.streamlit/secrets.toml` with `OPENAI_API_KEY = "your-key"`
    2. Set environment variable `OPENAI_API_KEY`
    3. Enter it in the sidebar above
    """)
    logger.warning("No API key configured")
    st.stop()

# Process PDF file
if file is not None:
    try:
        # Calculate file hash for change detection
        current_file_hash = get_file_hash(file)
        
        # Only process if it's a new file (based on hash)
        if st.session_state.processed_file_hash != current_file_hash:
            with st.spinner("Reading and processing PDF..."):
                logger.info(f"Processing new PDF: {file.name}")
                
                # Extract text from PDF
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                if not text.strip():
                    error_msg = "No text could be extracted from the PDF. The file might be image-based or corrupted."
                    st.error(f"{error_msg}")
                    logger.error(f"Failed to extract text from {file.name}")
                    st.stop()
                
                # Split text into chunks with configurable parameters
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", " ", ""],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                logger.info(f"Split text into {len(chunks)} chunks")
                
                # Generate embeddings and create vector store
                try:
                    embeddings = OpenAIEmbeddings()
                    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
                    st.session_state.processed_file_hash = current_file_hash
                    st.session_state.processed_file_name = file.name
                    
                    st.success(f"PDF processed successfully! Found {len(chunks)} text chunks.")
                    logger.info(f"Successfully processed {file.name}")
                except Exception as e:
                    st.error(f"Error creating embeddings: {str(e)}")
                    logger.error(f"Embedding error: {str(e)}", exc_info=True)
                    st.info("This might be an API key issue. Please verify your OpenAI API key.")
                    st.stop()
        else:
            st.info(f"Using cached version of: {st.session_state.processed_file_name}")
        
        # Question input
        st.markdown("---")
        user_question = st.text_input(
            "Ask a question about your PDF:",
            placeholder="E.g., What is the main topic of this document?"
        ).strip()
        
        if user_question:
            # Input validation
            if len(user_question) < 5:
                st.warning("Please enter a more detailed question (at least 5 characters).")
            elif len(user_question) > 500:
                st.warning("Question is too long. Please keep it under 500 characters.")
            elif st.session_state.vector_store is None:
                st.error("Please upload and process a PDF first.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        logger.info(f"Processing question: {user_question[:100]}...")
                        
                        # Find relevant chunks
                        matches = st.session_state.vector_store.similarity_search(
                            user_question,
                            k=num_chunks
                        )
                        logger.info(f"Found {len(matches)} relevant chunks")
                        
                        # Initialize LLM with configured parameters
                        llm = ChatOpenAI(
                            temperature=temperature,
                            max_tokens=max_tokens,
                            model_name=model_name
                        )
                        
                        # Create prompt template for better context
                        prompt = ChatPromptTemplate.from_template(
                            """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
                        )
                        
                        # Generate answer using modern LangChain API
                        chain = create_stuff_documents_chain(llm, prompt)
                        response = chain.invoke({
                            "context": matches,
                            "question": user_question
                        })
                        
                        logger.info("Successfully generated answer")
                        
                        # Display response
                        st.markdown("### Answer:")
                        st.write(response)
                        
                        # Show source chunks (optional)
                        with st.expander("View source text chunks"):
                            for i, doc in enumerate(matches, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.text(doc.page_content[:300] + "...")
                                st.markdown("---")
                    
                    except ImportError as e:
                        error_msg = "Missing required dependencies. Please run: pip install -r requirements.txt"
                        st.error(f"{error_msg}")
                        logger.error(f"Import error: {str(e)}", exc_info=True)
                    
                    except Exception as e:
                        error_str = str(e).lower()
                        
                        # Provide specific error messages based on error type
                        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
                            st.error("Invalid API key. Please check your OpenAI API key.")
                            logger.error("Authentication error", exc_info=True)
                        elif "rate limit" in error_str or "429" in error_str:
                            st.error("Rate limit exceeded. Please try again in a few moments.")
                            logger.error("Rate limit error", exc_info=True)
                        elif "quota" in error_str or "insufficient" in error_str:
                            st.error("API quota exceeded. Please check your OpenAI account billing.")
                            logger.error("Quota error", exc_info=True)
                        else:
                            st.error(f"An unexpected error occurred: {str(e)}")
                            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                        
                        st.info("Try refreshing the page or re-uploading your PDF.")
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.info("Please make sure the PDF is valid and not corrupted.")

else:
    st.info("Please upload a PDF file to get started!")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, and OpenAI*")