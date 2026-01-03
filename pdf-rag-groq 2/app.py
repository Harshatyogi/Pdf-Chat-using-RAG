import streamlit as st
from groq import Groq
import PyPDF2
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "")
# Page configuration
st.set_page_config(
    page_title="PDF Q&A with Groq",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text() + "\n"
        return text, len(pdf_reader.pages)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None, 0

def get_answer(question, context, api_key, model):
    """Get answer from Groq API"""
    try:
        client = Groq(api_key=api_key)
        
        # Truncate context if too long
        if len(context) > 15000:
            context = context[:15000] + "\n\n[Document truncated...]"
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided document. Be specific and cite relevant parts. If the answer is not in the document, say so."
                },
                {
                    "role": "user",
                    "content": f"""Document Content:

{context}

Question: {question}

Answer based on the document above:"""
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Main UI
st.title("⚡ PDF Q&A System with Groq")
st.markdown("Upload a PDF and ask questions - powered by Groq's lightning-fast AI!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Paste your Groq API key here"
    )
    
    if api_key:
        st.success("✓ API Key entered!")
    else:
        st.info("👆 Enter your Groq API key to start")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0
    )
    
    st.markdown("---")
    
    # PDF Upload
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf']
    )
    
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("📖 Processing PDF..."):
                pdf_text, num_pages = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    st.session_state.pdf_text = pdf_text
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"✓ Processed {num_pages} pages!")
                    st.info(f"📝 {len(pdf_text):,} characters extracted")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main content
if not api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar")
    st.markdown("""
    ### 🚀 Getting Started:
    1. Get free API key from: **https://console.groq.com**
    2. Paste it in the sidebar
    3. Upload a PDF
    4. Start asking questions!
    """)
    
elif not st.session_state.pdf_text:
    st.info("👈 Upload a PDF document in the sidebar to begin")
    st.markdown("""
    ### 💡 What you can do:
    - Ask questions about the document
    - Get summaries and key points
    - Extract specific information
    - Understand complex topics
    """)
    
else:
    # Show current document
    st.success(f"📄 **Current Document:** {st.session_state.pdf_name}")
    
    # Chat history
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**❓ Question {i+1}:** {q}")
            st.markdown(f"**💡 Answer:** {a}")
            st.divider()
    
    # Question input
    st.subheader("💬 Ask a Question")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Type your question:",
            placeholder="e.g., What is the main topic of this document?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    if ask_button and question:
        with st.spinner("🤔 Thinking..."):
            answer = get_answer(
                question, 
                st.session_state.pdf_text, 
                api_key, 
                model
            )
            st.session_state.chat_history.append((question, answer))
            st.rerun()
    
    # Document preview
    with st.expander("📖 View Full Document"):
        st.text_area(
            "Document Content",
            st.session_state.pdf_text,
            height=400,
            label_visibility="collapsed"
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "⚡ Powered by Groq AI | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)