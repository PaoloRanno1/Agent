# -*- coding: utf-8 -*-
import streamlit as st
import os
from typing import Dict, List, Optional, Type
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import MessagesPlaceholder
from pydantic import BaseModel, Field
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Strada Legal PDF Agent",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if 'pdf_manager' not in st.session_state:
    st.session_state.pdf_manager = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}

# PDF Manager Class
class PDFManager:
    """Simple PDF manager to handle uploads and track files"""
    
    def __init__(self):
        self.uploaded_pdfs = {}
    
    def upload_pdf(self, file_path: str, file_name: str = None) -> str:
        """Upload and register a PDF file"""
        if not os.path.exists(file_path):
            return f"❌ Error: File not found at {file_path}"
        
        if not file_path.lower().endswith('.pdf'):
            return f"❌ Error: File must be a PDF"
        
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        self.uploaded_pdfs[file_name] = {
            'path': file_path,
            'upload_time': None
        }
        
        return f"✅ PDF '{file_name}' uploaded successfully!"
    
    def get_pdf_path(self, file_name: str = None) -> str:
        """Get the path of an uploaded PDF"""
        if not self.uploaded_pdfs:
            return None
        
        if file_name is None:
            file_name = list(self.uploaded_pdfs.keys())[-1]
        
        return self.uploaded_pdfs.get(file_name, {}).get('path')
    
    def list_pdfs(self) -> List[str]:
        """List all uploaded PDF names"""
        return list(self.uploaded_pdfs.keys())

# Summarization function
def summarize_legal_pdf(pdf_path: str, openai_api_key: str) -> str:
    """Summarize legal PDF with enhanced prompts"""
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)
    
    enhanced_legal_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior legal analyst specializing in private equity transactions and M&A documentation.
        Your expertise includes share purchase agreements, investment agreements, shareholder agreements, and related PE documentation.
        
        When summarizing legal document sections, focus on:
        1. KEY COMMERCIAL TERMS: Purchase price, valuation, payment structure, earn-outs
        2. LEGAL PROVISIONS: Representations & warranties, indemnities, covenants, conditions precedent
        3. GOVERNANCE & CONTROL: Board composition, voting rights, information rights, consent requirements
        4. FINANCIAL TERMS: Working capital adjustments, debt arrangements, escrow provisions
        5. RISK FACTORS: Material adverse change clauses, liability caps, survival periods
        6. TIMELINE & CONDITIONS: Closing conditions, regulatory approvals, key dates and deadlines
        7. EXIT PROVISIONS: Tag-along, drag-along, pre-emption rights, liquidity preferences
        
        Structure your summary with clear headings and bullet points for easy reference.
        Highlight any unusual, non-standard, or particularly favorable/unfavorable terms.
        Always specify page ranges being summarized."""),
        
        ("user", """Analyze and summarize the following section from a private equity legal document:
        
DOCUMENT SECTION: Pages {start_page} to {end_page}

{text}

Provide a structured summary covering the key areas mentioned in your instructions. If certain categories don't apply to this section, focus on what's most relevant. Highlight any critical terms, deadlines, or unusual provisions.""")
    ])
    
    summarization_chain = enhanced_legal_prompt | llm | StrOutputParser()
    
    def chunk_pdf_by_pages(pdf_path: str, pages_per_chunk: int = 10) -> List[Dict]:
        """Chunk PDF by pages"""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        chunks = []
        for i in range(0, len(pages), pages_per_chunk):
            chunk_pages = pages[i:i + pages_per_chunk]
            combined_text = ""
            for page in chunk_pages:
                combined_text += page.page_content + "\n\n"
            
            start_page = chunk_pages[0].metadata['page'] + 1
            end_page = chunk_pages[-1].metadata['page'] + 1
            
            chunk = {
                'text': combined_text.strip(),
                'start_page': start_page,
                'end_page': end_page,
                'chunk_id': len(chunks) + 1,
                'page_count': len(chunk_pages)
            }
            chunks.append(chunk)
        
        return chunks
    
    chunks = chunk_pdf_by_pages(pdf_path=pdf_path)
    
    summaries = []
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        summary = summarization_chain.invoke({
            "text": chunk['text'],
            "start_page": chunk['start_page'],
            "end_page": chunk['end_page']
        })
        summaries.append({
            "chunk_id": chunk['chunk_id'],
            "start_page": chunk['start_page'],
            "end_page": chunk['end_page'],
            "summary": summary
        })
        progress_bar.progress((i + 1) / len(chunks))
    
    progress_bar.empty()
    
    Final_summary = ""
    for summary_data in summaries:
        Final_summary += f"Chunk {summary_data['chunk_id']} (Pages {summary_data['start_page']}-{summary_data['end_page']}):\n"
        Final_summary += summary_data['summary'] + "\n\n"
    
    return Final_summary

# Tool definitions
class PDFUploadInput(BaseModel):
    """Input schema for PDF upload tool"""
    file_path: str = Field(description="Path to the PDF file to upload")
    file_name: Optional[str] = Field(default=None, description="Optional custom name for the PDF file")

class PDFSummaryInput(BaseModel):
    """Input schema for PDF summary tool"""
    file_name: Optional[str] = Field(default=None, description="Name of the PDF file to summarize")

class ListPDFsInput(BaseModel):
    """Input schema for listing PDFs"""
    pass

class PDFUploadTool(BaseTool):
    name: str = "upload_pdf"
    description: str = "Upload a PDF file to the system"
    args_schema: Type[BaseModel] = PDFUploadInput
    
    def __init__(self, pdf_manager: PDFManager, **kwargs):
        super().__init__(**kwargs)
        self._pdf_manager = pdf_manager
    
    def _run(self, file_path: str, file_name: Optional[str] = None) -> str:
        return self._pdf_manager.upload_pdf(file_path, file_name)
    
    async def _arun(self, file_path: str, file_name: Optional[str] = None) -> str:
        raise NotImplementedError("upload_pdf does not support async yet")

class PDFSummaryTool(BaseTool):
    name: str = "summarize_pdf"
    description: str = "Create a detailed legal summary of an uploaded PDF"
    args_schema: Type[BaseModel] = PDFSummaryInput
    
    def __init__(self, openai_api_key: str, pdf_manager: PDFManager, **kwargs):
        super().__init__(**kwargs)
        self._openai_api_key = openai_api_key
        self._pdf_manager = pdf_manager
    
    def _run(self, file_name: Optional[str] = None) -> str:
        pdf_path = self._pdf_manager.get_pdf_path(file_name)
        
        if pdf_path is None:
            if not self._pdf_manager.list_pdfs():
                return "❌ No PDF files have been uploaded yet."
            else:
                return f"❌ PDF file '{file_name}' not found."
        
        try:
            summary = summarize_legal_pdf(pdf_path, self._openai_api_key)
            return f"✅ Summary completed!\n\n{summary}"
        except Exception as e:
            return f"❌ Error creating summary: {str(e)}"
    
    async def _arun(self, file_name: Optional[str] = None) -> str:
        raise NotImplementedError("summarize_pdf does not support async yet")

class ListPDFsTool(BaseTool):
    name: str = "list_pdfs"
    description: str = "List all uploaded PDF files"
    args_schema: Type[BaseModel] = ListPDFsInput
    
    def __init__(self, pdf_manager: PDFManager, **kwargs):
        super().__init__(**kwargs)
        self._pdf_manager = pdf_manager
    
    def _run(self) -> str:
        pdfs = self._pdf_manager.list_pdfs()
        if not pdfs:
            return "No PDF files have been uploaded yet."
        return f"📄 Uploaded PDFs: {', '.join(pdfs)}"
    
    async def _arun(self) -> str:
        raise NotImplementedError("list_pdfs does not support async yet")

def create_pdf_tools(openai_api_key: str, pdf_manager: PDFManager):
    """Create all PDF-related tools"""
    return [
        PDFUploadTool(pdf_manager=pdf_manager),
        PDFSummaryTool(openai_api_key=openai_api_key, pdf_manager=pdf_manager),
        ListPDFsTool(pdf_manager=pdf_manager)
    ]

def create_pdf_agent(openai_api_key: str, pdf_manager: PDFManager):
    """Create the complete PDF processing agent"""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
    tools = create_pdf_tools(openai_api_key, pdf_manager)
    
    system_prompt = """You are a helpful AI assistant specialized in processing legal PDF documents.
    You have access to tools for uploading PDFs, creating detailed legal summaries, and managing uploaded files.
    
    Your capabilities:
    - Upload PDF files to the system for processing
    - Create comprehensive legal summaries focused on private equity and M&A documents
    - List and manage uploaded PDF files
    - Provide guidance on document analysis
    
    Be helpful and guide users through the process."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# Streamlit UI

def main():
    st.title("⚖️ Strada Legal PDF Agent")
    st.markdown("Upload and analyze legal documents with AI-powered insights")
    
    # Add password check
    #if not check_password():
     #   st.stop()
    
    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key to use the agent")
        
        if api_key:
            if st.session_state.pdf_manager is None:
                st.session_state.pdf_manager = PDFManager()
            if st.session_state.agent is None:
                st.session_state.agent = create_pdf_agent(api_key, st.session_state.pdf_manager)
                st.success("Agent initialized!")
        
        st.divider()
        
        # File upload
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None and api_key:
            if st.button("Process PDF"):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Upload to PDF manager
                result = st.session_state.pdf_manager.upload_pdf(tmp_path, uploaded_file.name)
                st.session_state.uploaded_files[uploaded_file.name] = tmp_path
                st.success(result)
        
        # List uploaded files
        if st.session_state.pdf_manager:
            st.divider()
            st.header("Uploaded Files")
            pdfs = st.session_state.pdf_manager.list_pdfs()
            if pdfs:
                for pdf in pdfs:
                    st.write(f"📄 {pdf}")
            else:
                st.write("No files uploaded yet")
    
    # Main chat interface
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to start")
        return
    
    # Chat interface
    st.header("Chat with Legal Agent")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your legal documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    agent_response = response["output"]
                    st.markdown(agent_response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Quick actions
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("List Files"):
            response = st.session_state.agent.invoke({"input": "What PDF files are currently uploaded?"})
            st.info(response["output"])
    
    with col2:
        if st.button("Summarize Latest"):
            response = st.session_state.agent.invoke({"input": "Please provide a detailed legal summary of the most recent PDF file"})
            st.info(response["output"])
    
    with col3:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
