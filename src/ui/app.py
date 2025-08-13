import streamlit as st
import requests
import json
import os
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="RAG Corporate Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

def call_rag_api(question: str) -> dict:
    """Call the RAG API with a question."""
    try:
        response = requests.post(
            f"{RAG_API_URL}/ask",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def format_sources(sources: list) -> str:
    """Format sources for display."""
    if not sources:
        return "No sources found."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Unknown")
        page = source.get("page")
        section = source.get("section", 0)
        snippet = source.get("snippet", "")[:200] + "..." if len(source.get("snippet", "")) > 200 else source.get("snippet", "")
        
        page_info = f" (Page {page})" if page else ""
        formatted.append(f"**{i}. {title}{page_info}**\n{snippet}")
    
    return "\n\n".join(formatted)

# Main UI
st.title("🤖 RAG Corporate Assistant")
st.markdown("Ask questions about your internal documents and get answers with citations.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Status Check
    try:
        health_response = requests.get(f"{RAG_API_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Unavailable")
    
    st.markdown("---")
    
    # Settings
    st.subheader("Settings")
    show_sources = st.checkbox("Show Sources", value=True)
    show_metadata = st.checkbox("Show Metadata", value=False)
    
    st.markdown("---")
    
    # Instructions
    st.subheader("💡 Tips")
    st.markdown("""
    - Ask specific questions about your documents
    - Use keywords from your document titles
    - Check the sources to verify information
    - Try different phrasings if you don't get good results
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Question")
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        placeholder="What does the document say about...?",
        height=100
    )
    
    # Submit button
    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("Searching documents and generating answer..."):
                result = call_rag_api(question.strip())
                
                if result:
                    # Store in session state for persistence
                    st.session_state.last_result = result
                    st.session_state.last_question = question.strip()
        else:
            st.warning("Please enter a question.")

# Display results
if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
    st.markdown("---")
    
    # Answer section
    st.subheader("📝 Answer")
    answer = st.session_state.last_result.get("answer", "No answer provided.")
    st.markdown(answer)
    
    # Sources section
    if show_sources:
        sources = st.session_state.last_result.get("sources", [])
        if sources:
            st.subheader("📚 Sources")
            
            # Create tabs for each source
            if len(sources) > 1:
                source_tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])
                for i, (tab, source) in enumerate(zip(source_tabs, sources)):
                    with tab:
                        title = source.get("title", "Unknown Document")
                        page = source.get("page")
                        section = source.get("section", 0)
                        snippet = source.get("snippet", "No snippet available.")
                        
                        st.markdown(f"**Document:** {title}")
                        if page:
                            st.markdown(f"**Page:** {page}")
                        st.markdown(f"**Section:** {section}")
                        st.markdown("**Relevant Text:**")
                        st.text_area("", snippet, height=150, key=f"snippet_{i}", disabled=True)
            else:
                # Single source
                source = sources[0]
                title = source.get("title", "Unknown Document")
                page = source.get("page")
                section = source.get("section", 0)
                snippet = source.get("snippet", "No snippet available.")
                
                st.markdown(f"**Document:** {title}")
                if page:
                    st.markdown(f"**Page:** {page}")
                st.markdown(f"**Section:** {section}")
                st.markdown("**Relevant Text:**")
                st.text_area("", snippet, height=150, disabled=True)
        else:
            st.info("No sources found for this answer.")
    
    # Metadata section
    if show_metadata:
        st.subheader("🔍 Metadata")
        with st.expander("Raw Response"):
            st.json(st.session_state.last_result)

with col2:
    st.subheader("📊 Quick Stats")
    
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        sources = st.session_state.last_result.get("sources", [])
        
        # Stats
        st.metric("Sources Found", len(sources))
        
        if sources:
            unique_docs = len(set(s.get("title", "Unknown") for s in sources))
            st.metric("Unique Documents", unique_docs)
            
            pages = [s.get("page") for s in sources if s.get("page")]
            if pages:
                st.metric("Page Range", f"{min(pages)}-{max(pages)}")
    
    # Recent questions (if you want to add history)
    st.subheader("🕒 Recent")
    if hasattr(st.session_state, 'last_question'):
        st.text_area("Last Question:", st.session_state.last_question, height=60, disabled=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        RAG Corporate Assistant | Built with Streamlit & FastAPI
    </div>
    """,
    unsafe_allow_html=True
)