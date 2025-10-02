import streamlit as st
from backend_api import upload_document_to_backend
from session_manager import init_session_state

def display_header():
    """Renders the main title and introductory markdown."""
    st.set_page_config(page_title="AxonBot", layout="wide") 
    st.title("🤖 AxonBot: AI Agent RAG Chatbot")
    st.markdown("Ask me anything! I can answer questions using my internal knowledge (RAG) or by searching the web.")
    st.markdown("---")

def render_document_upload_section(fastapi_base_url: str):

    with st.sidebar:
        st.header("Upload Document to Knowledge Base")
        with st.expander("Upload New Document (PDF Only)"):
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
            
            if st.button("Upload PDF", key="upload_pdf_button"):
                if uploaded_file is not None:
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        try:
                            upload_data = upload_document_to_backend(fastapi_base_url, uploaded_file)
                            st.success(f"PDF '{upload_data.get('filename')}' uploaded successfully! Processed {upload_data.get('processed_chunks')} pages.")
                        except Exception as e:
                            st.error(f"An error occurred during upload: {e}")
                else:
                    st.warning("Please upload a PDF file before clicking 'Upload PDF'.")
        st.markdown("---")

def render_agent_settings_section():
    with st.sidebar:
        st.header("Agent Settings")
        st.session_state.web_search_enabled = st.checkbox(
            "Enable Web Search (🌐)", 
            value=st.session_state.web_search_enabled,
            help="If enabled, the agent can use web search when its knowledge base is insufficient. If disabled, it will only use uploaded documents."
        )
        # st.markdown("---")

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_trace_events(trace_events: list):
    if trace_events:
        with st.expander("🔬 Agent Workflow Trace"):
            for event in trace_events:
                icon_map = {
                    'router': "➡️",
                    'rag_lookup': "📚",
                    'web_search': "🌐",
                    'answer': "💡",
                    '__end__': "✅"
                }
                icon = icon_map.get(event['node_name'], "⚙️")
                
                st.subheader(f"{icon} Step {event['step']}: {event['node_name']}")
                st.write(f"**Description:** {event['description']}")
                
                if event['node_name'] == 'rag_lookup' and 'sufficiency_verdict' in event['details']:
                    verdict = event['details']['sufficiency_verdict']
                    if verdict == "Sufficient":
                        st.success(f"**RAG Verdict:** {verdict} - Relevant info found in Knowledge Base.")
                    else:
                        st.warning(f"**RAG Verdict:** {verdict} - No sufficient info in Knowledge Base. Diverting to Web Search.")
                    
                    if 'retrieved_content_summary' in event['details']:
                        st.markdown(f"**Retrieved Content Summary:** `{event['details']['retrieved_content_summary']}`")
                elif event['node_name'] == 'web_search' and 'retrieved_content_summary' in event['details']:
                    st.markdown(f"**Web Search Content Summary:** `{event['details']['retrieved_content_summary']}`")
                elif event['node_name'] == 'router' and 'router_override_reason' in event['details']:
                    st.info(f"**Router Override:** {event['details']['router_override_reason']}")
                    st.json({"initial_decision": event['details']['initial_decision'], "final_decision": event['details']['final_decision']})
                elif event['details']:
                    st.json(event['details'])
                
                st.markdown("---")