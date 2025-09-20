import requests
import json


def upload_document_to_backend(fastapi_base_url: str, uploaded_file):
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    
    response = requests.post(f"{fastapi_base_url}/upload-document", files=files)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    
    return response.json()

def chat_with_backend_agent(fastapi_base_url: str, session_id: str, query: str, enable_web_search: bool):
    payload = {
        "session_id": session_id,
        "query": query,
        "enable_web_search": enable_web_search
    }

    response = requests.post(f"{fastapi_base_url}/execute", json=payload, stream=False)
    response.raise_for_status() 

    data = response.json()
    agent_response = data.get("response", "Sorry, I couldn't get a response from the agent.")
    trace_events = data.get("trace_events", [])
    
    return agent_response, trace_events