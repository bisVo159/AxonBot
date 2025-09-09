from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List

class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, 
                        description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool = Field(..., 
                    description="True if retrieved information is sufficient to answer the user's question, False otherwise.")

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int
    document: str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    enable_web_search: bool = True

class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_type: str

class AgentResponse(BaseModel):
    response: str
    trace_events: List[TraceEvent] = Field(default_factory=list)