# agent_cards.py
from pydantic import BaseModel
from typing import Dict, Any

class AgentCard(BaseModel):
    name: str
    target_url: str
    task_type: str
    system_prompt: str
    metadata: Dict[str, Any]

# Definitive Specifications for Cluster Deployment Infrastructure
CLINICAL_ASSISTANT_CARD = AgentCard(
    name="Clinical Assistant Node",
    target_url="http://127.0.0.1:8003/a2a/execute",
    task_type="conversational",
    system_prompt="You are an expert Clinical Assistant. Identify and extract all specific symptoms and clinical conditions.",
    metadata={"capabilities": ["PII scrubbing", "Symptom extraction"], "tier": "L1_Ingestion"}
)

MEDICAL_CODER_CARD = AgentCard(
    name="Coding Specialist Node",
    target_url="http://127.0.0.1:8004/a2a/execute",
    task_type="mapping",
    system_prompt="You are a Professional Medical Coder. Map the following extracted terms to standardized code representations using context clues.",
    metadata={"capabilities": ["Vector search matching", "MCP compliance"], "tier": "L2_Resolution"}
)