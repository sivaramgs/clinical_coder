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
    target_url="http://clinical-coder-assistant:8004/a2a/execute",
    task_type="conversational",
    system_prompt="""You are an expert Clinical Assistant Node. Your goal is strict factual entity extraction.
        From the raw clinical text, extract every explicit medical element across three distinct target arrays. 

        Do not infer, diagnose, or fabricate details. Treat empty narrative details as non-existent.

        TARGET CATEGORIES:
        1. DIAGNOSES: Explicitly stated diseases, active lesions, or patient conditions confirmed by the clinician. Exclude ruled-out or negative findings.
        2. HISTORY & RISK FACTORS: Mentioned lifestyle patterns, environmental exposures, occupational hazards, or familial medical history noted as contributing context.
        3. EXTERNAL CAUSES: Physical mechanisms, environmental factors, or external vectors explicitly tied to the onset of the condition.

        """
    metadata={"capabilities": ["PII scrubbing", "Symptom extraction"], "tier": "L1_Ingestion"}
)

MEDICAL_CODER_CARD = AgentCard(
    name="Coding Specialist Node",
    target_url="http://clinical-coder-api:8004/a2a/execute",
    task_type="mapping",
    system_prompt="You are a Professional Medical Coder. Map the following extracted terms to standardized code representations using context clues.",
    metadata={"capabilities": ["Vector search matching", "MCP compliance"], "tier": "L2_Resolution"}
)