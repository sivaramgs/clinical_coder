# ops.py
import re
import os
from typing import Dict, Any
from langfuse import Langfuse
from langfuse.decorators import observe
from prometheus_client import Counter

# --- TELEMETRY & OBSERVABILITY INITIALIZATION ---
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-mock"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-mock"),
    host=os.getenv("LANGFUSE_HOST", "http://clinical-langfuse-web:3000")
)

# --- PROMETHEUS METRICS ---
ROUTER_DECISIONS = Counter('llm_router_decisions_total', 'Model routing decisions', ['target_engine'])
GUARDRAIL_INTERCEPTIONS = Counter('llm_guardrail_intercepts_total', 'PII elements blocked by guardrails')
AGENT_COMPUTE_FAULTS = Counter('agent_compute_faults_total', 'Out-of-process node failures', ['node_url'])

# --- DATA PROTECTION & SECURITY ---
class LLMGuardrails:
    @staticmethod
    @observe(name="pii_scrubber")
    def scrub_pii(text: str) -> str:
        """
        Scrubs Singapore-specific identifiers to protect patient identity 
        before sending data to any local LLM or Actor node.
        """
        original_text = text

        # 1. Singapore NRIC/FIN (e.g., S1234567A, G7654321Z, T0123456H)
        text = re.sub(r'[STFGM]\d{7}[A-Z]', '[REDACTED NRIC]', text, flags=re.IGNORECASE)
        
        # 2. Singapore Phone Numbers (e.g., +65 8123 4567, 91234567)
        text = re.sub(r'(?:\+65[\s-]?)?[89]\d{3}[\s-]?\d{4}', '[REDACTED PHONE]', text)
        
        # 3. Common SG Hospital MRN formats (Often 7-8 alphanumeric chars like MRN: A1234567)
        text = re.sub(r'\b(?:MRN|UHID)[\s:]*[A-Z0-9]{6,10}\b', '[REDACTED MRN]', text, flags=re.IGNORECASE)

        if text != original_text:
            GUARDRAIL_INTERCEPTIONS.inc()
            
        return text

# --- METRIC EVALUATIONS ---
class LLMEvaluations:
    @staticmethod
    def is_valid_icd10_format(prediction: str) -> bool:
        """
        Strict structural validation for ICD-10 formats (e.g., J45.909, E11.9, A00.0)
        Prevents downstream pipeline crashes if the LLM hallucinates conversational text.
        """
        icd_pattern = re.compile(r'^[A-TV-Z][0-9][0-9A-B](\.[0-9A-TV-Z]{1,4})?$')
        return bool(icd_pattern.match(prediction.strip()))

    @staticmethod
    @observe(name="evaluate_clinical_accuracy")
    def score_clinical_accuracy(prediction: str, expected: str = None) -> float:
        if not prediction or not LLMEvaluations.is_valid_icd10_format(prediction):
            return 0.0
        if expected and prediction.strip().upper() == expected.strip().upper():
            return 1.0
        return 0.5  # Formatted correctly, but pending human/expert review

@observe(name="agentops_trace")
def log_agentops_trace(session_id: str, action: str, details: dict = None):
    print(f"[AgentOps Trace] Session: {session_id} | Action: {action} | Details: {details}")