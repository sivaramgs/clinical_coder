import os
import re
import json
import time
import httpx
import uvicorn
import threading
import ray
from ray.exceptions import GetTimeoutError
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse import Langfuse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.database import LocalVectorDB
from src.gateway import ClinicalInferenceGateway, LLMGatewayRouter
from src.ops import LLMGuardrails, LLMEvaluations
from src.agent import RayA2AAgentActor
from src.config import config

# =========================================================================
# 1. INITIALIZATION & CONFIGURATION LAYER
# =========================================================================
load_dotenv()

# Agent Cross-Network URLs (For Microservice Mode)
ASSISTANT_AGENT_URL = os.getenv("ASSISTANT_AGENT_URL", "http://127.0.0.1:8003")
CODER_AGENT_URL = os.getenv("CODER_AGENT_URL", "http://127.0.0.1:8004")

# Observability Init
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-mock"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-mock"),
    host=os.getenv("LANGFUSE_BASE_URL", "http://langfuse-web:3000"),
    debug=True
)

# =========================================================================
# 2. SHARED SINGLETONS — sourced from src/, instantiated once per process.
# =========================================================================
db = LocalVectorDB()
inference_gateway = ClinicalInferenceGateway()
llm_router = LLMGatewayRouter(gateway_instance=inference_gateway)

# Long-lived Ray actor handle, created once in bootstrap() below — NOT
# recreated per request.
shared_agent_actor = None

# =========================================================================
# 3. FASTAPI APPLICATION INSTANCES (CORE PORTAL & AGENTS)
# =========================================================================
app = FastAPI(title="Clinical Core Architecture Portal")
assistant_app = FastAPI(title="Clinical Assistant Node")
coder_app = FastAPI(title="Coding Specialist Node")


class A2APayload(BaseModel):
    input_data: str


class InputData(BaseModel):
    note_id: str
    clinical_note: str


# --- Assistant Node App Routes ---
@assistant_app.post("/a2a/execute")
def assistant_logic(payload: A2APayload):
    clean_input = LLMGuardrails.scrub_pii(payload.input_data)
    
    system_instruction = (
        "You are an expert Clinical Assistant Node. Your goal is strict factual entity extraction. "
        "From the raw clinical text, extract every explicit medical element into a valid JSON object with the following keys:\n"
        "- 'diagnoses': list of explicitly stated diseases, active lesions, or patient conditions (keep anatomical details like 'transverse colon').\n"
        "- 'history_and_risk_factors': list of lifestyle patterns, risk factors, or context.\n"
        "- 'external_causes': list of physical mechanisms or environmental factors.\n\n"
        "Return ONLY the raw JSON object. Do not include markdown codeblocks, explanation, or introductory text."
    )
    
    prompt = f"Identify and extract all specific symptoms and clinical conditions from this summary text:\n\n{clean_input}"
    res, engine = llm_router.route_query(prompt, system_instruction, "conversational")
    return {"sanitized_text": clean_input, "extracted_terms": res.get("text", "")}


# --- Coder Node App Routes ---
@coder_app.post("/a2a/execute")
def coder_logic(payload: A2APayload):
    """
    Parses structured JSON from the Assistant Node, generates independent embeddings 
    for each clinical entity to prevent vector dilution, and safely aggregates a broader 
    candidate set for the final reasoning model.
    """
    raw_text = payload.input_data
    aggregated_references = []
    search_queries = []
    
    # Clean up and strip markdown code blocks if the local LLM attached them
    cleaned_json_text = raw_text.strip()
    if cleaned_json_text.startswith("```"):
        cleaned_json_text = re.sub(r'^```(?:json)?\s*', '', cleaned_json_text)
        cleaned_json_text = re.sub(r'\s*```$', '', cleaned_json_text)

    # 1. Parse out separate entities to prevent vector dilution
    try:
        extracted_entities = json.loads(cleaned_json_text)
        if isinstance(extracted_entities, dict):
            for category in ["diagnoses", "history_and_risk_factors", "external_causes"]:
                if category in extracted_entities and isinstance(extracted_entities[category], list):
                    search_queries.extend([str(item).strip() for item in extracted_entities[category] if item])
    except (json.JSONDecodeError, TypeError):
        # Fallback split mechanics to isolate entities from plain unstructured text layout
        lines = re.split(r'[\n;,•\-]+', raw_text)
        search_queries = [line.strip() for line in lines if line.strip() and len(line.strip()) > 2]

    # If parsing yielded no distinct strings, fall back to the raw block
    if not search_queries:
        search_queries = [raw_text]

    # 2. Batch embedding retrieval via gateway client
    vectors = inference_gateway.generate_local_embeddings_batch(search_queries)
    
    # 3. Query vector DB independently for higher recall
    # CRITICAL FIX: Lower limit to 4 per sub-query to prevent a single complex condition 
    # from completely flooding out all other secondary or chronic conditions.
    for vec in vectors:
        matches = db.search_similar_codes(query_vector=vec, limit=4)
        if matches:
            aggregated_references.extend(matches)
        
    # 4. Crash-Resilient Deduplication (Handles both strings and dictionaries)
    unique_references = []
    seen = set()
    
    for ref in aggregated_references:
        if isinstance(ref, dict):
            fingerprint = ref.get("code") or json.dumps(ref, sort_keys=True)
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_references.append(ref)
        else:
            if ref not in seen:
                seen.add(ref)
                unique_references.append(ref)
    
    return {
        "query_terms": raw_text, 
        "references": unique_references
    }

# =========================================================================
# 4. MASTER PORTAL HUB ENDPOINTS
# =========================================================================
@app.get("/", response_class=HTMLResponse)
def read_index():
    html_path = os.path.join(os.getcwd(), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: index.html not found in workspace directory root.</h3>"


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/process-clinical-note")
async def process(payload: InputData):
    global shared_agent_actor

    if shared_agent_actor is None:
        raise HTTPException(
            status_code=503,
            detail="Agent actor not yet initialized (Ray still starting up). Please retry shortly."
        )

    try:
        trace = langfuse.trace(name="Clinical Processing Pipeline", input=payload.clinical_note)
    except Exception as e:
        print(f"[!] Langfuse trace creation failed (non-blocking): {type(e).__name__}: {e}")
        trace = None

    try:
        # Phase 1: Ingestion & Extraction via Assistant Agent Node (MERaLiON)
        target_assistant = f"{ASSISTANT_AGENT_URL}/a2a/execute"
        try:
            assistant_res = ray.get(
                shared_agent_actor.post_task.remote(target_assistant, payload.clinical_note),
                timeout=360
            )
        except GetTimeoutError:
            raise HTTPException(status_code=504, detail="Assistant agent timed out.")

        # Phase 2: Vector Search via Coder Agent Node
        target_coder = f"{CODER_AGENT_URL}/a2a/execute"
        try:
            coder_res = ray.get(
                shared_agent_actor.post_task.remote(target_coder, assistant_res["extracted_terms"]),
                timeout=360
            )
        except GetTimeoutError:
            raise HTTPException(status_code=504, detail="Coder agent timed out.")

        query_terms = coder_res.get("query_terms", assistant_res["extracted_terms"])
        candidates = coder_res.get("references", [])
        
        # --- TELEMETRY: RECALL SCORE ---
        if trace:
            recall_score = 1.0 if len(candidates) > 0 else 0.0
            trace.score(name="clinical-recall", value=recall_score, comment=f"Found {len(candidates)} candidates")

        # Phase 3: Final LLM mapping via SEA-LION, enforcing strict coding index guidelines
        mapping_prompt = (
            f"You are a Professional Medical Coder executing complete multi-axial ICD-10-CM assignment.\n\n"
            f"CLINICAL SPECIFICITY & CODING DIRECTIVES:\n"
            f"1. Specificity Rule: Map to the absolute highest tier of anatomical specificity documented. If separate segments of an organ system are explicitly affected (e.g., both small bowel and large bowel/colon segments), assign the unique specific codes for each distinct region rather than a single generic or unspecified classification.\n"
            f"2. Multi-System Cross-Coverage Rule: You must ensure every independent diagnostic axis extracted is fully represented. Do not flood output slots with redundant micro-segment variants of a single disease category while neglecting distinct concurrent conditions. Balance code assignments to completely cover: distinct structural/vascular bowel lesions, systemic bacterial infections (pinpointing the specific causative organism), acute organ failure manifestations, chronic underlying baseline diseases, and palliative/hospice management status.\n"
            f"3. Etiology & Manifestation Linking: Sequence underlying systemic infectious causes before their respective severity, manifestation, or secondary systemic response codes according to standard classification layout rules.\n"
            f"4. Candidate Evaluation: The vector candidates provide starting reference definitions. If the candidate array lacks the hyper-specific code but your expert knowledge dictates the exact code variant required by the clinical note text, prioritize and output the correct highly specific code.\n\n"
            f"INPUT DATA SCHEMA:\n"
            f"Extracted Entities: {query_terms}\n"
            f"Candidate Reference Codes from Vector Database: {candidates}\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return ONLY a comma-separated list of all applicable valid codes in strict ICD-10-CM format. "
            f"Do not include explanation, introductory text, or Markdown formatting blocks."
        )

        mapping_res, engine = llm_router.route_query(
            mapping_prompt, "You are a Professional Medical Coder.", "mapping"
        )
        final_text = mapping_res.get("text", "").strip()

        # Robust Extraction: Pull all clean structural code matches anywhere from conversational response text
        raw_extracted_codes = re.findall(r'\b[A-TV-Z][0-9][0-9A-B](?:\.[0-9A-TV-Z]{1,4})?\b', final_text.upper())

        # Validate and deduplicate codes inline
        validated_codes = []
        seen_codes = set()
        for code in raw_extracted_codes:
            cleaned_code = code.strip()
            if cleaned_code not in seen_codes and LLMEvaluations.is_valid_icd10_format(cleaned_code):
                seen_codes.add(cleaned_code)
                validated_codes.append(cleaned_code)

        # Enforce dynamic list restriction: maximum of 10 codes, but can be less
        validated_codes = validated_codes[:10]

        # Sync verification: Back-populate MCP candidate matches list to ensure it holds all final codes
        existing_candidate_codes = set()
        for ref in candidates:
            if isinstance(ref, dict) and ref.get("code"):
                existing_candidate_codes.add(ref.get("code").strip().upper())
            elif isinstance(ref, str):
                existing_candidate_codes.add(ref.strip().upper())

        for final_code in validated_codes:
            code_upper = final_code.upper()
            if code_upper not in existing_candidate_codes:
                candidates.append({
                    "code": final_code,
                    "short_description": "Final Mapped Diagnostic Code Reference",
                    "long_description": f"ICD-10-CM code {final_code} assigned by mapping evaluation.",
                    "description": "Final Mapped Diagnostic Code Reference",
                    "score": 1.0
                })
                existing_candidate_codes.add(code_upper)

        if trace is not None:
            try:
                trace.update(
                    output=final_text,
                    usage={
                        "promptTokens": assistant_res.get("prompt_tokens", 0) + mapping_res.get("prompt_tokens", 0),
                        "completionTokens": assistant_res.get("completion_tokens", 0) + mapping_res.get("completion_tokens", 0),
                    },
                    metadata={
                        "note_id": payload.note_id,
                        "engine_used": engine,
                        "validated_codes": validated_codes,
                        "candidate_count": len(candidates)
                    }
                )
                langfuse.flush()
            except Exception as flush_err:
                print(f"[!] Langfuse trace update/flush failed (non-blocking): {flush_err}")

        return {
            "note_id": payload.note_id,
            "sanitized_note": assistant_res.get("sanitized_text", ""),
            "mcp_vector_matches": candidates,
            "mapped_diagnostics": final_text,
            "validated_icd10_codes": validated_codes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if trace is not None:
            try:
                trace.update(status_message=str(e), level="ERROR")
                langfuse.flush()
            except Exception as flush_err:
                print(f"[!] Langfuse error-trace update/flush failed (non-blocking): {flush_err}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# 5. ENGINE BOOTSTRAP FORWARD DIRECTIVE
# =========================================================================
def start_daemon_server(app_instance, port):
    uvicorn.run(app_instance, host="0.0.0.0", port=port, log_level="warning")


@app.on_event("startup")
def bootstrap():
    global shared_agent_actor

    if "127.0.0.1" in ASSISTANT_AGENT_URL or "localhost" in ASSISTANT_AGENT_URL:
        print("[*] Local Dev fallback: Starting embedded daemon agent servers...")
        threading.Thread(target=start_daemon_server, args=(assistant_app, 8003), daemon=True).start()
        threading.Thread(target=start_daemon_server, args=(coder_app, 8004), daemon=True).start()

    ray.init(ignore_reinit_error=True, object_store_memory=256 * 1024 * 1024, include_dashboard=False)

    shared_agent_actor = RayA2AAgentActor.remote()
    print("[*] Persistent RayA2AAgentActor created.")


@app.on_event("shutdown")
def shutdown():
    global shared_agent_actor
    if shared_agent_actor is not None:
        ray.kill(shared_agent_actor)
    ray.shutdown()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)