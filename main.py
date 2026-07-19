import os
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
    prompt = f"Identify and extract all specific symptoms and clinical conditions from this summary text:\n\n{clean_input}"
    res, engine = llm_router.route_query(prompt, "You are an expert Clinical Assistant.", "conversational")
    return {"sanitized_text": clean_input, "extracted_terms": res.get("text", "")}


# --- Coder Node App Routes ---
import json

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
    
    # 1. Parse out separate entities to prevent vector dilution
    try:
        extracted_entities = json.loads(raw_text)
        if isinstance(extracted_entities, dict):
            for category in ["diagnoses", "history_and_risk_factors", "external_causes"]:
                if category in extracted_entities and isinstance(extracted_entities[category], list):
                    # Clean up entries and drop empty values
                    search_queries.extend([str(item).strip() for item in extracted_entities[category] if item])
    except (json.JSONDecodeError, TypeError):
        # Fallback if Assistant returned plain unstructured text or malformed payload
        search_queries = []

    # If parsing yielded no distinct strings, fall back to the raw block
    if not search_queries:
        search_queries = [raw_text]

    # 2. Batch embedding retrieval via gateway client
    vectors = inference_gateway.generate_local_embeddings_batch(search_queries)
    
    # 3. Query vector DB independently for higher recall
    for vec in vectors:
        matches = db.search_similar_codes(query_vector=vec, limit=10)
        if matches:
            aggregated_references.extend(matches)
        
    # 4. Crash-Resilient Deduplication (Handles both strings and dictionaries)
    unique_references = []
    seen = set()
    
    for ref in aggregated_references:
        # If references are dictionaries, unique-check them by stringifying or matching a key
        if isinstance(ref, dict):
            # Fallback to key 'code' if available, otherwise serialize to string
            fingerprint = ref.get("code") or json.dumps(ref, sort_keys=True)
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_references.append(ref)
        else:
            # If references are flat strings
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

    # IMPORTANT FIX: trace creation is now wrapped in its own try/except,
    # separate from the main pipeline logic. Previously, if langfuse.trace()
    # itself failed (e.g. bad host/credentials), `trace` was never assigned
    # — but the outer except block unconditionally called trace.update(...),
    # raising UnboundLocalError and masking the real error entirely. Now a
    # Langfuse failure degrades gracefully (trace=None) instead of crashing
    # the whole request in a confusing way.
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
                timeout=90
            )
        except GetTimeoutError:
            raise HTTPException(status_code=504, detail="Assistant agent timed out.")

        # Phase 2: Vector Search via Coder Agent Node
        target_coder = f"{CODER_AGENT_URL}/a2a/execute"
        try:
            coder_res = ray.get(
                shared_agent_actor.post_task.remote(target_coder, assistant_res["extracted_terms"]),
                timeout=90
            )
        except GetTimeoutError:
            raise HTTPException(status_code=504, detail="Coder agent timed out.")

        query_terms = coder_res.get("query_terms", assistant_res["extracted_terms"])
        candidates = coder_res.get("references", [])

        # Phase 3: Final LLM mapping via SEA-LION, using the top-5 vector
        # search candidates directly as context (no reranking step).
        mapping_prompt = (
            f"You are a Professional Medical Coder executing complete multi-axial ICD-10-CM assignment.\n\n"
            f"INPUT DATA SCHEMA:\n"
            f"Extracted Entities: {query_terms}\n"
            f"Candidate Reference Codes from Vector Database: {candidates}\n\n"
            f"TASK CRITERIA:\n"
            f"1. Evaluate the clinical 'diagnoses' against candidates to assign primary/secondary disease codes .\n"
            f"2. Evaluate 'history_and_risk_factors' against candidates to assign supplementary status codes.\n"
            f"3. Evaluate 'external_causes' against candidates to assign environmental mechanism codes.\n\n"
            f"OUTPUT FORMAT:\n"
            f"Return ONLY a comma-separated list of all applicable valid codes in strict ICD-10-CM format. "
            f"Do not include explanation, introductory text, or Markdown formatting blocks."
            )

        mapping_res, engine = llm_router.route_query(
            mapping_prompt, "You are a Professional Medical Coder.", "mapping"
        )
        final_text = mapping_res.get("text", "").strip()

        # Validate the model's output against strict ICD-10-CM format.
        validated_codes = [
            c.strip() for c in final_text.split(",")
            if LLMEvaluations.is_valid_icd10_format(c.strip())
        ]

        if trace is not None:
            try:
                trace.update(
                    output=final_text,
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
        langfuse.flush()
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