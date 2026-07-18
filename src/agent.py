import ray
import httpx
from src.ops import  LLMGuardrails


@ray.remote
class RayA2AAgentActor:
    """
    Ray remote actor executing out-of-process A2A pipeline tasks.

    NOTE: this previously also loaded a sentence-transformers CrossEncoder
    reranker in __init__ (cross-encoder/ms-marco-MiniLM-L-6-v2). That model
    load requires downloading weights from Hugging Face Hub and pulls in
    torch — expensive, and a likely cause of the actor's worker process
    dying outright on first use. Removed per request: keep this pipeline
    to the sovereign local Ollama models (MERaLiON / SEA-LION) only, no
    external model downloads.
    """

    def __init__(self, metadata: dict = None):
        self.metadata = metadata or {}
        # Persistent client prevents repeated connection/allocation overhead
        self.client = httpx.Client(timeout=120.0)
        print(f"[*] Initialized Distributed Agent Actor instance with metadata: {self.metadata}")

    def post_task(self, target_url: str, text_payload: str) -> dict:
        """Sanitizes text and safely dispatches payload to external microservice endpoints."""
        sanitized_payload = LLMGuardrails.scrub_pii(text_payload)

        try:
            res = self.client.post(target_url, json={"input_data": sanitized_payload})
            res.raise_for_status()
            return res.json()
        except Exception as e:
            from src.ops import AGENT_COMPUTE_FAULTS
            AGENT_COMPUTE_FAULTS.labels(node_url=target_url).inc()
            raise RuntimeError(f"Ray compute fault calling node [{target_url}]: {str(e)}")

    def __del__(self):
        self.client.close()