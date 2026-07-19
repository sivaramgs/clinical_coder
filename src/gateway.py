# gateway.py
import os
import httpx
from typing import Dict, Any, Tuple
from src.ops import ROUTER_DECISIONS
from src.config import config


class ClinicalInferenceGateway:
    """
    Handles local embeddings and dynamic inference calls via the authenticated nginx proxy.
    """
    def __init__(self):
        self.ollama_host = os.getenv("OLLAMA_HOST", config.OLLAMA_HOST)
        self.api_key = os.getenv("OLLAMA_API_KEY", config.OLLAMA_API_KEY)

        # Timeout extended to 300.0s to allow local Ollama instances ample time to warm up
        self.gateway_client = httpx.Client(
            base_url=self.ollama_host,
            timeout=300.0,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

        self.embed_model_name = config.EMBEDDING_MODEL_NAME
        self.reasoning_model_name = config.MODEL_SEA_LION
        self.instruct_model_name = config.MODEL_MERALION

    def generate_local_embedding(self, text: str) -> list[float]:
        """
        Executes a singular embedding call against the `/api/embed` endpoint.
        Fixes missing dimension routing required by the Coder Node.
        """
        payload = {
            "model": self.embed_model_name,
            "input": text
        }
        try:
            response = self.gateway_client.post("/api/embed", json=payload, timeout=300.0)
            response.raise_for_status()

            vectors = response.json().get("embeddings", [])
            if vectors and len(vectors) > 0:
                return vectors[0]
            return [0.0] * config.VECTOR_DIMENSION
        except Exception as e:
            print(f"[!] Local Single Embedding Error via {self.embed_model_name}: {str(e)}")
            return [0.0] * config.VECTOR_DIMENSION

    def generate_local_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Batched counterpart for ingest pipelines."""
        payload = {
            "model": self.embed_model_name,
            "input": texts
        }
        try:
            response = self.gateway_client.post("/api/embed", json=payload, timeout=300.0)
            response.raise_for_status()
            vectors = response.json().get("embeddings", [])

            if len(vectors) != len(texts):
                raise ValueError(f"Batch mismatch: sent {len(texts)}, got {len(vectors)}")
            return vectors
        except Exception as e:
            print(f"[!] Local Batch Embedding Error via {self.embed_model_name}: {str(e)}")
            return [[0.0] * config.VECTOR_DIMENSION for _ in texts]

    def run_dynamic_inference(self, prompt: str, system_instruction: str = "", target_model: str = None) -> dict:
        selected_model = target_model or self.instruct_model_name

        try:
            return self._execute_request(prompt, system_instruction, selected_model)
        except Exception as e:
            fallback_model = (
                self.instruct_model_name if selected_model == self.reasoning_model_name
                else self.reasoning_model_name
            )
            print(f"[!] Primary local model {selected_model} failed ({str(e)}). Rolling back to {fallback_model}...")
            try:
                return self._execute_request(prompt, system_instruction, fallback_model)
            except Exception as critical_err:
                return {
                    "text": f"All sovereign routing paths exhausted. Error: {str(critical_err)}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "model_used": "None"
                }

    def _execute_request(self, prompt: str, system_instruction: str, model_name: str) -> dict:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": f"Task instructions: {system_instruction}"},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 8192}
        }

        response = self.gateway_client.post("/api/chat", json=payload)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama /api/chat returned {response.status_code} for model '{model_name}': "
                f"{response.text[:2000]}"
            )

        body = response.json()

        return {
            "text": body["message"]["content"].strip(),
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "completion_tokens": body.get("eval_count", 0),
            "model_used": model_name
        }


class LLMGatewayRouter:
    """
    Thin routing layer on top of ClinicalInferenceGateway: picks which local
    model handles a task, then delegates the actual call to the gateway's
    /api/chat path.
    """
    def __init__(self, gateway_instance: "ClinicalInferenceGateway"):
        if gateway_instance is None:
            raise ValueError(
                "LLMGatewayRouter requires a ClinicalInferenceGateway instance — "
                "pass one explicitly, e.g. LLMGatewayRouter(gateway_instance=inference_gateway)."
            )
        self.gw = gateway_instance
        self.model_meralion = self.gw.instruct_model_name
        self.model_sealion = self.gw.reasoning_model_name

    def route_query(self, prompt: str, system_prompt: str, task_type: str) -> Tuple[Dict[str, Any], str]:
        if task_type in ["mapping", "extraction", "reasoning"]:
            target = self.model_sealion
        else:
            target = self.model_meralion

        ROUTER_DECISIONS.labels(target_engine=target).inc()

        result = self.gw.run_dynamic_inference(prompt, system_instruction=system_prompt, target_model=target)
        return result, result.get("model_used", target)