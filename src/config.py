import os


class PipelineConfig:
    # -------------------------------------------------------------------------
    # System Network Ports & Microservice Routing
    # -------------------------------------------------------------------------
    GATEWAY_PORT: int = 8050
    MCP_QDRANT_SERVER_PORT: int = 8011
    A2A_ASSISTANT_PORT: int = 8003
    A2A_CODER_PORT: int = 8004

    # -------------------------------------------------------------------------
    # Security Credentials & External Gateways
    # -------------------------------------------------------------------------
    OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "sk-ollama-secure-key-12345")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "sk-qdrant-secure-key-12345")

    # Qdrant Database Direct Entrypoint
    # Default matches the container_name 'local-qdrant-db' specified in docker-compose
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://local-qdrant-db:6333")
    
    # Qdrant Gateway Base Proxy Endpoint
    # Routed dynamically via docker-compose env; falls back to localhost if run natively
    QDRANT_GATEWAY_URL: str = os.getenv("QDRANT_GATEWAY_URL", "http://localhost:8001")

    # Unified Reverse Proxy URL (Nginx Routing Bus)
    # Routed dynamically via docker-compose env; falls back to localhost loopback natively
    NGINX_BASE_URL: str = os.getenv("NGINX_BASE_URL", "http://127.0.0.1:8000")

    # Ollama endpoint reachable from inside containers (via host-gateway translation bridge)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")

    # -------------------------------------------------------------------------
    # Sovereign Generative Language Model Pool (Ollama Local Enclaves)
    # -------------------------------------------------------------------------
    # Flagship model (Excellent regional and cultural knowledge)
    MODEL_SEA_LION: str = "aisingapore/Llama-SEA-LION-v3.5-8B-R:latest"
    # Zero-shot task instruction specialist
    MODEL_MERALION: str = "u1i/LLaMA-3-MERaLiON-8B-Instruct:q4k"

    # -------------------------------------------------------------------------
    # Local High-Speed Vector Space Architecture
    # -------------------------------------------------------------------------
    # Dedicated high-speed embedder engine (Run `ollama pull nomic-embed-text` first)
    EMBEDDING_MODEL_NAME: str = "nomic-embed-text"
    
    # Matches Nomic's default dense vector footprint output dimension
    VECTOR_DIMENSION: int = 768
    COLLECTION_NAME: str = "icd10_diagnoses"


config = PipelineConfig()