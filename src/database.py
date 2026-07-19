# database.py
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from src.config import config


class LocalVectorDB:
    """Encapsulates the vector reference layer connecting to Qdrant."""

    def __init__(self):
        # Default: direct connection to Qdrant on the internal Docker network.
        # This prevents deadlocks during offline data ingestion tasks.
        qdrant_url = os.getenv("QDRANT_URL", config.QDRANT_URL)

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=config.QDRANT_API_KEY,
            prefer_grpc=False,
            https=False,
        )
        self.collection_name = config.COLLECTION_NAME
        self.vector_dimension = config.VECTOR_DIMENSION
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE,
                        on_disk=True  # Keeps RAM footprint low
                    )
                )
        except Exception as e:
            print(f"[!] Warning: Could not initialize database collection: {str(e)}")

    def bulk_upsert(self, points: list[PointStruct]):
        """Standard insert for small operational updates."""
        self.client.upsert(collection_name=self.collection_name, points=points)

    def parallel_batch_ingest(self, points: list[PointStruct]):
        """
        High-speed ingestion for initial database population.
        Bypasses standard upsert to use Qdrant's optimized upload_points.
        """
        print(f"[*] Starting high-speed ingestion of {len(points)} points...")
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=256,  # Optimal chunk size for network IO
            parallel=4       # Uses 4 CPU threads to push data
        )
        print("[*] Ingestion complete.")

    def search_similar_codes(self, query_vector: list[float], limit: int = 10) -> list[dict]:
        """Performs a similarity search against stored ICD-10 vectors."""
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [
                {
                    "code": h.payload.get("code", "UNKNOWN"),
                    # Keep raw fields for API responses
                    "short_description": h.payload.get("short_description", ""),
                    "long_description": h.payload.get("long_description", ""),
                    # ALIGNMENT FIX: Synthesize 'description' for agent.py's cross-encoder
                    "description": h.payload.get("long_description", h.payload.get("short_description", "")),
                    "score": round(float(h.score), 4)
                }
                for h in hits
            ]
        except Exception as e:
            print(f"[!] Database Search Error: {str(e)}")
            return []