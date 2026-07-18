import os
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.database import LocalVectorDB
from src.gateway import ClinicalInferenceGateway
from src.config import config


def format_icd10_code(raw_code: str) -> str:
    """Formats raw alphanumeric characters into standard ICD-10-CM dotted notation."""
    code = raw_code.strip()
    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"
    return code


def parse_and_seed_icd10_txt(txt_path: str, max_records: int = None, batch_size: int = 128):
    """
    1. Checks if the collection is already seeded — skips everything if so,
       unless FORCE_REINGEST=true is set (idempotent by default).
    2. Otherwise: drops any old data, reinitializes a clean collection sized
       for the configured embedding model, parses the CMS fixed-width file
       with multi-line recovery, and batches embeddings via local Ollama
       (nomic-embed-text) into Qdrant.
    """
    db = LocalVectorDB()

    # Use config as the single source of truth so collection name / dimension
    # can never drift out of sync between ingest.py and LocalVectorDB again.
    collection_name = config.COLLECTION_NAME
    vector_size = config.VECTOR_DIMENSION
    force_reingest = os.getenv("FORCE_REINGEST", "false").strip().lower() == "true"

    # DOCKER FIX: explicit HTTP URL with TLS disabled (host=/port= defaults to HTTPS and fails)
    db.client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        prefer_grpc=False,
        https=False
    )
    # CRITICAL: keep db.collection_name in sync — bulk_upsert() reads this internally.
    # Without this line, ingest writes to a different collection than the one created below.
    db.collection_name = collection_name

    gw = ClinicalInferenceGateway()

    if not os.path.exists(txt_path):
        raise FileNotFoundError(
            f"[!] Missing file at {txt_path}.\n"
            "Please ensure 'icd10cm_order_2026.txt' is in your 'data/' folder."
        )

    # ----------------------------------------------------
    # PARSE FIRST — needed up front now so the idempotency check below can
    # compare against the true expected count, not just "is it non-empty".
    # ----------------------------------------------------
    print(f"[*] Parsing official ICD-10-CM 2026 order file: {txt_path}")

    parsed_records = []
    current_record = None

    with open(txt_path, mode='r', encoding='utf-8') as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue

            if len(line) >= 5 and line[0:5].isdigit():
                if current_record and current_record["billable_flag"] == "1":
                    parsed_records.append({
                        "code": current_record["code"],
                        "short_description": current_record["short_description"],
                        "long_description": current_record["long_description"]
                    })

                raw_code = line[6:14].strip()
                billable_flag = line[14:15].strip()
                short_desc = line[16:77].strip()
                long_desc = line[77:].strip()

                current_record = {
                    "code": format_icd10_code(raw_code),
                    "billable_flag": billable_flag,
                    "short_description": short_desc,
                    "long_description": long_desc
                }
            else:
                if current_record:
                    current_record["long_description"] += " " + stripped

        if current_record and current_record["billable_flag"] == "1":
            parsed_records.append({
                "code": current_record["code"],
                "short_description": current_record["short_description"],
                "long_description": current_record["long_description"]
            })

    if max_records and len(parsed_records) > max_records:
        print(f"[*] Limiting ingestion dataset to first {max_records} billable codes.")
        parsed_records = parsed_records[:max_records]

    expected_count = len(parsed_records)
    print(f"[*] Ingestion Queue: {expected_count} billable codes ready for processing.")

    # ----------------------------------------------------
    # IDEMPOTENCY CHECK — skip only if the collection ALREADY has the full
    # expected count. A partial ingest (e.g. interrupted by a container
    # crash/restart) will no longer be silently treated as "done" — this
    # was a real bug: the previous version only checked points_count > 0,
    # so a partial commit (say ~55,000 of ~74,719 codes) got permanently
    # skipped on every subsequent startup instead of being detected and
    # completed/redone.
    # ----------------------------------------------------
    if not force_reingest:
        try:
            existing = db.client.get_collection(collection_name=collection_name)
            existing_count = existing.points_count or 0
            if existing_count >= expected_count and existing_count > 0:
                print(
                    f"[*] Collection '{collection_name}' already has {existing_count} points "
                    f"(expected {expected_count}). Skipping ingestion. "
                    "Set FORCE_REINGEST=true to wipe and re-embed regardless."
                )
                return
            elif existing_count > 0:
                print(
                    f"[!] Collection '{collection_name}' has only {existing_count} of {expected_count} "
                    "expected points — looks like a previous ingest was interrupted. "
                    "Proceeding with a full wipe and clean re-ingest to fix this."
                )
        except Exception:
            # Collection doesn't exist yet (first run ever) — proceed to create + seed it.
            pass
    else:
        print(f"[*] FORCE_REINGEST=true — will wipe and re-embed '{collection_name}' regardless of current state.")

    # ----------------------------------------------------
    # GROUND ZERO RESET
    # ----------------------------------------------------
    print(f"[-] Dropping existing collection '{collection_name}' to start fresh (force_reingest={force_reingest})...")
    try:
        db.client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"[*] Collection deletion skipped or not found: {e}")

    print(f"[-] Reinitializing '{collection_name}' with {vector_size} dimensions...")
    db.client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    # ----------------------------------------------------

    failed_embeddings = 0

    try:
        points_batch = []

        for i in tqdm(range(0, len(parsed_records), batch_size), desc="Processing clinical batches"):
            batch = parsed_records[i: i + batch_size]

            text_blocks = [
                f"ICD-10-CM Code: {row['code']} | Description: {row['long_description']} "
                f"(Alternative term: {row['short_description']})"
                for row in batch
            ]

            vectors = gw.generate_local_embeddings_batch(text_blocks)

            for row, vector in zip(batch, vectors):
                if not any(vector):
                    failed_embeddings += 1

                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row['code']))

                points_batch.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "code": row['code'],
                            "short_description": row['short_description'],
                            "long_description": row['long_description']
                        }
                    )
                )

            if points_batch:
                db.bulk_upsert(points_batch)
                points_batch.clear()
        
        print(f"[+] Sovereign Vector Space successfully seeded with {config.EMBEDDING_MODEL_NAME} embeddings.")
        if failed_embeddings:
            print(
                f"[!] WARNING: {failed_embeddings} of {len(parsed_records)} codes got zero-vectors "
                "(embedding calls failed) and will not be retrievable via similarity search. "
                "Check Ollama connectivity / model availability and re-run if this number is high."
            )
    except Exception as e:
        print(f"[!] Critical Loader Failure: {str(e)}")


if __name__ == "__main__":
    TARGET_FILE = "data/icd10cm_order_2026.txt"
    # Full run — set max_records to a smaller number only for local dev smoke tests.
    parse_and_seed_icd10_txt(TARGET_FILE, max_records=None, batch_size=64)