"""
CareerRAG — Centralized Configuration
All tuneable constants live here so they can be adjusted in one place.
"""

# ---------------------------
# LLM Models (Groq)
# ---------------------------
PRIMARY_MODEL = "llama-3.1-8b-instant"
FALLBACK_MODEL = "llama-3.3-70b-versatile"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800

VALIDATION_TEMPERATURE = 0.0
VALIDATION_MAX_TOKENS = 500

# ---------------------------
# Embedding & Re-ranking
# ---------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------
# Chunking
# ---------------------------
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
CHUNK_SEPARATORS = ["\n\n", "\n", ".", " ", ""]

# ---------------------------
# Retrieval
# ---------------------------
RRF_K = 60                    # Reciprocal Rank Fusion constant
CANDIDATE_MULTIPLIER = 2      # Retrieve top_k * this many candidates before re-ranking
MIN_CANDIDATES = 10           # Minimum candidate pool size

# ---------------------------
# Confidence thresholds
# (applied to sigmoid-normalized cross-encoder scores, range 0–1)
# ---------------------------
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.45
