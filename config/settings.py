"""
Pipeline Configuration
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# =============================================================================
# OLLAMA CONFIG
# =============================================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"  # llama3 8B
OLLAMA_TIMEOUT = 120  # seconds per request

# =============================================================================
# LABELING CONFIG (Stage 1)
# =============================================================================
LABELING_SAMPLE_SIZE = 18500      # 10% of 185K reviews for LLM labeling
LABELING_BATCH_SIZE = 10          # Reviews per Ollama request
LABELING_CONFIDENCE_THRESHOLD = 0.7  # Min confidence to keep a label

# =============================================================================
# FINE-TUNING CONFIG (Stage 2)
# =============================================================================
MODEL_NAME = "microsoft/deberta-v3-large"  # 304M params, fits ~10GB VRAM
MAX_SEQ_LENGTH = 512              # Paper Sec. III-C4: maximum sequence length 512
TRAIN_BATCH_SIZE = 8              # Reduce to 4 if OOM
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10                   # Paper Sec. III-C4: up to 10 epochs w/ early stopping
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 3
PREDICTION_THRESHOLD = 0.8       # Same as paper: 0.8 for best accuracy

# =============================================================================
# ISSUE GENERATION CONFIG (Stage 4)
# =============================================================================
ISSUE_GEN_BATCH_SIZE = 5

# =============================================================================
# EMOTION CONFIG (Stage 5)
# =============================================================================
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"  # Pre-trained GoEmotions

# =============================================================================
# ANALYSIS CONFIG (Stage 6)
# =============================================================================
MIN_REVIEWS_PER_THEME = 50        # Minimum reviews for a theme to be reported
TOP_N_THEMES = 31                 # Paper Table V: 20 inherited + 11 new
TIME_INTERVAL_DAYS = 30           # 30-day intervals like the paper
BIANNUAL_WINDOW_DAYS = 730        # 2-year sliding window for trend analysis