"""
Hallucination Evaluation Configuration

Dual-model setup via SiliconFlow:
- Decomposition: Kimi-K2.5 (long text understanding)
- Verification: Qwen3.5-122B-A10B (efficient fact checking)

Aligned with FActScore methodology (Min et al., 2023).
"""

import os
from pathlib import Path
from datetime import datetime

# === Project Paths ===
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "ablation_experiment" / "reports"
RESULTS_BASE_DIR = Path(__file__).parent / "results"

# === SiliconFlow API ===
EVAL_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
EVAL_BASE_URL = "https://api.siliconflow.cn/v1"

# === Decomposition Model (needs strong Chinese understanding) ===
DECOMPOSE_MODEL = "Pro/moonshotai/Kimi-K2.5"
DECOMPOSE_TIMEOUT = 180
DECOMPOSE_MAX_TOKENS = 4096

# === Verification Model (only outputs 1 word, can be faster) ===
VERIFY_MODEL = "Pro/moonshotai/Kimi-K2.5"
VERIFY_TIMEOUT = 180

# === Retrieval Configuration ===
RAG_TOP_K = 5                # B-type: RAG semantic search results
CITED_CHUNKS_TOP_K = 5       # A-type: Filtered chunks from cited source
MAX_CONTEXT_TOKENS = 2000    # Max context sent to verification LLM
KB_CHUNKS_PATH = DATA_DIR / "all_chunks.json"  # Knowledge base chunks file

# === Concurrency Control ===
LLM_CALL_INTERVAL = 2        # Seconds between LLM calls (rate limiting)
RAG_CALL_INTERVAL = 1        # Seconds between RAG searches

# === Statistical Analysis ===
ALPHA = 0.05                 # Significance level for hypothesis tests
GROUPS = ["group0", "group1", "group2", "group3"]
NUM_RUNS = 1                 # Reports per group

# === Result Output ===
def create_experiment_dir() -> Path:
    """Create a new experiment results directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    exp_dir = RESULTS_BASE_DIR / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "raw").mkdir(exist_ok=True)

    return exp_dir

# === Ablation Chapter Plan (from config.py) ===
CHAPTER_PLAN = [
    {"title": "第一章：宏观政策环境与时代要求", "phase": "diagnosis", "index": 0},
    {"title": "第二章：区域战略与'交通强省'建设剖析", "phase": "diagnosis", "index": 1},
    {"title": "第三章：行业演进趋势与当前内部诊断", "phase": "diagnosis", "index": 2},
    {"title": "第四章：总体战略思路与政策响应目标", "phase": "initiatives", "index": 3},
    {"title": "第五章：主责主业：高质量建设与保通保畅举措", "phase": "initiatives", "index": 4},
    {"title": "第六章：创新驱动：绿色低碳与智慧交投建设", "phase": "initiatives", "index": 5},
    {"title": "第七章：产业协同：交旅融合与服务地方经济", "phase": "initiatives", "index": 6},
    {"title": "第八章：治理效能：深化国企改革与党建引领", "phase": "initiatives", "index": 7},
]