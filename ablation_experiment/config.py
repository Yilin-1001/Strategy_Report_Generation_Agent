"""
Ablation Experiment Configuration

This module contains shared configuration for all ablation experiment groups:
- API keys and LLM settings
- The fixed 8-chapter plan (identical across all groups)
- Experiment settings (number of runs, output paths)
- Test query string
"""

import os
from pathlib import Path

# === Test Query ===
TEST_QUERY = "生成江西交通投资集团的战略规划报告"

# === Generation LLM (DeepSeek) ===
GEN_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEN_BASE_URL = "https://api.deepseek.com"
GEN_MODEL = "deepseek-chat"
GEN_TIMEOUT = 180

# === Evaluation LLM (Qwen via SiliconFlow) ===
EVAL_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
EVAL_BASE_URL = "https://api.siliconflow.cn/v1"
EVAL_MODEL = "Qwen/Qwen3.5-122B-A10B"  # Qwen 3.5 122B for rigorous evaluation
EVAL_TIMEOUT = 300

# === Repeated Evaluation Settings ===
NUM_EVAL_ROUNDS = 10       # 每份报告独立打分10次
EVAL_TEMPERATURE = 0.3     # 降低评估温度，受控变异以检测幻觉
EVAL_INTERVAL = 8          # 两次打分间隔(秒)，避免TPM限流
OUTLIER_METHOD = "trimmed_mean"  # 离群值处理方式
TRIM_COUNT = 2             # 截尾均值：去掉最高最低各2个，保留中间6个

# === Experiment Settings ===
NUM_RUNS = 3  # Each group runs 3 times for statistical significance

# === Output Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "ablation_experiment" / "reports"
RESULTS_DIR = PROJECT_ROOT / "ablation_experiment" / "results"

# Ensure output directories exist
for group_dir in ["group0", "group1", "group2", "group3"]:
    (REPORTS_DIR / group_dir).mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === Fixed 8-Chapter Strategic Plan ===
# This plan is identical to coordinator.py's global_plan
# All 4 groups use this same plan structure

CHAPTER_PLAN = [
    {
        "title": "第一章：宏观政策环境与时代要求",
        "phase": "diagnosis",
        "analysis_model": "PEST模型 (侧重P-政策与E-经济维度)",
        "index": 0
    },
    {
        "title": "第二章：区域战略与'交通强省'建设剖析",
        "phase": "diagnosis",
        "analysis_model": "无特定模型，侧重省级政策承接与区域占位分析",
        "index": 1
    },
    {
        "title": "第三章：行业演进趋势与当前内部诊断",
        "phase": "diagnosis",
        "analysis_model": "波特五力模型与SWOT分析 (强制要求在分析结尾输出结构化的SWOT矩阵)",
        "index": 2
    },
    {
        "title": "第四章：总体战略思路与政策响应目标",
        "phase": "initiatives",
        "analysis_model": "平衡计分卡(BSC)模型 (从财务、客户/民生、内部运营、学习与成长四个维度设定目标)",
        "index": 3
    },
    {
        "title": "第五章：主责主业：高质量建设与保通保畅举措",
        "phase": "initiatives",
        "analysis_model": "BCG波士顿矩阵 (将主业作为'现金牛'业务，侧重精益化与稳健回报)",
        "index": 4
    },
    {
        "title": "第六章：创新驱动：绿色低碳与智慧交投建设",
        "phase": "initiatives",
        "analysis_model": "安索夫矩阵 (将创新业务作为新产品/新市场拓展，侧重第二增长曲线)",
        "index": 5
    },
    {
        "title": "第七章：产业协同：交旅融合与服务地方经济",
        "phase": "initiatives",
        "analysis_model": "产业链协同与ESG社会责任模型",
        "index": 6
    },
    {
        "title": "第八章：治理效能：深化国企改革与党建引领",
        "phase": "initiatives",
        "analysis_model": "麦肯锡7S模型 (从结构、制度、风格、员工、技能等维度构建组织保障)",
        "index": 7
    },
]

# === Agent Temperature Settings (from rag_project config) ===
AGENT_TEMPERATURES = {
    "researcher": 0.1,  # Precise and thorough
    "analyst": 0.5,     # Balanced and analytical
    "writer": 0.7,      # Creative and clear
    "single_agent": 0.5,  # Middle ground for single-agent groups
}