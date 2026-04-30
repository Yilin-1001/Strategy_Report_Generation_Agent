# Reranker Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reranking module to the existing RAG pipeline using SiliconFlow's rerank API, then evaluate retrieval quality before and after reranking with two models (BGE-Reranker-v2-m3 and Qwen3-Reranker).

**Architecture:** A standalone `rag_project/reranker/` package with an abstract base class and a SiliconFlow API implementation. The existing `RAGPipeline.search()` gains an optional `use_reranker=False` parameter — default behavior is completely unchanged. A dedicated evaluation script runs the same ragas testset through three configurations (baseline, BGE, Qwen3) and produces a comparison report.

**Tech Stack:** SiliconFlow Rerank API (`POST /rerank`), existing ragas evaluation framework, existing `ragas_testset_doc_51q.json` testset.

---

### Task 1: Create Reranker Abstract Base Class

**Files:**
- Create: `rag_project/reranker/__init__.py`
- Create: `rag_project/reranker/reranker_base.py`

**Step 1: Create the `__init__.py`**

```python
# rag_project/reranker/__init__.py
from rag_project.reranker.reranker_base import BaseReranker

__all__ = ["BaseReranker"]
```

**Step 2: Create the abstract base class**

```python
# rag_project/reranker/reranker_base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Single reranked result"""
    text: str
    score: float
    original_index: int
    metadata: Dict


class BaseReranker(ABC):
    """Abstract base class for rerankers"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank a list of retrieved documents.

        Args:
            query: The search query
            documents: List of dicts with 'text', 'score', 'metadata' keys
                       (same format as MilvusManager.search() output)
            top_k: Number of results to return after reranking

        Returns:
            List of dicts with same format: {'text', 'score', 'metadata'}
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Return model info for logging"""
        pass
```

**Step 3: Commit**

```bash
git add rag_project/reranker/__init__.py rag_project/reranker/reranker_base.py
git commit -m "feat: add reranker abstract base class"
```

---

### Task 2: Implement SiliconFlow Reranker

**Files:**
- Create: `rag_project/reranker/siliconflow_reranker.py`
- Create: `config/reranker_config.yaml`

**Step 1: Create the configuration file**

```yaml
# config/reranker_config.yaml
reranker:
  enabled: true
  provider: "siliconflow"

  # Model: "BAAI/bge-reranker-v2-m3" or "Qwen/Qwen3-Reranker-0.6B" etc.
  model: "BAAI/bge-reranker-v2-m3"

  # SiliconFlow API (reuse same key as embedding)
  api_base_url: "https://api.siliconflow.cn/v1"
  api_key: "sk-galtgnlesfoydjzetapqerkapxzqbtdnlnyulgibiqymvqik"

  # Retrieval expansion: Milvus retrieves top_k * expansion_factor, then rerank down to top_k
  expansion_factor: 3

  # Request timeout in seconds
  api_timeout: 30

  # BGE-specific: max chunks per long document
  max_chunks_per_doc: 1

  # BGE-specific: overlap tokens between chunks
  overlap_tokens: 40
```

**Step 2: Implement the SiliconFlow reranker**

```python
# rag_project/reranker/siliconflow_reranker.py
import requests
from typing import List, Dict, Optional
from rag_project.reranker.reranker_base import BaseReranker
from rag_project.utils.config_loader import load_config
from rag_project.utils.logger import logger


class SiliconFlowReranker(BaseReranker):
    """Reranker using SiliconFlow's /rerank API"""

    def __init__(self, config_path: str = "config/reranker_config.yaml"):
        self.config = load_config(config_path).get('reranker', {})

        self.model = self.config.get('model', 'BAAI/bge-reranker-v2-m3')
        self.api_base_url = self.config.get('api_base_url', 'https://api.siliconflow.cn/v1')
        self.api_key = self.config.get('api_key')
        self.timeout = self.config.get('api_timeout', 30)
        self.max_chunks_per_doc = self.config.get('max_chunks_per_doc', 1)
        self.overlap_tokens = self.config.get('overlap_tokens', 40)

        if not self.api_key:
            import os
            self.api_key = os.environ.get('SILICONFLOW_API_KEY')
            if not self.api_key:
                raise ValueError(
                    "SiliconFlow API key required. Set in reranker_config.yaml "
                    "or SILICONFLOW_API_KEY env var."
                )

        logger.info(f"SiliconFlowReranker initialized: model={self.model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents via SiliconFlow API.

        Args:
            query: Search query string
            documents: Milvus search results [{'text', 'score', 'metadata'}]
            top_k: Number of results to return

        Returns:
            Reranked results in same format, with new 'rerank_score' field
        """
        if not documents:
            return []

        doc_texts = [doc['text'] for doc in documents]

        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_k, len(documents)),
            "return_documents": True,
        }

        # BGE-specific params
        if 'bge' in self.model.lower():
            payload["max_chunks_per_doc"] = self.max_chunks_per_doc
            payload["overlap_tokens"] = self.overlap_tokens

        try:
            response = requests.post(
                f"{self.api_base_url}/rerank",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Rerank API error: {e}")
            logger.warning("Falling back to original order")
            return documents[:top_k]

        # Build reranked results
        reranked = []
        for item in data.get('results', []):
            idx = item['index']
            original_doc = documents[idx]
            reranked.append({
                'text': item.get('document', {}).get('text', original_doc['text']),
                'score': item['relevance_score'],
                'original_score': original_doc['score'],
                'original_rank': idx,
                'metadata': original_doc['metadata'],
            })

        logger.info(f"Reranked {len(documents)} -> {len(reranked)} results")
        return reranked

    def get_model_info(self) -> Dict:
        return {
            'type': 'siliconflow_reranker',
            'model': self.model,
            'api_base_url': self.api_base_url,
        }
```

**Step 3: Update `__init__.py` to export both classes**

```python
# rag_project/reranker/__init__.py
from rag_project.reranker.reranker_base import BaseReranker
from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker

__all__ = ["BaseReranker", "SiliconFlowReranker"]
```

**Step 4: Commit**

```bash
git add rag_project/reranker/ config/reranker_config.yaml
git commit -m "feat: implement SiliconFlow reranker with BGE and Qwen3 support"
```

---

### Task 3: Integrate Reranker into RAGPipeline

**Files:**
- Modify: `rag_project/pipeline.py`

This is the ONLY modification to existing code. The change is minimal and backward-compatible.

**Step 1: Add reranker initialization to `__init__`**

In `rag_project/pipeline.py`, add after line 56 (`self.milvus_manager = ...`):

```python
        # Reranker (optional, disabled by default)
        self.reranker = None
        self.reranker_config = {}
        try:
            from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
            reranker_config = load_config("config/reranker_config.yaml").get('reranker', {})
            if reranker_config.get('enabled', False):
                self.reranker = SiliconFlowReranker("config/reranker_config.yaml")
                self.reranker_config = reranker_config
                logger.info("Reranker enabled")
            else:
                logger.info("Reranker disabled")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")
```

**Step 2: Modify `search()` method**

Replace the existing `search()` method (lines 176-203) with:

```python
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_reranker: bool = False
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (e.g., {"doc_type": ["news"]})
            use_reranker: Whether to apply reranking (requires reranker enabled)

        Returns:
            List of search results
        """
        # Generate query embedding
        query_vector = self.embedding_model.embed_text(query)

        # Determine retrieval count
        if use_reranker and self.reranker:
            expansion = self.reranker_config.get('expansion_factor', 3)
            retrieve_k = top_k * expansion
            logger.info(f"Rerank mode: retrieving {retrieve_k} -> reranking to {top_k}")
        else:
            retrieve_k = top_k

        # Search Milvus
        results = self.milvus_manager.search(
            query_vector.tolist(),
            top_k=retrieve_k,
            filters=filters
        )

        # Apply reranking if requested and available
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)

        return results
```

**Step 3: Commit**

```bash
git add rag_project/pipeline.py
git commit -m "feat: integrate reranker into RAGPipeline with opt-in flag"
```

---

### Task 4: Create Reranker Evaluation Script

**Files:**
- Create: `rag_eval/evaluate_reranker.py`

This script runs three experiments using the SAME testset and produces a side-by-side comparison.

**Step 1: Write the evaluation script**

```python
# rag_eval/evaluate_reranker.py
"""
Reranker comparison evaluation.

Runs three experiments on the same testset:
1. Baseline (vector search only)
2. BGE-Reranker-v2-m3
3. Qwen3-Reranker-0.6B

Produces comparison report with ragas + traditional metrics.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_project.pipeline import RAGPipeline


# Reranker configurations to test
RERANKER_CONFIGS = {
    "baseline": None,
    "bge_reranker": {
        "reranker": {
            "enabled": True,
            "provider": "siliconflow",
            "model": "BAAI/bge-reranker-v2-m3",
            "api_base_url": "https://api.siliconflow.cn/v1",
            "api_key": "sk-galtgnlesfoydjzetapqerkapxzqbtdnlnyulgibiqymvqik",
            "expansion_factor": 3,
            "api_timeout": 30,
            "max_chunks_per_doc": 1,
            "overlap_tokens": 40,
        }
    },
    "qwen3_reranker": {
        "reranker": {
            "enabled": True,
            "provider": "siliconflow",
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "api_base_url": "https://api.siliconflow.cn/v1",
            "api_key": "sk-galtgnlesfoydjzetapqerkapxzqbtdnlnyulgibiqymvqik",
            "expansion_factor": 3,
            "api_timeout": 30,
        }
    },
}


def run_single_experiment(
    experiment_name: str,
    testset: List[Dict],
    top_k: int = 5,
    reranker_config: Dict = None
) -> Dict[str, Any]:
    """
    Run a single retrieval experiment.

    Args:
        experiment_name: Name of the experiment
        testset: List of {'question', 'ground_truth'} dicts
        top_k: Number of results to retrieve
        reranker_config: If None, baseline. If dict, enable reranker.

    Returns:
        Experiment results dict
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")

    # Create pipeline (always fresh to pick up config changes)
    pipeline = RAGPipeline(
        chunking_config_path="config/chunking_config.yaml",
        milvus_config_path="config/milvus_config.yaml",
    )

    # If reranker config provided, manually set it
    use_reranker = False
    if reranker_config is not None:
        from rag_project.reranker.siliconflow_reranker import SiliconFlowReranker
        pipeline.reranker = SiliconFlowReranker.__new__(SiliconFlowReranker)
        pipeline.reranker.model = reranker_config['reranker']['model']
        pipeline.reranker.api_base_url = reranker_config['reranker']['api_base_url']
        pipeline.reranker.api_key = reranker_config['reranker']['api_key']
        pipeline.reranker.timeout = reranker_config['reranker'].get('api_timeout', 30)
        pipeline.reranker.max_chunks_per_doc = reranker_config['reranker'].get('max_chunks_per_doc', 1)
        pipeline.reranker.overlap_tokens = reranker_config['reranker'].get('overlap_tokens', 40)
        pipeline.reranker_config = reranker_config['reranker']
        use_reranker = True
        print(f"  Reranker: {reranker_config['reranker']['model']}")
    else:
        print("  Mode: Baseline (vector search only)")

    results = []
    total_latency = 0

    for i, item in enumerate(testset, 1):
        question = item['question']
        ground_truth = item['ground_truth']

        start_time = time.time()
        search_results = pipeline.search(
            question, top_k=top_k, use_reranker=use_reranker
        )
        latency = time.time() - start_time
        total_latency += latency

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "num_retrieved": len(search_results),
            "top1_score": search_results[0]['score'] if search_results else 0,
            "top1_source": (
                search_results[0]['metadata'].get('source', '')
                if search_results else ''
            ),
            "retrieved_docs": search_results,
            "latency": latency,
        })

        print(f"  [{i}/{len(testset)}] {question[:50]}... "
              f"(top1={results[-1]['top1_score']:.4f}, {latency:.2f}s)")

    # Compute statistics
    total = len(results)
    avg_score = sum(r['top1_score'] for r in results) / total
    avg_latency = total_latency / total
    high_score = sum(1 for r in results if r['top1_score'] >= 0.8)
    mid_score = sum(1 for r in results if 0.6 <= r['top1_score'] < 0.8)
    low_score = sum(1 for r in results if r['top1_score'] < 0.6)

    return {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_questions": total,
            "avg_top1_score": avg_score,
            "avg_latency": avg_latency,
            "total_latency": total_latency,
            "score_distribution": {
                "high_gte_0.8": high_score,
                "mid_0.6_0.8": mid_score,
                "low_lt_0.6": low_score,
            },
        },
        "results": results,
    }


def compare_experiments(all_results: List[Dict]) -> Dict:
    """Generate comparison report from multiple experiment results."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
        "summary_table": [],
    }

    for exp in all_results:
        name = exp["experiment_name"]
        stats = exp["statistics"]
        comparison["experiments"][name] = stats
        comparison["summary_table"].append({
            "experiment": name,
            "avg_top1_score": round(stats["avg_top1_score"], 4),
            "avg_latency": round(stats["avg_latency"], 2),
            "high_score_pct": round(
                stats["score_distribution"]["high_gte_0.8"] / stats["total_questions"] * 100, 1
            ),
            "mid_score_pct": round(
                stats["score_distribution"]["mid_0.6_0.8"] / stats["total_questions"] * 100, 1
            ),
            "low_score_pct": round(
                stats["score_distribution"]["low_lt_0.6"] / stats["total_questions"] * 100, 1
            ),
        })

    return comparison


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reranker comparison evaluation")
    parser.add_argument(
        "--testset",
        type=str,
        default="rag_eval/evals/datasets/ragas_testset_doc_51q.json",
        help="Testset JSON file",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rag_eval/evals/experiments",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["baseline", "bge_reranker", "qwen3_reranker"],
        choices=["baseline", "bge_reranker", "qwen3_reranker"],
        help="Which experiments to run",
    )

    args = parser.parse_args()

    # Load testset
    with open(args.testset, 'r', encoding='utf-8') as f:
        data = json.load(f)
        testset = data['questions']

    print(f"Loaded {len(testset)} questions from {args.testset}")
    print(f"Experiments: {args.experiments}")
    print(f"Top-K: {args.top_k}")

    # Run experiments
    all_results = []
    for exp_name in args.experiments:
        config = RERANKER_CONFIGS[exp_name]
        result = run_single_experiment(
            experiment_name=exp_name,
            testset=testset,
            top_k=args.top_k,
            reranker_config=config,
        )
        all_results.append(result)

        # Save individual results
        output_path = Path(args.output_dir) / f"reranker_{exp_name}_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved: {output_path}")

    # Generate comparison report
    comparison = compare_experiments(all_results)
    comparison_path = Path(args.output_dir) / "reranker_comparison_report.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\nComparison report: {comparison_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<25} {'Avg Score':>10} {'Avg Latency':>12} "
          f"{'High%':>8} {'Mid%':>8} {'Low%':>8}")
    print("-" * 80)
    for row in comparison["summary_table"]:
        print(f"{row['experiment']:<25} {row['avg_top1_score']:>10.4f} "
              f"{row['avg_latency']:>10.2f}s "
              f"{row['high_score_pct']:>7.1f}% {row['mid_score_pct']:>7.1f}% "
              f"{row['low_score_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add rag_eval/evaluate_reranker.py
git commit -m "feat: add reranker comparison evaluation script"
```

---

### Task 5: Manual Smoke Test

**No new files.** Just verify the integration works before running the full evaluation.

**Step 1: Test baseline search (should work exactly as before)**

```bash
cd "E:/02 Final Year Project/RAG Project"
python -c "
from rag_project.pipeline import RAGPipeline
p = RAGPipeline()
results = p.search('江西省交通投资集团的主要职责是什么？', top_k=3)
for r in results:
    print(f'Score: {r[\"score\"]:.4f} | {r[\"metadata\"][\"source\"][:40]}')
"
```

**Step 2: Test reranker search (with use_reranker=True)**

```bash
python -c "
from rag_project.pipeline import RAGPipeline
p = RAGPipeline()
results = p.search('江西省交通投资集团的主要职责是什么？', top_k=3, use_reranker=True)
for r in results:
    score = r.get('rerank_score', r.get('score', 0))
    orig = r.get('original_score', 'N/A')
    print(f'RerankScore: {score:.4f} (orig: {orig}) | {r[\"metadata\"][\"source\"][:40]}')
"
```

**Step 3: Test evaluation script with a small subset**

```bash
cd "E:/02 Final Year Project/RAG Project"
python rag_eval/evaluate_reranker.py --experiments baseline bge_reranker --top-k 3
```

Verify output files appear in `rag_eval/evals/experiments/`.

**Step 4: Commit any fixes if needed**

---

### Task 6: Run Full Evaluation

**Step 1: Run all three experiments on the full testset**

```bash
cd "E:/02 Final Year Project/RAG Project"
python rag_eval/evaluate_reranker.py \
    --testset rag_eval/evals/datasets/ragas_testset_doc_51q.json \
    --top-k 5 \
    --experiments baseline bge_reranker qwen3_reranker
```

This will:
1. Run 51 questions x 3 experiments = 153 total queries
2. Save individual results to `rag_eval/evals/experiments/reranker_{name}_results.json`
3. Save comparison report to `rag_eval/evals/experiments/reranker_comparison_report.json`
4. Print a summary table

**Step 2: Review the comparison report**

```bash
cat rag_eval/evals/experiments/reranker_comparison_report.json
```

**Step 3: Commit results**

```bash
git add rag_eval/evals/experiments/reranker_*
git commit -m "eval: add reranker comparison results (baseline vs BGE vs Qwen3)"
```
