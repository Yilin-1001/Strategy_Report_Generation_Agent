# -*- coding: utf-8 -*-
"""Simple Ragas test without parallel processing"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rag_project.pipeline import RAGPipeline
from openai import OpenAI
from datasets import Dataset as HFDataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision
from ragas.llms import llm_factory

# Config
API_KEY = "sk-1e0d8cc0ecea4d4d9c54dad669fcc73b"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

print("=" * 80)
print("Simple Ragas Test - Single Question")
print("=" * 80)

# Initialize Pipeline
print("\n1. Initializing Pipeline...")
pipeline = RAGPipeline(
    chunking_config_path="config/chunking_config.yaml",
    milvus_config_path="config/milvus_config.yaml"
)
print("   Pipeline ready")

# Create client
print("\n2. Creating OpenAI client...")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
print("   Client ready")

# Test question
question = "2025年全省农村公路工作部署会是在哪里召开的？"
ground_truth = "2025年全省农村公路工作部署会是在南昌召开的。"

# Search
print(f"\n3. Searching for: {question}")
results = pipeline.search(question, top_k=3)
contexts = [doc['text'] for doc in results]
print(f"   Found {len(contexts)} contexts")

if not contexts:
    print("   ERROR: No contexts found!")
    sys.exit(1)

answer = contexts[0][:500]
print(f"   Answer preview: {answer[:100]}...")

# Create dataset
print("\n4. Creating HuggingFace dataset...")
sample_ds = HFDataset.from_list([{
    "question": question,
    "answer": answer,
    "contexts": contexts,
    "ground_truth": ground_truth
}])
print("   Dataset created")

# Create LLM
print("\n5. Creating Ragas LLM...")
llm = llm_factory(
    model=MODEL_NAME,
    provider="openai",
    client=client,
    max_tokens=8192,
    timeout=120
)
print("   LLM ready")

# Evaluate
print("\n6. Running Ragas evaluation...")
print("   (This may take 20-30 seconds per question)")
result = evaluate(
    dataset=sample_ds,
    metrics=[context_precision, context_recall],
    llm=llm,
    raise_exceptions=True
)

df = result.to_pandas()
row = df.iloc[0]

print("\n" + "=" * 80)
print("Results:")
print("=" * 80)
print(f"Context Precision: {row.get('context_precision', 'N/A')}")
print(f"Context Recall:    {row.get('context_recall', 'N/A')}")
print("=" * 80)

print("\nTest completed successfully!")
