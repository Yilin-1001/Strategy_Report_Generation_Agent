# RAG 评估模块使用指南

## 📁 项目结构

```
rag_eval/
├── README.md                           # 本文档
├── build_testset_simple.py             # 测试集生成器（成功版本）
├── ragas_glm_fixed.py                  # Ragas 评估脚本（成功版本）
├── evaluate_testset.py                 # 基础评估脚本
├── retrieval_eval.py                   # 检索质量评估
└── evals/
    ├── datasets/
    │   └── ragas_testset_doc_51q.json  # 51个测试问题
    └── experiments/
        ├── ragas_metrics_fixed.json    # Ragas 评估指标
        ├── ragas_metrics_fixed.csv     # Ragas CSV 格式
        ├── ragas_detailed_results_fixed.csv  # 详细评估结果
        ├── retrieval_eval_results.json # 检索评估结果
        └── retrieval_eval_results.csv  # 检索评估 CSV
```

## 🎯 评估结果

### Ragas 框架评估（使用 GLM-4-flash）

| 指标 | 数值 | 说明 |
|------|------|------|
| **Context Precision** | 0.9616 (96.16%) | 检索到的文档与问题高度相关 |
| **Context Recall** | 0.9274 (92.74%) | 检索到了大部分相关信息 |
| **F1 Score** | 0.9442 (94.42%) | 综合性能优秀 |

### 测试集信息
- **问题数量**: 51个
- **来源**: 从17个完整文档生成
- **Top-K**: 5个文档/问题
- **评估模型**: GLM-4-flash (智谱AI)

## 🚀 使用方法

### 1. 生成测试集（如需要）

```bash
python rag_eval/build_testset_simple.py \
  --input 知识库/知识库 \
  --output rag_eval/evals/datasets/ragas_testset_doc_51q.json \
  --num-questions 3
```

### 2. 运行 Ragas 评估

```bash
python rag_eval/ragas_glm_fixed.py \
  --testset rag_eval/evals/datasets/ragas_testset_doc_51q.json \
  --top-k 5 \
  --output rag_eval/evals/experiments/ragas_metrics_fixed.json
```

### 3. 运行基础检索评估

```bash
python rag_eval/evaluate_testset.py \
  --testset rag_eval/evals/datasets/ragas_testset_doc_51q.json \
  --top-k 5
```

## 📊 结果文件说明

### Ragas 评估结果

1. **ragas_metrics_fixed.json** - 指标汇总
   ```json
   {
     "metrics": {
       "context_precision": 0.9616,
       "context_recall": 0.9274,
       "f1_score": 0.9442
     }
   }
   ```

2. **ragas_metrics_fixed.csv** - CSV 格式指标

3. **ragas_detailed_results_fixed.csv** - 每个问题的详细评分
   - 包含每个问题的 context_precision 和 context_recall
   - 可用于分析具体问题的表现

### 检索评估结果

- **retrieval_eval_results.json/csv** - 基础检索质量统计
  - 检索成功率
  - 平均分数
  - 分数分布

## 🔧 配置说明

### API 配置
在 `ragas_glm_fixed.py` 中配置：

```python
API_KEY = "your_glm_api_key"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"
```

### 关键配置

- **max_tokens=8192** - 必须设置为 8192 以避免输出截断
- **provider="openai"** - 使用 OpenAI 兼容接口
- **llm 参数** - 必须将 LLM 实例传递给 evaluate() 函数

## ⚠️ 重要提示

1. **max_tokens 必须设置为 8192**
   - Ragas 的结构化输出需要较多 token
   - 使用默认的 3072 会导致输出被截断

2. **必须创建 LLM 实例并传递给 evaluate()**
   ```python
   ragas_llm = llm_factory(model, provider="openai", client=client, max_tokens=8192)
   result = evaluate(dataset, metrics, llm=ragas_llm)
   ```

3. **使用 result.to_pandas() 访问结果**
   - Ragas 返回 EvaluationResult 对象
   - 使用 to_pandas() 方法转换为 DataFrame

## 📈 性能指标说明

### Context Precision (上下文精确度)
- **定义**: 检索到的文档中与问题相关的比例
- **范围**: 0-1，越高越好
- **我们的结果**: 0.9616 (优秀)

### Context Recall (上下文召回率)
- **定义**: 检索到的文档覆盖 ground truth 的程度
- **范围**: 0-1，越高越好
- **我们的结果**: 0.9274 (优秀)

### F1 Score
- **定义**: Precision 和 Recall 的调和平均数
- **范围**: 0-1，越高越好
- **我们的结果**: 0.9442 (优秀)

## 🎓 结论

使用 Ragas 框架和 GLM-4-flash 模型成功评估了 RAG 系统的检索质量：

✅ **Precision 96.16%** - 检索结果高度相关
✅ **Recall 92.74%** - 检索覆盖全面
✅ **F1 Score 94.42%** - 综合性能优秀

这表明当前的 RAG 系统在检索质量方面表现优异，能够准确且全面地找到与用户问题相关的文档。
