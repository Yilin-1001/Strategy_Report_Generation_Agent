# RAG评估手动运行指南

## 📋 快速开始

### 方案1: 快速评估（推荐首次使用）

**评估10个问题** - 快速验证系统是否正常工作

```bash
cd "E:\02 Final Year Project\RAG Project"
python rag_eval/ragas_quick_eval.py --num-questions 10
```

**预期时间：** 5-10分钟
**适用场景：** 快速测试、验证功能

---

### 方案2: 中等规模评估

**评估50个问题** - 获得可靠的结果

```bash
python rag_eval/ragas_quick_eval.py --num-questions 50
```

**预期时间：** 15-25分钟
**适用场景：** 常规评估、性能基准测试

---

### 方案3: 大规模评估

**评估100个问题** - 获得统计显著的结果

```bash
python rag_eval/ragas_quick_eval.py --num-questions 100
```

**预期时间：** 30-50分钟
**适用场景：** 全面评估、论文实验

---

### 方案4: 完整评估

**评估全部464个问题** - 使用完整Ragas框架

```bash
python rag_eval/ragas_glm_fixed.py
```

**预期时间：** 2-3小时
**适用场景：** 最终评估、生产部署前
**注意：** 可以在后台运行，中断可继续

---

## 📝 命令行参数说明

### ragas_quick_eval.py 参数

```bash
python rag_eval/ragas_quick_eval.py [选项]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--testset` | `rag_eval/evals/datasets/smart_testset_full.json` | 测试集文件路径 |
| `--num-questions` | `50` | 评估的问题数量 |
| `--top-k` | `5` | 检索返回的文档数 |

**示例：**
```bash
# 使用默认参数（50问）
python rag_eval/ragas_quick_eval.py

# 自定义问题数量
python rag_eval/ragas_quick_eval.py --num-questions 100

# 自定义Top-K
python rag_eval/ragas_quick_eval.py --num-questions 30 --top-k 10

# 使用不同的测试集
python rag_eval/ragas_quick_eval.py --testset "rag_eval/evals/datasets/smart_testset_10doc_test.json"
```

### ragas_glm_fixed.py 参数

```bash
python rag_eval/ragas_glm_fixed.py [选项]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--testset` | `rag_eval/evals/datasets/smart_testset_full.json` | 测试集文件路径 |
| `--top-k` | `5` | 检索返回的文档数 |
| `--output` | `rag_eval/evals/experiments/ragas_metrics_fixed.json` | 结果输出路径 |

**示例：**
```bash
# 使用默认参数
python rag_eval/ragas_glm_fixed.py

# 自定义Top-K
python rag_eval/ragas_glm_fixed.py --top-k 10

# 自定义输出路径
python rag_eval/ragas_glm_fixed.py --output "my_results.json"
```

---

## 📊 查看评估结果

### 1. 查看控制台输出

评估完成后，控制台会显示：

```
================================================================================
Ragas 评估结果
================================================================================

检索质量指标:
  context_precision: 0.9456
  context_recall: 1.0000
  f1_score: 0.9720
```

### 2. 查看JSON结果文件

```bash
# 快速评估结果
cat rag_eval/evals/experiments/ragas_metrics_10q.json
```

**内容示例：**
```json
{
  "metrics": {
    "context_precision": 0.9456,
    "context_recall": 1.0,
    "f1_score": 0.972
  },
  "testset": "rag_eval/evals/datasets/smart_testset_full.json",
  "top_k": 5,
  "num_samples": 10,
  "total_questions_in_testset": 464
}
```

### 3. 查看CSV文件（Excel友好）

```bash
# Windows
start rag_eval\evals\experiments\ragas_metrics_10q.csv

# 或用Excel打开
```

CSV文件包含：
- 指标分数（context_precision, context_recall, f1_score）
- 可用Excel打开进行可视化分析

### 4. 查看详细评估结果

```bash
# 查看每个问题的详细评分
cat rag_eval/evals/experiments/ragas_quick_10q_results.csv
```

**内容示例：**
| question | context_precision | context_recall | ground_truth |
|----------|-------------------|----------------|--------------|
| 问题1 | 0.95 | 1.0 | 标准答案 |
| 问题2 | 0.89 | 1.0 | 标准答案 |

---

## ⏱️ 预期时间参考

| 问题数 | 检索时间 | Ragas评估时间 | 总时间 |
|--------|---------|--------------|--------|
| 10问 | ~1分钟 | ~4分钟 | 5-10分钟 |
| 50问 | ~3分钟 | ~15分钟 | 15-25分钟 |
| 100问 | ~6分钟 | ~30分钟 | 30-50分钟 |
| 464问 | ~30分钟 | ~2小时 | 2-3小时 |

**说明：**
- 检索时间：向量检索 + 结果收集
- Ragas评估时间：GLM API调用（主要耗时）

---

## 🛠️ 运行前检查清单

### 1. 检查Milvus服务

```bash
# 查看Milvus是否在运行
docker ps | findstr milvus

# 或者运行测试连接
python -c "from rag_project.storage.milvus_manager import MilvusManager; m = MilvusManager(); print('Milvus连接成功')"
```

**预期输出：**
```
Connected to Milvus at localhost:19530
Milvus连接成功
```

### 2. 检查Chunks数据

```bash
# 确认chunks文件存在
ls data/all_chunks_with_tags.json

# 查看chunks数量
python -c "import json; data = json.load(open('data/all_chunks_with_tags.json', 'r', encoding='utf-8')); print(f'Chunks数量: {len(data)}')"
```

**预期输出：**
```
Chunks数量: 3422
```

### 3. 检查测试集

```bash
# 确认测试集存在
ls rag_eval/evals/datasets/smart_testset_full.json

# 查看问题数量
python -c "import json; data = json.load(open('rag_eval/evals/datasets/smart_testset_full.json', 'r', encoding='utf-8')); print(f'问题数量: {data[\"num_questions\"]}')"
```

**预期输出：**
```
问题数量: 464
```

### 4. 检查API密钥

确认GLM API密钥配置正确（在脚本中已配置）

---

## 🐛 常见问题排查

### 问题1: Milvus连接失败

**错误信息：**
```
MilvusException: Fail connecting to server on localhost:19530
```

**解决方法：**
```bash
# 1. 检查Docker是否运行
docker ps

# 2. 如果没有运行，启动Milvus
# （根据您的安装方式启动）
```

---

### 问题2: Chunks数据未找到

**错误信息：**
```
错误: Chunks文件不存在
```

**解决方法：**
```bash
# 检查chunks文件路径
ls data/all_chunks_with_tags.json

# 如果不存在，先运行chunking
python chunk_all_documents_no_milvus.py
```

---

### 问题3: 测试集格式错误

**错误信息：**
```
JSON解析错误
```

**解决方法：**
```bash
# 验证测试集格式
python -c "import json; data = json.load(open('rag_eval/evals/datasets/smart_testset_full.json', 'r', encoding='utf-8')); print('格式正确'); print(f'问题数: {len(data[\"questions\"])}')"
```

---

### 问题4: GLM API调用失败

**错误信息：**
```
API密钥无效或API调用超时
```

**解决方法：**
```bash
# 检查API密钥配置
# 在脚本中确认API_KEY和BASE_URL正确

# 检查网络连接
ping open.bigmodel.cn
```

---

## 📊 后台运行完整评估

### Windows后台运行

```bash
# 使用start命令在后台运行
start /B python rag_eval/ragas_glm_fixed.py > eval_log.txt 2>&1
```

### 查看进度

```bash
# 查看日志文件
tail -f eval_log.txt

# 或定期检查
type eval_log.txt | findstr "已处理"
```

### 终止后台任务

```bash
# 查找Python进程
tasklist | findstr python

# 终止进程（使用PID）
taskkill /PID <进程ID>
```

---

## 📈 结果解读指南

### Context Precision（上下文精确度）

**含义：** 检索到的文档中有多少是真正相关的

**范围：** 0-1，越接近1越好

**示例：**
- 0.95 = 95%的检索文档是相关的
- 0.80 = 80%的检索文档是相关的

### Context Recall（上下文召回率）

**含义：** 标准答案中的内容有多少被检索到了

**范围：** 0-1，越接近1越好

**示例：**
- 1.0 = 所有标准答案都被检索到
- 0.85 = 85%的内容被检索到

### F1 Score

**含义：** Precision和Recall的调和平均

**范围：** 0-1，越接近1越好

**计算公式：**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 🎯 推荐评估流程

### 第1步：快速测试（10问）
```bash
python rag_eval/ragas_quick_eval.py --num-questions 10
```
**目的：** 验证系统正常工作

### 第2步：中等评估（50问）
```bash
python rag_eval/ragas_quick_eval.py --num-questions 50
```
**目的：** 获得可靠的结果

### 第3步：完整评估（464问，可选）
```bash
python rag_eval/ragas_glm_fixed.py
```
**目的：** 获得最终评估报告

---

## 📂 输出文件说明

### 1. JSON汇总文件

**位置：** `rag_eval/evals/experiments/ragas_metrics_Nq.json`

**内容：** 指标汇总

```json
{
  "metrics": {
    "context_precision": 0.9456,
    "context_recall": 1.0,
    "f1_score": 0.972
  },
  "num_samples": 10,
  "total_questions_in_testset": 464
}
```

### 2. CSV汇总文件

**位置：** `rag_eval/evals/experiments/ragas_metrics_Nq.csv`

**内容：** 指标汇总（Excel友好）

### 3. 详细结果CSV

**位置：** `rag_eval/evals/experiments/ragas_quick_Nq_results.csv`

**内容：** 每个问题的详细评分

---

## 💡 优化建议

### 如果Precision低（<0.7）
- 检索结果中包含太多不相关文档
- 建议：增加embedding维度、调整Top-K
- 建议：使用重排序（Rerank）

### 如果Recall低（<0.7）
- 很多相关文档没有被检索到
- 建议：增加Top-K数量
- 建议：优化chunking策略
- 建议：检查Milvus索引参数

### 如果评估太慢
- 减少评估问题数
- 使用批量处理
- 增加并发数

---

## 📞 获取帮助

如果遇到问题：

1. 查看日志文件
2. 检查Milvus连接
3. 验证测试集格式
4. 确认API密钥有效

**祝评估顺利！**
