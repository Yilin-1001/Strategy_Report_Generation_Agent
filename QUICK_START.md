# 🚀 Agent系统快速启动指南

## 前置检查清单

在运行系统之前，请确保以下条件都已满足：

- [ ] ✅ Python 3.10+ 已安装
- [ ] ✅ 所有依赖已安装 (`pip install -r requirements-agent.txt`)
- [ ] ✅ Milvus正在运行 (`docker ps | grep milvus`)
- [ ] ✅ Milvus中有数据 (>3000 chunks)
- [ ] ✅ DeepSeek API Key已设置 (`DEEPSEEK_API_KEY`环境变量)

---

## 🎯 快速启动（3步走）

### 第1步：设置API Key

```bash
# Windows CMD
set DEEPSEEK_API_KEY=sk-your-api-key-here

# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-your-api-key-here"

# Linux/Mac
export DEEPSEEK_API_KEY="sk-your-api-key-here"
```

**获取API Key**: 访问 https://platform.deepseek.com/

### 第2步：验证环境

```bash
cd "E:\02 Final Year Project\RAG Project"

# 检查依赖
python -c "import langgraph; print(f'LangGraph: {langgraph.__version__}')"

# 检查Milvus
python -c "from rag_project.storage.milvus_manager import MilvusManager; mm = MilvusManager('config/milvus_config.yaml'); stats = mm.get_collection_stats(); print(f'Milvus: {stats[\"num_entities\"]} chunks')"

# 检查API Key
python -c "import os; print('API Key:', 'DEEPSEEK_API_KEY' in os.environ)"
```

**期望输出**:
```
LangGraph: 1.0.7
Milvus: 3426 chunks
API Key: True
```

### 第3步：运行系统

#### 选项A：交互模式（推荐实际使用）

```bash
python scripts/run_agent_report.py "生成2024年江西交通投资政策解读报告"
```

**运行说明**:
1. 系统会自动生成5-8章的大纲
2. 每章生成后会暂停，等待你审核
3. 你可以选择：
   - `1` - 通过，进入下一章
   - `2` - 数据不足，重新检索
   - `3` - 逻辑问题，重新分析
   - `4` - 文笔问题，重新润色
4. 全部章节完成后，报告保存到 `output/report.md`

**预计时间**: 15-30分钟（取决于章节数量）

#### 选项B：自动模式（推荐测试）

```bash
python scripts/run_agent_report.py "生成2026年江西交通投资集团政策解读报告" --auto
```

**运行说明**:
1. 系统自动运行，不暂停
2. 所有章节默认通过
3. 适合快速测试功能

**预计时间**: 10-20分钟

#### 选项C：自定义输出路径

```bash
python scripts/run_agent_report.py "生成年度总结报告" --output reports/2024_summary.md
```

---

## 📊 运行过程示例

### 交互模式完整流程

```bash
$ python scripts/run_agent_report.py "生成2024年江西交通投资政策解读报告"

============================================================
🚀 智能体战略报告生成系统
============================================================
📝 请求: 生成2024年江西交通投资政策解读报告
💾 输出: output/report.md
👤 交互模式
============================================================

📋 Report generation started...

✅ Coordinator: 生成大纲完成
   章节列表: ['行业概述', '政策环境分析', '现状评估', '问题与挑战', '战略建议']

--- 第1章开始 ---

📋 准备章节: 行业概述 (第1/5章)

   🔍 Researcher: 检索完成
   查询: ['2024年交通投资概述', '江西交通行业现状', '交通行业发展']
   文档: 20篇

   📊 Analyst: 分析完成
   关键事实: 8条
   商业洞察: 4条

   ✍️  Writer: 草稿生成完成 (956字)

============================================================
📖 人工审核: 行业概述
============================================================

# 行业概述

## 总体概况
2024年以来，中国交通行业保持稳健发展态势。根据国家统计局数据，全年交通基础设施投资达到2.5万亿元，同比增长15.3%...

## 投资规模
2024年交通投资规模持续扩大，主要体现在以下方面...

## 基础设施建设
铁路、公路、水运、民航等基础设施建设全面推进...

## 小结
综上所述，2024年中国交通行业发展态势良好...

============================================================

请选择操作:
  1. approve - 通过，进入下一章
  2. revise:data - 数据不足，重新检索
  3. revise:logic - 逻辑问题，重新分析
  4. revise:writing - 文笔问题，重新润色

请输入选择 (1-4): 1

✅ Chapter approved!

--- 第2章开始 ---

📋 准备章节: 政策环境分析 (第2/5章)
   ... (重复上述流程)

请输入选择 (1-4): 1

✅ Chapter approved!

--- 第3章开始 ---

... (继续直到所有章节完成)

============================================================
📦 Archiver: 报告归档完成

============================================================
✅ 报告生成完成！
============================================================

✅ 报告已保存到: output/report.md
```

---

## ⚠️ 常见错误与解决

### 错误1: `DEEPSEEK_API_KEY环境变量未设置`

```
ValueError: DEEPSEEK_API_KEY环境变量未设置
```

**解决方法**:
```bash
# Windows CMD
set DEEPSEEK_API_KEY=sk-your-api-key-here

# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-your-api-key-here"

# 然后重新运行
python scripts/run_agent_report.py "..."
```

### 错误2: `Milvus连接失败`

```
MilvusException: <MilvusException> (code=1, message=failed to connect to server)
```

**解决方法**:
```bash
# 1. 检查Milvus是否运行
docker ps | grep milvus

# 2. 如果没有运行，启动Milvus
docker-compose up -d

# 3. 等待Milvus启动完成（约30秒）
# 4. 重新运行
python scripts/run_agent_report.py "..."
```

### 错误3: `Milvus中没有数据`

```
⚠️  Warning: Collection is empty
```

**解决方法**:
```bash
# 检查数据量
python -c "from rag_project.storage.milvus_manager import MilvusManager; mm = MilvusManager('config/milvus_config.yaml'); print(mm.get_collection_stats())"

# 如果数据<1000，需要重新索引
python core_scripts/chunk_all_documents.py
```

### 错误4: `检索结果为空`

```
⚠️  Researcher: Retrieved 0 documents
```

**解决方法**:
- 这是正常情况，表示该章节的查询词在知识库中没有匹配
- 选择 `revise:data` 并提供具体的补充数据要求
- 或者扩充知识库内容

### 错误5: `LLM调用超时`

```
TimeoutError: Request timed out
```

**解决方法**:
```yaml
# 编辑 config/agent_config.yaml
llm:
  timeout: 120  # 从60秒增加到120秒
```

---

## 🎛️ 高级配置

### 修改报告章节数量

编辑 `rag_project/agent/nodes/coordinator.py`:

```python
# 修改Prompt中的章节数量要求
prompt = f"""
...
1. 章节数量: 3-10章  # ← 改成你想要的数量
...
"""
```

### 修改每章字数

编辑 `rag_project/agent/nodes/writer.py`:

```python
# 修改Prompt中的字数要求
prompt = f"""
...
7. 篇幅: 1500-2000字  # ← 改成你想要的范围
...
"""
```

### 修改检索策略

编辑 `rag_project/agent/nodes/researcher.py`:

```python
# 修改查询数量
prompt = f"""
请生成5-8个不同的检索查询  # ← 改成你想要的数量
...
"""

# 修改返回文档数量
search_results = retriever.search_multiple(queries, top_k=10)  # ← 改成10
```

---

## 📈 性能优化建议

### 加速报告生成

1. **使用GPU**: 确保 `config/milvus_config.yaml` 中 `device: cuda`
2. **减少章节数**: 5章比8章快40%
3. **自动模式**: 比交互模式快20%（无需等待人工输入）
4. **本地LLM**: 考虑使用本地模型替代DeepSeek API

### 提升报告质量

1. **交互模式**: 逐章审核和调整
2. **扩充知识库**: 添加更多高质量文档
3. **调整温度**: 降低`temperature`使输出更稳定
4. **优化提示词**: 根据实际效果调整系统提示词

---

## 📝 报告示例

生成的报告格式如下：

```markdown
# 江西交通投资集团战略规划报告

**生成时间**: 2026年03月28日

**主题**: 生成2024年江西交通投资政策解读报告

---

# 第一章 行业概述

## 总体概况
2024年以来，中国交通行业保持稳健发展态势...

## 投资规模
...

## 基础设施建设
...

---

# 第二章 政策环境分析

...

---

# 第三章 现状评估

...

---

# 第四章 问题与挑战

...

---

# 第五章 战略建议

...

---

**报告说明**:
本报告由AI智能体系统基于企业知识库自动生成，包含以下章节:

1. 行业概述
2. 政策环境分析
3. 现状评估
4. 问题与挑战
5. 战略建议

**共5章**
```

---

## 🔍 故障排查

### 系统卡在某一步不动

**可能原因**: LLM API调用超时

**解决方法**:
```bash
# Ctrl+C 中断程序

# 检查网络
ping api.deepseek.com

# 增加超时时间
# 编辑 config/agent_config.yaml
llm:
  timeout: 180

# 重新运行
python scripts/run_agent_report.py "..."
```

### 报告内容质量差

**可能原因**:
- 知识库数据不足
- 查询词不准确
- Temperature设置不当

**解决方法**:
1. 使用 `revise:data` 补充数据
2. 扩充知识库
3. 降低 `temperature` 值（0.3-0.5）

### 内存不足

**可能原因**:
- Embeddings模型占用GPU内存
- Milvus占用内存

**解决方法**:
```yaml
# config/milvus_config.yaml
embedding:
  device: cpu  # 从cuda改为cpu
  batch_size: 16  # 从32改为16
```

---

## 📚 相关文档

- **详细架构**: `docs/AGENT_ENGINEERING_GUIDE.md`
- **架构文档**: `docs/AGENT_ARCHITECTURE.md`
- **用户指南**: `docs/AGENT_USER_GUIDE.md`
- **实施总结**: `docs/IMPLEMENTATION_COMPLETE.md`

---

## 🎉 开始使用

现在你已经准备好了！运行以下命令开始：

```bash
cd "E:\02 Final Year Project\RAG Project"
python scripts/run_agent_report.py "生成2024年江西交通投资政策解读报告"
```

**祝你使用愉快！** 🚀
