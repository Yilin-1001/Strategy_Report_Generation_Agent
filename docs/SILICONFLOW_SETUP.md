# 硅基流动 Embedding API 快速设置指南

## 📖 简介

使用硅基流动（SiliconFlow）的云API服务进行文本向量化，无需本地GPU，快速启动Agent系统。

**优势**:
- ✅ 启动速度提升 10倍（5-7秒 → <1秒）
- ✅ 无需本地GPU
- ✅ 无需下载大模型文件
- ✅ 按使用量付费，成本低

---

## 🚀 3步快速设置

### 步骤1: 获取API密钥

1. 访问 https://api.siliconflow.cn
2. 注册/登录账号
3. 在控制台获取API密钥（格式: `sk-xxxxxx`）

---

### 步骤2: 配置API密钥

**方式A: 环境变量（推荐）**

```bash
# Windows CMD
set SILICONFLOW_API_KEY=sk-your_api_key_here

# Windows PowerShell
$env:SILICONFLOW_API_KEY="sk-your_api_key_here"

# Linux/Mac
export SILICONFLOW_API_KEY=sk-your_api_key_here
```

**方式B: 配置文件**

编辑 `config/milvus_config.yaml`:

```yaml
embedding:
  mode: "api"
  api_provider: "siliconflow"
  api_key: "sk-your_api_key_here"  # ← 填入你的密钥
```

---

### 步骤3: 测试连接

```bash
python test_siliconflow_api.py
```

预期输出:
```
✓ 找到API密钥: sk-xxxxx...xxxx
✓ 连接成功
✅ 所有测试通过！
```

---

## ✅ 验证配置

确认 `config/milvus_config.yaml` 配置正确:

```yaml
embedding:
  mode: "api"              # ← 必须是 "api"
  api_provider: "siliconflow"  # ← 使用硅基流动
  api_model: "BAAI/bge-m3"     # ← BGE-M3模型
  api_key: ""             # ← 或使用环境变量
  api_timeout: 30
```

---

## 🏃 运行Agent

配置完成后，运行Agent:

```bash
python test_agent_run.py
```

你会注意到启动速度显著提升！

---

## 📊 性能对比

| 项目 | 本地模型 | 硅基流动API |
|------|---------|------------|
| 启动时间 | 5-7秒 | <1秒 |
| GPU要求 | 需要 | 不需要 |
| 模型下载 | ~2GB | 无需下载 |
| 首次embedding | 10-20ms | 100-300ms |
| 批量embedding(10) | 50-100ms | 200-500ms |

**总结**: 硅基流动API牺牲少量推理延迟，换取快速启动和零GPU依赖。

---

## 💰 成本估算

硅基流动BGE-M3定价（参考）:
- ¥0.0001 / 1K tokens

典型使用成本:
- 单次文档索引(1000 chunks): ~¥0.1
- 单次查询(5个检索): ~¥0.0005
- 每日100次查询: ~¥0.05

**提示**: 新用户通常有免费额度

---

## 🔧 支持的模型

硅基流动支持多种embedding模型:

| 模型 | 向量维度 | 配置 |
|------|---------|------|
| BAAI/bge-m3 | 1024 | `api_model: "BAAI/bge-m3"` |
| BAAI/bge-large-zh | 1024 | `api_model: "BAAI/bge-large-zh"` |
| text-embedding-ada-002 | 1536 | `api_model: "text-embedding-ada-002"` |

**注意**: 切换模型后需要更新Milvus集合的向量维度配置。

---

## 🐛 故障排除

### 问题1: API密钥错误

```
ValueError: API key required for siliconflow
```

**解决方案**:
1. 确认环境变量已设置: `echo %SILICONFLOW_API_KEY%`
2. 或在配置文件中设置 `api_key`

### 问题2: 连接超时

```
requests.exceptions.Timeout
```

**解决方案**:
1. 检查网络连接
2. 增加超时时间:
   ```yaml
   embedding:
     api_timeout: 60  # 增加到60秒
   ```

### 问题3: 向量维度不匹配

```
Milvus error: dimension mismatch
```

**解决方案**: 确认Milvus配置中的向量维度与模型一致:
```yaml
# config/milvus_config.yaml
collection:
  dimension: 1024  # BAAI/bge-m3 是 1024 维
```

### 问题4: 配额不足

```
HTTP 402: Payment Required
```

**解决方案**:
1. 检查账户余额
2. 充值或等待免费额度重置

---

## 🔄 切换回本地模型

如果想切换回本地模型:

```yaml
# config/milvus_config.yaml
embedding:
  mode: "local"  # 改回 local
```

---

## 📚 API参考

### 硅基流动Embedding API

**端点**: `POST https://api.siliconflow.cn/v1/embeddings`

**请求**:
```json
{
  "input": "要向量化的文本",
  "model": "BAAI/bge-m3"
}
```

**响应**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],  // 1024维向量
      "index": 0
    }
  ],
  "model": "BAAI/bge-m3",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

---

## 🌍 其他API提供商

`EmbeddingClient` 还支持其他提供商:

### OpenAI

```yaml
embedding:
  mode: "api"
  api_provider: "openai"
  api_key: "sk-your_openai_key"
  api_model: "text-embedding-3-small"
```

### 自定义端点

```yaml
embedding:
  mode: "api"
  api_provider: "siliconflow"
  api_base_url: "https://your-custom-endpoint.com/v1"
```

---

## 📞 获取帮助

- 硅基流动文档: https://api.siliconflow.cn/docs
- API状态页: https://status.siliconflow.cn
- 问题反馈: 查看测试脚本输出 `python test_siliconflow_api.py`

---

**祝你使用愉快！** 🎉
