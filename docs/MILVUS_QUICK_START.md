# 🚀 Milvus 快速启动指南

## 快速开始（推荐）

### Windows 用户

```bash
# 1. 启动 Milvus
start_milvus.bat

# 2. 测试连接
python test_milvus_connection.py

# 3. 运行完整测试
pytest rag_project/tests/ -v

# 4. 停止 Milvus（可选）
stop_milvus.bat
```

---

## 方法一：使用启动脚本（最简单）

### 步骤 1：启动 Milvus
双击运行 `start_milvus.bat` 或在命令行中：
```bash
start_milvus.bat
```

### 步骤 2：验证连接
```bash
python test_milvus_connection.py
```

如果看到 "SUCCESS! Milvus is running and working correctly!" 说明安装成功。

### 步骤 3：运行测试
```bash
# 运行所有测试
pytest rag_project/tests/ -v

# 只运行 Milvus 相关测试
pytest rag_project/tests/test_milvus_manager.py -v
pytest rag_project/tests/test_pipeline.py -v
```

---

## 方法二：使用 Docker Compose

```bash
# 启动
docker-compose up -d

# 查看状态
docker ps

# 查看日志
docker logs milvus-standalone

# 停止
docker-compose down
```

---

## 方法三：使用单条 Docker 命令

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

---

## 📋 检查清单

在启动 Milvus 之前，确保：

- [ ] Docker Desktop 已安装
- [ ] Docker Desktop 正在运行
- [ ] 端口 19530 未被占用
- [ ] 至少 4GB 可用内存

---

## 🔍 验证 Milvus 运行状态

### 方法 1：使用 Python 脚本
```bash
python test_milvus_connection.py
```

### 方法 2：检查端口
```bash
netstat -an | findstr 19530
```

### 方法 3：检查 Docker 容器
```bash
docker ps | findstr milvus
```

### 方法 4：使用 Python
```python
from pymilvus import connections
connections.connect('default', host='localhost', port='19530')
print('Milvus 连接成功!')
```

---

## 🛠️ 常用命令

```bash
# 启动 Milvus
start_milvus.bat

# 停止 Milvus
stop_milvus.bat

# 查看日志
docker logs milvus-standalone

# 重启 Milvus
docker-compose restart

# 完全删除（包括数据）
docker-compose down -v
```

---

## ⚠️ 故障排查

### 问题：Docker 未安装
**解决：** 下载并安装 Docker Desktop
https://www.docker.com/products/docker-desktop

### 问题：Docker 未运行
**解决：** 启动 Docker Desktop 应用程序

### 问题：端口被占用
```bash
# 查看占用进程
netstat -ano | findstr :19530

# 终止进程
taskkill /PID <进程ID> /F
```

### 问题：内存不足
**解决：** 在 Docker Desktop 设置中增加内存分配
- Settings → Resources → Memory
- 建议至少 4GB

---

## 📊 预期结果

启动成功后，您应该看到 3 个容器运行：

```
CONTAINER ID   IMAGE                               STATUS
milvus-etcd        quay.io/coreos/etcd:v3.5.5          Up
milvus-minio       minio/minio:RELEASE.2023-03-...     Up
milvus-standalone  milvusdb/milvus:latest              Up
```

---

## 🎯 下一步

1. ✅ Milvus 运行成功
2. ✅ 运行测试验证：`pytest rag_project/tests/ -v`
3. ✅ 开始使用 RAG 管道处理文档

---

## 📚 相关文档

- [完整安装指南](docs/install_milvus.md)
- [Milvus 官方文档](https://milvus.io/docs)
- [PyMilvus API 参考](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)

---

## 💡 提示

- 首次启动 Milvus 需要下载 Docker 镜像，可能需要几分钟
- Milvus 数据存储在 `./volumes/milvus` 目录
- 生产环境建议使用 Docker Compose 方式部署
- 开发完成后记得停止 Milvus 以节省资源
